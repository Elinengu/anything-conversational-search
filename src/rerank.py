"""S6 - reranking the candidate pool.

Retrieval already places the target in the pool almost always; what costs score
is its *position*, because MRR carries 30% of the technical score and the session
ends at the first hit, freezing whatever rank was achieved.

The dominant signal is verbatim span coverage. Constraints the customer discloses
are copied from the target product's own metadata, so a candidate whose text
literally contains "stainless steel band" is far more likely to be the target than
one that merely shares those tokens. Popularity is a tie-break - but the tie-break
is where the remaining headroom lives, so it is no longer a token weight: dissecting
every near-miss session showed all lexical signals exactly tied 33/33 between the
target and the impostor above it, with the tie broken by the retrieval score
(wrong 33/33 - BM25 length normalization favours thin listings) while popularity
pointed at the target 31/33 (the target is a real purchase, hence a reviewed,
documented product). See docs/team/rerank_signals.md §10.

Weighting spans by pool-local rarity was implemented and measured on the theory
that "buckle closure" should count for less than "two row stitch" among belts. It
changed the dev score by 0.0002 and the holdout not at all, because a pool
retrieved by those same terms has little rarity spread left to exploit, so it was
removed rather than kept as a dead option.

Pair spans complement the fragments rather than replace them. Fragment
coverage asks "does this product mention 90% cotton at all?"; the message's
actual evidence is "does it say that about heather grey?" - the colon/comma
structure that ``constraint_spans`` severs. Three fragment de-weighting
variants (sum/rootk, mean, best-of-group) were prototyped and all measured flat
or worse, because the 8-of-8-vs-5-of-8 fragment gradient is load-bearing; the
win is adding the intact association as its own term, which is what finally
moved the homogeneous-cluster bucket (MRR 0.431 -> 0.478) where fragments
saturate by construction.

Negative facet evidence (``_facet_conflicts``) penalises a candidate that
resolves an attribute the customer constrained and whose text never mentions the
stated value. Positive signals cannot do this job: a black-only shirt matches
"cotton shirt" spans exactly as well as a grey one. Contradiction is judged
against ``focused_text`` rather than the full history, which costs 0.047 MRR on
the adversarial override bucket. The reason is *not* staleness, as this file
long claimed - it is that ``focused_text`` drops turn 1, whose category framing
extracts as a style/use_case constraint. See the note at the call site.

Two more signals were measured and not shipped. Profile-weighted facet
agreement (extra credit on attributes named by ``preference_tags``) gained
+0.0013 on dev but lost 0.0008 on holdout - the customer already discloses
their profiled preferences verbatim in-session, so the profile adds noise, not
information. A budget/price closeness term was rejected before implementation:
the evaluator's own card builder emits a budget constraint for 0.45% of catalog
products (0 of 200 public sessions), so the signal could essentially never
fire. Both measurements are recorded in docs/team/rerank_signals.md.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.index import CatalogIndex
from src.state import DialogState
from src.facets import (
    extract,
    extract_query_facets,
)
from src.text import terms

if TYPE_CHECKING:  # avoid a hard numpy dependency on the BM25-only path
    from src.embed import EmbeddingIndex
    from src.llm import LLMReranker


@dataclass
class RerankConfig:
    enabled: bool = True
    span_weight: float = 1.0
    # Longer spans are rarer and therefore stronger evidence.
    length_bonus: float = 0.12
    retrieval_weight: float = 1.0
    # Raised 0.02 -> 0.4 after the near-miss anatomy (rerank_signals.md §10): in
    # the tie-break regime that holds all remaining public headroom, popularity
    # picks the target 31/33 but at 0.02 was drowned 50:1 by the retrieval
    # score, which picks the impostor 33/33. Measured at 0.4: dev 0.9268 ->
    # 0.9418, holdout 0.9096 -> 0.9136, public 0.9199 -> 0.9305, hard 0.7981 ->
    # 0.8020 with a converted miss (hit 0.885 -> 0.896) - every split up, public
    # 200/200 kept. 0.1 / 0.3 / 0.5 are all >= baseline on all four splits, so
    # 0.4 sits mid-plateau with both neighbours measured. The coordinate-ascent
    # dev argmax (tools/fit_weights.py) wanted 0.8 with retrieval at 0.1 - that
    # scores higher on dev/holdout/public but regresses the adversarial set
    # 0.7981 -> 0.7824, whose targets are deliberately thin and unreviewed; the
    # one-weight change is the qualifier under the no-bucket-regresses rule.
    popularity_weight: float = 0.4
    # Candidate facets matching the customer's stated facets (material, colour, ...).
    facet_weight: float = 0.3
    category_weight: float = 0.4
    # The customer's opening names the target's category-path *tail* (its two
    # most specific levels), so a candidate whose own tail is fully named in the
    # opening is a far stronger match than one that merely shares an ancestor
    # like "Novelty" - see _tail_match. 0.6-1.5 score identically on dev and
    # holdout; this sits mid-plateau rather than at either split's argmax.
    tail_weight: float = 0.8
    # Penalty per stated facet value the candidate contradicts (customer said
    # "grey", the candidate resolves a colour and "grey" appears nowhere in its
    # text). Positive signals saturate inside homogeneous clusters - every
    # bucketmate matches the same spans and facets - so contradiction is the
    # only evidence left that separates them. See _facet_conflicts for the
    # guards that keep this conservative. Measured: dev 0.9207->0.9224,
    # holdout +0.001, hard set +0.0003 with no bucket regressing; 0.4 and 0.8
    # score alike, so this sits at the low end of the plateau (a penalty term
    # earns the smallest weight that works). 0.0 disables.
    facet_conflict_weight: float = 0.4
    # Association-preserving spans (state.query_pair_spans): "heather grey 90
    # cotton 10 polyester" as one unit instead of three fragments, so the
    # composition must be stated about that colour. Flat 1.0 per matched pair -
    # the pair's evidential value is the intact association, not its length.
    # Measured: public 0.9149->0.9159, hard 0.7917->0.7944, with the gain
    # concentrated in the homogeneous-cluster bucket (MRR 0.431->0.478) where
    # fragments saturate. 0.4-1.5 score identically; mid-plateau. 0.0 disables.
    pair_weight: float = 0.8

    # Dual-track routing (opt-in, off unless AgentConfig.use_router routes a
    # "buying" track and hands rerank() a track-specific config with this set).
    # A decided buyer has stated a hard requirement, so a candidate that
    # *positively contradicts* an authoritative stated facet is not a ranking
    # question - it is not the target. When True and track == "buying", such
    # candidates are dropped from the head instead of merely penalised by
    # facet_conflict_weight. Same three-part conflict test as _facet_conflicts
    # (silence is never a contradiction), judged against focused_text() so an
    # overridden-away or misparsed constraint cannot evict the target. Browsing
    # keeps every candidate and the soft penalty. Default False = today's
    # behaviour on every track.
    hard_filter: bool = False

    # Rescore the whole retrieval pool (RetrievalConfig.pool_size), not a prefix -
    # ~12% of cluster-target sessions had the target in the pool but past rank 200,
    # where it was left in bm25 order and the span signal never applied.
    depth: int = 300

    # Dense sentence-embedding cosine term (bge-small, src/embed.py). Every other
    # S6 signal is exact-token: span coverage, facet agreement, category overlap
    # all go to zero the moment the customer says "cowhide" instead of "leather".
    # This term is the only one that scores meaning. 0.0 -> off, and every
    # existing config/test keeps the exact behaviour. Needs `embed` + `qvec`
    # passed to rerank() (AgentConfig loads the index when this is > 0); missing
    # artifact / deps -> silently 0.
    dense_weight: float = 0.0
    #: Which text is encoded as the query vector: "full" reuses the agent's
    #: full_text() vector (free - encoded once per turn), "spans" encodes the
    #: disclosed constraint spans only, "blend" averages the two, "slots" encodes
    #: state.authoritative_text() - the state machine's compact active-slot
    #: query, no simulator boilerplate. Step 3.3: untested before this config -
    #: shorter and cleaner, but shorter can also mean less to go on.
    dense_query: str = "full"
    #: Rescore the head even when no verbatim span was disclosed. Only meaningful
    #: with dense_weight > 0 - the degenerate-card / paraphrased-opening lever
    #: where span coverage is a frozen no-op.
    rescore_without_spans: bool = False
    #: Step 3.2: fire dense_weight only when state.over_general (pool_size>=100,
    #: pool_entropy>=0.97, leader_margin<0.05 - src/state.py) says lexical
    #: matching has stopped discriminating the live pool. False (default) is
    #: today's behaviour: dense_weight applies unconditionally every turn - the
    #: one measurement on record for that (branch_state_encoder_eval_changes.md)
    #: is net -0.016 on the 21-session generic tail. See _dense_gate_open().
    dense_gate_over_general: bool = False
    #: Step 3.2, informed by that same measurement: buying (+0.0103) and
    #: intent_override (+0.0396) improved, browsing collapsed (-0.0900) - the
    #: embedding may be diluting an already-strong lexical signal specifically on
    #: the track state-encoder-eval's policy fix made richest. True withholds
    #: dense_weight on intent_track=="browsing" regardless of dense_gate_over_general.
    #: A no-op alone if dense_weight would not otherwise fire.
    dense_gate_exclude_browsing: bool = False

    # Tier-2 opt-in layer (docs/team/ideas_to_integrate_llm.md #3): a remote LLM
    # (DeepSeek, src/llm.py) reorders the top ``llm_depth`` lexically-ranked
    # candidates once per turn. Fused with the lexical score exactly like
    # dense_weight is - never a replacement, and a network failure, timeout or
    # malformed reply is caught in LLMReranker.rank() and returns None, which
    # this file treats as "no opinion" and leaves the lexical order untouched.
    # 0.0 (default) never calls llm.rank() at all - see _llm_gate_open below and
    # rerank()'s call site. Needs an ``llm`` (LLMReranker) passed to rerank()
    # with ``.available`` True (LLMConfig.enabled plus an actual API key); the
    # Agent builds one from AgentConfig.llm and passes it through.
    llm_weight: float = 0.0
    # Gate: only call the model when the *previous* turn's observed pool was
    # this undecided (state.leader_margin, src/state.py) - the same live
    # pool-shape signal Step 3.2 gates the dense term with. A pool with a clear
    # lexical leader has nothing to gain and a nondeterministic call to lose;
    # an ambiguous one (low leader_margin) is exactly where a semantic re-read
    # of the candidates can break a tie exact-token matching cannot see.
    # <= 0.0 disables the gate (always eligible, subject to llm_weight itself).
    llm_gate_margin: float = 0.05
    # How many of the already lexically-sorted head candidates go into the
    # prompt. Bounds latency, cost and prompt size; the model can only ever
    # reorder within this window; it never promotes a candidate ranked below it.
    llm_depth: int = 8


def _dense_similarities(
    embed: "EmbeddingIndex",
    state: DialogState,
    spans: list[str],
    qvec: Any,
    mode: str,
    asins: list[str],
) -> dict[str, float]:
    """Cosine of every head candidate against the chosen query vector."""
    if mode == "full" and qvec is not None:
        query_vec = qvec
    elif mode == "spans":
        query_vec = embed.encode_query(" ".join(spans)) if spans else qvec
    elif mode == "blend":
        import numpy as np

        parts = [
            v for v in (qvec, embed.encode_query(" ".join(spans)) if spans else None)
            if v is not None
        ]
        if not parts:
            return {}
        stacked = np.mean(np.stack(parts), axis=0)
        norm = float(np.linalg.norm(stacked))
        query_vec = stacked / norm if norm > 0.0 else stacked
    elif mode == "slots":
        # The state machine's compact active-slot query - no simulator
        # boilerplate, unlike full_text(). Encoded fresh (not the cached qvec,
        # which is always full_text()); falls back to qvec if there is nothing
        # active yet (authoritative_text() itself falls back to focused_text(),
        # so this is only reached pre-turn-1).
        text = state.authoritative_text()
        query_vec = embed.encode_query(text) if text else qvec
    else:
        query_vec = qvec
    if query_vec is None:
        return {}
    return embed.similarities(query_vec, asins)


def _popularity(product: dict) -> float:
    """Small, bounded prior in [0, 1]. Tie-break only - see module docstring."""
    rating = product.get("average_rating") or 0.0
    count = product.get("rating_number") or 0
    try:
        return (float(rating) / 5.0) * min(1.0, math.log10(float(count) + 1.0) / 4.0)
    except (TypeError, ValueError):
        return 0.0


def _facet_agreement(
    customer_facets: dict[str, str],
    product_facets: dict[str, str],
) -> float:
    """
    Count matching facet values between
    customer constraints and product facets.
    """

    score = 0.0

    for key, value in customer_facets.items():

        if product_facets.get(key) == value:
            score += 1.0

    return score


# The only synonym pair in the facet vocabularies (src/facets.py). Without it a
# customer's "gray" would count as contradicted by a product spelling it "grey".
_VALUE_ALIASES = {"grey": ("grey", "gray"), "gray": ("gray", "grey")}


def _facet_conflicts(
    customer_facets: dict[str, str],
    product_facets: dict[str, str],
    product_text: str,
) -> float:
    """Count customer-stated facet values the candidate contradicts.

    A conflict needs all three of:
      1. the customer stated a value for the attribute;
      2. the candidate resolves that attribute too - silence is never punished,
         missing data is not disagreement;
      3. the stated value (or an alias) appears nowhere in the candidate's text.
         extract() keeps only the first vocabulary match, so a "black/grey
         reversible" product extracts color=black yet still contains "grey";
         the substring check keeps such multi-value products unpunished.
    """
    conflicts = 0.0
    for key, value in customer_facets.items():
        if key not in product_facets:
            continue
        aliases = _VALUE_ALIASES.get(value, (value,))
        if any(
            re.search(rf"\b{re.escape(alias)}\b", product_text)
            for alias in aliases
        ):
            continue
        conflicts += 1.0
    return conflicts


# Store-wide wrapper levels that appear on nearly every product and therefore
# say nothing about which subtree a product lives in.
GENERIC_CATEGORY_PARTS = {
    "clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry",
}


def _tail_match(
    state: DialogState,
    product: dict,
) -> float:
    """Alignment between the opening message and the candidate's category tail.

    The simulated customer opens with the target's coarse category, which the
    evaluator builds from the two most specific levels of the target's category
    path ("Novelty > Women" -> "I'm looking for Novelty Women"). Ancestor
    overlap alone cannot use this: a deep candidate ("... > Novelty > Women >
    Tops & Tees > T-Shirts") shares every ancestor the target has, yet its own
    tail ("Tops & Tees T-Shirts") goes unmentioned in the opening. Scoring the
    tail separates the two - and it is matched by token containment, not by
    parsing the opening template, so paraphrased private-set wording still
    works. In the one public-set miss (public_0020) this cut a 159-way rerank
    tie down to the handful of candidates on the right leaf.
    """
    opening_terms = set(
        terms(state.opening, drop_boilerplate=True)
    )
    if not opening_terms:
        return 0.0

    cleaned: list[str] = []
    for value in product.get("categories", []):
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in GENERIC_CATEGORY_PARTS:
                cleaned.append(part)

    score = 0.0
    for part in cleaned[-2:]:
        part_tokens = set(terms(part))
        if part_tokens and part_tokens <= opening_terms:
            score += 1.0
    return score


def _category_match(
    state: DialogState,
    product: dict,
) -> float:
    """
    Category agreement between
    opening query and candidate product.
    """

    opening_terms = set(
        terms(state.opening, drop_boilerplate=True)
    )

    categories = {
        str(value).lower()
        for value in product.get("categories", [])
    }

    if not opening_terms:
        return 0.0

    score = 0.0

    for category in categories:

        category_tokens = set(
            terms(category)
        )

        if opening_terms.intersection(category_tokens):
            score += 1.0

    return score


def _dense_gate_open(state: DialogState, config: RerankConfig, track: str | None) -> bool:
    """True -> dense_weight applies this turn (RerankConfig.dense_gate_*, Step 3.2).

    Both gates default False, and default-False -> always True (today's
    unconditional behaviour, byte-identical). ``track`` (the router's live
    per-turn track, e.g. from ``_route_for`` in starter/agent.py) takes
    precedence when a caller passes it explicitly; ``state.intent_track``
    (DialogState's own default is "browsing" - src/state.py) is only consulted
    as a fallback when ``track is None``, never as an additional veto - an
    explicit track="buying" must not be overridden by that default.
    """
    if config.dense_gate_exclude_browsing:
        effective_track = track if track is not None else state.intent_track
        if effective_track == "browsing":
            return False
    if config.dense_gate_over_general and not state.over_general:
        return False
    return True


def _llm_gate_open(state: DialogState, config: RerankConfig) -> bool:
    """True -> the LLM layer is eligible to fire this turn (RerankConfig.llm_gate_margin).

    Mirrors ``_dense_gate_open``: consults the *previous* turn's observed pool
    (``state.leader_margin``) since this turn's pool is exactly what is being
    computed right now. ``llm_gate_margin <= 0.0`` disables the gate outright
    (always eligible) - the caller still needs ``llm_weight > 0.0`` and an
    available ``llm`` for anything to actually happen.
    """
    if config.llm_gate_margin <= 0.0:
        return True
    return state.leader_margin < config.llm_gate_margin


def _llm_rerank(
    scored: list[tuple[str, float]],
    state: DialogState,
    index: CatalogIndex,
    config: RerankConfig,
    llm: "LLMReranker",
) -> list[tuple[str, float]]:
    """Fuse the model's reordering of the top ``llm_depth`` candidates into the
    lexical score. ``llm.rank()`` returning ``None`` (any failure at all - see
    src/llm.py) is a no-op: ``scored`` comes back exactly as it went in, so a
    flaky call degrades to precisely the ``llm_weight=0.0`` behaviour.
    """
    depth = max(2, config.llm_depth)
    top = scored[:depth]
    rest = scored[depth:]
    items = []
    for asin, _score in top:
        product = index.products.get(asin)
        items.append({"asin": asin, "text": product["text"] if product else ""})

    try:
        order = llm.rank(state.authoritative_text(), items)
    except Exception:
        order = None
    if not order:
        return scored

    rank_of = {asin: position for position, asin in enumerate(order)}
    denom = max(1, len(top) - 1)
    boosted: list[tuple[str, float]] = []
    for position, (asin, score) in enumerate(top):
        # An asin the model dropped keeps its own lexical position as its
        # implied rank, rather than being punished to the back of the window.
        model_rank = rank_of.get(asin, position)
        bonus = (len(top) - 1 - model_rank) / denom
        boosted.append((asin, score + config.llm_weight * bonus))
    boosted.sort(key=lambda item: (-item[1], item[0]))
    return boosted + rest


def rerank(
    index: CatalogIndex,
    state: DialogState,
    candidates: list[tuple[str, float]],
    config: RerankConfig | None = None,
    track: str | None = None,
    embed: "EmbeddingIndex | None" = None,
    qvec: Any = None,
    llm: "LLMReranker | None" = None,
) -> list[tuple[str, float]]:
    config = config or RerankConfig()
    if not config.enabled or not candidates:
        return candidates

    spans = state.query_spans()
    head = candidates[: config.depth]
    tail = candidates[config.depth :]

    dense_active = (
        config.dense_weight > 0.0
        and embed is not None
        and getattr(embed, "available", False)
        and _dense_gate_open(state, config, track)
    )
    if not spans and not (dense_active and config.rescore_without_spans):
        return candidates

    sims: dict[str, float] = {}
    dense_lo, dense_span = 0.0, 1.0
    if dense_active:
        try:
            sims = _dense_similarities(
                embed, state, spans, qvec, config.dense_query,
                [asin for asin, _ in head],
            )
        except Exception:
            sims = {}
        if sims:
            # Min-max over the head so the term spans [0, 1] like span coverage -
            # raw catalog cosines sit in a narrow ~[0.55, 0.8] band (every pool
            # member is already the right category), so dividing by the max would
            # leave almost no spread for dense_weight to act on.
            values = sims.values()
            dense_lo = min(values)
            dense_span = (max(values) - dense_lo) or 1.0

    pairs = state.query_pair_spans() if config.pair_weight else []

    # Normalise retrieval scores so the two signals combine on one scale.
    top_score = max(score for _asin, score in head) or 1.0

    # Per-session quantities, computed once rather than per candidate.
    customer_facets = extract_query_facets(state.full_text())
    # Judged against focused_text(), which on an override session means "turn 1
    # excluded". That is what this line actually buys, and the original
    # "discarded value" reading of it was wrong: measured over all 30 public
    # override sessions, full history picks a value the post-override turns
    # contradict in *zero* of them. The one session that regresses
    # (hard_generic_override_08) conflicts on turn 1 - coarse_category() emits
    # the target's two most specific category levels, and those are drawn from
    # the same vocabulary as the style/use_case facets, so "I'm looking for
    # Pants Casual" extracts style=casual and then punishes every candidate
    # whose own style resolves to something else.
    #
    # The asymmetry is deliberate, not an oversight. Excluding turn 1 from
    # conflict scoring *everywhere* was measured and is worse (public 0.9159 ->
    # 0.9150, holdout 0.9048 -> 0.9035): the category framing is a real
    # constraint, and the impostors it demotes outweigh the 8 targets it
    # wrongly penalises. Keeping turn 1 on override sessions is also worse
    # (hard 0.7944 -> 0.7920). Excluding it on override sessions only - what
    # focused_text() happens to do here - is the best of the three.
    # docs/team/rerank_signals.md records all four variants.
    authoritative_facets = extract_query_facets(state.focused_text())

    # Dual-track hard filter: on the buying track, a candidate that positively
    # contradicts an authoritative stated facet is banished to the very bottom of
    # the list rather than merely penalised by facet_conflict_weight. Guarded
    # exactly like the soft penalty (_facet_conflicts): needs the stated value
    # AND the attribute resolved on the candidate AND the value absent from its
    # text - silence is never a contradiction. Banished candidates are appended
    # after the tail, not removed, so retrieval recall is never lost; in practice
    # a shown slate is <= 10, so this makes the contradiction decisive without a
    # recall cliff. Browsing keeps every candidate rankable on its other signals.
    banished: list[tuple[str, float]] = []
    if track == "buying" and config.hard_filter and authoritative_facets:
        survivors: list[tuple[str, float]] = []
        for asin, score in head:
            product = index.products.get(asin)
            if product is not None and _facet_conflicts(
                authoritative_facets, extract(product), product["text"]
            ) > 0.0:
                banished.append((asin, score))
            else:
                survivors.append((asin, score))
        if survivors:
            head = survivors
        else:
            # Every candidate contradicts - the extraction is almost certainly
            # wrong. Rank them all normally rather than banish the whole pool.
            banished = []

    scored: list[tuple[str, float]] = []
    for parent_asin, retrieval_score in head:
        product = index.products.get(parent_asin)
        if product is None:
            scored.append((parent_asin, 0.0))
            continue
        # Word-bounded matching: text is token-joined, so padding both sides
        # anchors every span at token edges ("90 cotton" no longer matches
        # "190 cotton").
        padded = f" {product['text']} "
        text = product["text"]
        coverage = 0.0
        for span in spans:
            if f" {span} " in padded:
                coverage += 1.0 + config.length_bonus * len(span.split())
        pair_coverage = 0.0
        for span in pairs:
            if f" {span} " in padded:
                pair_coverage += 1.0
        product_facets = extract(product)
        facet_score = _facet_agreement(
            customer_facets,
            product_facets,
        )
        conflict_score = _facet_conflicts(
            authoritative_facets,
            product_facets,
            text,
        )
        category_score = _category_match(
            state,
            product,
        )
        tail_score = _tail_match(
            state,
            product,
        )
        total = (
            config.span_weight * coverage
            + config.pair_weight * pair_coverage
            + config.retrieval_weight * (retrieval_score / top_score)
            + config.popularity_weight * _popularity(product)
            + config.facet_weight * facet_score
            + config.category_weight * category_score
            + config.tail_weight * tail_score
            - config.facet_conflict_weight * conflict_score
            + config.dense_weight * ((sims.get(parent_asin, dense_lo) - dense_lo) / dense_span)
        )
        scored.append((parent_asin, total))

    scored.sort(key=lambda item: (-item[1], item[0]))

    if (
        config.llm_weight > 0.0
        and llm is not None
        and getattr(llm, "available", False)
        and scored
        and _llm_gate_open(state, config)
    ):
        scored = _llm_rerank(scored, state, index, config, llm)

    return scored + tail + banished
