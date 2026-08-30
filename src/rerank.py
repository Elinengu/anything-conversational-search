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

from src.index import CatalogIndex
from src.state import DialogState
from src.facets import (
    extract,
    extract_query_facets,
)
from src.text import terms


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

    # Rescore the whole retrieval pool (RetrievalConfig.pool_size), not a prefix -
    # ~12% of cluster-target sessions had the target in the pool but past rank 200,
    # where it was left in bm25 order and the span signal never applied.
    depth: int = 300


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


def rerank(
    index: CatalogIndex,
    state: DialogState,
    candidates: list[tuple[str, float]],
    config: RerankConfig | None = None,
) -> list[tuple[str, float]]:
    config = config or RerankConfig()
    if not config.enabled or not candidates:
        return candidates

    spans = state.query_spans()
    head = candidates[: config.depth]
    tail = candidates[config.depth :]
    if not spans:
        return candidates
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
        )
        scored.append((parent_asin, total))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored + tail
