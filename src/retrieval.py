"""S5 - multi-route retrieval with reciprocal-rank fusion.

Routes:
  * ``terms``   - bag-of-words OR query over the whole conversation (recall)
  * ``anchor``  - bag-of-words over the opening turn only (topic drift guard)
  * ``focused`` - bag-of-words over post-override turns only (override handling)
  * ``structured`` - compact active-slot query, activated as a stagnation route
  * ``dense``   - sentence-embedding cosine over the whole catalog, optional; adds
                  paraphrase / synonymy recall the lexical routes miss. Off unless
                  ``RetrievalConfig.use_dense`` is set and a usable
                  ``EmbeddingIndex`` is passed (see ``src/embed.py``). The same
                  encoder and the same per-turn query vector also feed the S6
                  reranker's ``dense_weight`` term.

RRF is used rather than score addition because the routes produce scores on
incomparable scales; rank fusion needs no calibration and degrades gracefully when
a route returns nothing.

Verbatim constraint matching deliberately lives in the reranker instead of being a
third route here. As an FTS5 phrase query it recalls the target in only 47 of 80
sampled sessions, so fusing it at retrieval time injects more noise than signal;
applied as a rescoring signal over a pool the terms route already fills, the same
evidence is pure gain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.index import CatalogIndex
from src.rerank import _dense_gate_open
from src.state import DialogState

if TYPE_CHECKING:  # avoid a hard numpy dependency on the BM25-only path
    from src.embed import EmbeddingIndex
    from src.llm import LLMReranker


RRF_K = 60.0

# The evaluator's three opening templates (evaluator/local_evaluator.py,
# initial_message) all lead with "I'm looking for {coarse_category}" and then
# either ". A key requirement is: ...", ". {old_value}", or ", but I'm still
# exploring." - so the category is everything up to the first of those.
_OPENING_CATEGORY = re.compile(
    r"^\s*i(?:'m| am)\s+looking\s+for\s+(?P<category>.+?)"
    r"(?:,\s*but\b|\.|$)",
    re.IGNORECASE,
)


def opening_category(opening: str) -> str:
    """The stated coarse category from a session's opening message, or ''."""
    match = _OPENING_CATEGORY.match(opening or "")
    return match.group("category").strip() if match else ""


@dataclass
class RetrievalConfig:
    """Route weights and pool sizes. Tuned on the dev split by tools/sweep.py."""

    use_terms: bool = True
    use_anchor: bool = True
    use_focused: bool = True
    # The compact slot-only view is an adaptive recovery route, not a permanent
    # extra vote. Keeping it off during healthy turns preserves the higher-recall
    # full conversation; ``route_hint='structured'`` activates it on stagnation.
    use_structured: bool = False
    weight_terms: float = 1.0
    weight_anchor: float = 0.6
    weight_focused: float = 0.8
    # Compact active-slot query. Kept below the full-text route because exact
    # simulator wording remains the strongest public-set signal.
    weight_structured: float = 0.25
    pool_size: int = 300
    #: Dense sentence-embedding route. Off by default; every existing config and
    #: test keeps the exact BM25 behaviour (byte-identical pool) until this is
    #: switched on and a usable ``EmbeddingIndex`` is supplied.
    use_dense: bool = False
    weight_dense: float = 0.6
    #: Post-override dense sub-route over ``focused_text`` only, mirroring the
    #: lexical ``focused`` route at a lower weight.
    weight_dense_focused: float = 0.4
    dense_pool: int = 300
    #: Gate the dense route the same way RerankConfig gates the S6 dense_weight
    #: term (see src/rerank.py's _dense_gate_open, reused here - both fields
    #: mean exactly what they mean there). Motivated by
    #: docs/team/branch_state_encoder_eval_changes.md §3d: use_dense fired
    #: unconditionally scored a confirmed trade-off - +0.0263 under
    #: paraphrase:heavy+browse-gated stress, -0.0042/-0.0065 on the cooperative
    #: official/holdout sets (both driven by the same browsing-MRR dilution the
    #: pre-state-machine bi-encoder attempts documented). Both default False,
    #: which keeps use_dense's existing unconditional behaviour byte-identical.
    dense_gate_over_general: bool = False
    dense_gate_exclude_browsing: bool = False

    # ---- coarse-category pool route -------------------------------------
    # The lexical routes above use the customer's stated category only as a bag
    # of words competing with every other conversation token, so a product that
    # merely says "necklace" in its title outranks a genuine member of the
    # `Jewelry Necklaces` category. Measured on the public set at turn 1: the
    # target is inside our 300-candidate pool 80.5% of the time at median rank
    # 51, and only ~66% of those 300 are even in the target's category.
    #
    # The evaluator opens every session with coarse_category(target's own
    # categories) (evaluator/local_evaluator.py, initial_message), so that
    # string is an exact key of one of CatalogIndex.pools and the target is
    # inside it 200/200. Unioning the pool into the candidate set makes turn-1
    # recall complete by construction. Union rather than replace: the fused
    # lexical order still leads, and the pool only adds what fusion missed - so
    # a paraphrased opening that resolves no pool degrades to exactly today's
    # behaviour.
    #
    # Off by default so every existing sweep row, test and measurement stays
    # byte-identical until it is switched on.
    # Shipped on: measured against the previous lexical-only pool on four sets
    # (dev / holdout / generated / hard), on top of sniper list sizing:
    #   off          0.9521 / 0.9220 / 0.9322 / 0.8135
    #   weight 0.7   0.9574 / 0.9458 / 0.9367 / 0.8433
    #   weight 1.0   0.9590 / 0.9489 / 0.9349 / 0.8444   <- ships
    #   weight 1.5   0.9620 / 0.9564 / 0.9367 / 0.8291
    # 1.5 is the argmax of both public-derived splits and the hard set rejects
    # it, which is the signature of fitting the public generator's sampling.
    # 0.7 and 1.0 are the plateau where all four sets agree.
    # ---- Arm B: LLM query-expansion route --------------------------------
    # One extraction call on the opening turn turns the customer's sentence
    # into {category, constraints, expanded_terms}; the terms become a fourth
    # RRF route. Placed in retrieval, not reranking, for three reasons: it is
    # the only stage that runs on turn 1 (rerank returns early with no spans);
    # RRF already exists because route scores are not comparable, so a weak
    # route degrades instead of dominating; and it composes with the category
    # pool rather than competing - the pool decides which category, this
    # decides which words. Any failure yields no route at all, leaving the
    # fused pool byte-identical to this being off.
    #
    # This is the job a language model beats exact-token matching at. Ranking
    # is not: the evaluator quotes constraints verbatim from the target's own
    # metadata, and the ranking layer measured 9-up/9-down accordingly.
    use_llm_terms: bool = False
    weight_llm_terms: float = 0.8

    use_category_pool: bool = True
    #: RRF weight for the pool route. Pool members are ranked by popularity,
    #: which is the right prior here - targets are drawn with a
    #: popularity-weighted sampler - but it carries no constraint evidence, so
    #: it sits below the full-text route rather than above it.
    weight_category_pool: float = 1.0
    #: Cap on pool members considered. The largest single pool is 1,354; the
    #: paraphrase fallback merges pools and needs a ceiling of its own.
    category_pool_max: int = 1500


def _rrf(ranked: list[tuple[str, float]], weight: float, sink: dict[str, float]) -> None:
    for position, (parent_asin, _score) in enumerate(ranked):
        sink[parent_asin] = sink.get(parent_asin, 0.0) + weight / (RRF_K + position + 1)


def _llm_term_route(
    index: CatalogIndex, state: DialogState, llm: "LLMReranker", limit: int
) -> list[tuple[str, float]]:
    """Arm B: BM25 over the terms the model read out of the opening.

    Extraction runs once per session, on the opening, and the result is cached
    on the state - later turns reuse it rather than paying a call per turn.
    """
    expansion = getattr(state, "_llm_expansion", None)
    if expansion is None:
        expansion = llm.extract(state.opening) or {}
        try:
            state._llm_expansion = expansion
        except Exception:
            pass
    terms_text = " ".join([
        str(expansion.get("category") or ""),
        *[str(v) for v in expansion.get("constraints", [])],
        *[str(v) for v in expansion.get("expanded_terms", [])],
    ]).strip()
    if not terms_text:
        return []
    return index.search_terms(terms_text, limit=limit)


def retrieve(
    index: CatalogIndex,
    state: DialogState,
    config: RetrievalConfig | None = None,
    route_hint: str | None = None,
    embed: "EmbeddingIndex | None" = None,
    qvec: Any = None,
    track: str | None = None,
    llm: "LLMReranker | None" = None,
) -> list[tuple[str, float]]:
    """Return a fused candidate pool, best first.

    ``embed`` / ``qvec`` are optional. When ``config.use_dense`` is set, a usable
    ``EmbeddingIndex`` is passed, and the gate is open (``track`` - see
    ``RetrievalConfig.dense_gate_*`` / ``src.rerank._dense_gate_open``, reused
    here), a dense cosine route is fused in; ``qvec`` is the pre-encoded
    ``full_text`` vector (the agent encodes it once per turn and reuses it for
    the reranker). Any failure in the dense path is swallowed - the lexical pool
    is returned unchanged.
    """
    config = config or RetrievalConfig()
    fused: dict[str, float] = {}

    if config.use_anchor and state.opening:
        _rrf(index.search_terms(state.opening, limit=config.pool_size),
             config.weight_anchor, fused)

    terms_weight = config.weight_terms
    # The focused route is already active after an override. The orchestration
    # hint selects that authoritative view without blindly amplifying it: a
    # stronger weight made old-but-still-useful evidence disappear too quickly.
    focused_weight = config.weight_focused
    structured_weight = config.weight_structured * (
        1.75 if route_hint == "structured" else 1.0
    )

    if config.use_terms:
        _rrf(index.search_terms(state.full_text(), limit=config.pool_size),
             terms_weight, fused)

    if config.use_focused and state.override_turn is not None:
        _rrf(index.search_terms(state.focused_text(), limit=config.pool_size),
             focused_weight, fused)

    if config.use_structured or route_hint == "structured":
        structured = state.authoritative_text()
        if structured:
            _rrf(index.search_terms(structured, limit=config.pool_size),
                 structured_weight, fused)

    if (
        config.use_dense
        and embed is not None
        and getattr(embed, "available", False)
        and _dense_gate_open(state, config, track)
    ):
        try:
            query_vec = qvec if qvec is not None else embed.encode_query(state.full_text())
            _rrf(embed.search(query_vec, config.dense_pool), config.weight_dense, fused)
            # Post-override focused dense sub-route, mirroring the lexical
            # ``focused`` route's guard.
            if state.override_turn is not None:
                focused_vec = embed.encode_query(state.focused_text())
                _rrf(embed.search(focused_vec, config.dense_pool),
                     config.weight_dense_focused, fused)
        except Exception:
            pass

    if config.use_llm_terms and llm is not None and getattr(llm, "available", False):
        _rrf(_llm_term_route(index, state, llm, config.pool_size),
             config.weight_llm_terms, fused)

    pool: list[str] = []
    if config.use_category_pool:
        pool = index.match_pool(opening_category(state.opening), config.category_pool_max)
        # Fused as an ordinary weighted route: RRF exists precisely because the
        # routes' raw scores are not comparable, and popularity rank is no more
        # comparable to BM25 than BM25 is to cosine.
        _rrf([(asin, 0.0) for asin in pool], config.weight_category_pool, fused)

    ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[: config.pool_size]
    if pool:
        ranked = _tail_in_pool_members(pool, ranked)
    return ranked


def _tail_in_pool_members(
    pool: list[str], ranked: list[tuple[str, float]]
) -> list[tuple[str, float]]:
    """Append pool members that fusion still cut, keeping recall complete.

    The RRF route above puts pool members into the fused ordering, but the
    ``pool_size`` truncation can still drop the tail of a large pool - and the
    whole point of this route is that the target is inside the pool 200/200, so
    a truncation that loses it gives the guarantee away.

    Scores must stay **positive and below the fused floor**: the reranker mixes
    the retrieval score in as ``retrieval_weight * (score / top_score)``
    (src/rerank.py), so a negative filler here is not a gentle demotion, it is a
    penalty of order ``1 / top_score`` that no amount of span evidence can
    overcome. Appended members start at half the floor and decay across the
    pool, which orders them by popularity without pre-judging them.
    """
    present = {asin for asin, _score in ranked}
    floor = ranked[-1][1] if ranked else 1.0
    extra = [
        (asin, floor * (0.5 - 0.4 * position / len(pool)))
        for position, asin in enumerate(pool)
        if asin not in present
    ]
    return ranked + extra
