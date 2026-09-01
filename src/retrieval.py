"""S5 - multi-route retrieval with reciprocal-rank fusion.

Routes:
  * ``terms``   - bag-of-words OR query over the whole conversation (recall)
  * ``anchor``  - bag-of-words over the opening turn only (topic drift guard)
  * ``focused`` - bag-of-words over post-override turns only (override handling)
  * ``structured`` - compact active-slot query, activated as a stagnation route

A dense sentence-embedding route was built and measured here (branch
``dense_rerank``, then re-measured against the live state machine on branch
``state-encoder-eval``) and is now removed: no embedding configuration ever
cleared the noise floor, and the one stress gain that did turned out to be
compensating for a lexical bug that has since been fixed. The measurements are
kept in ``docs/team/dense_route.md``,
``docs/team/branch_state_encoder_eval_changes.md`` and IMPLEMENTATION.md; the
code is not, so the pipeline needs no third-party packages at all.

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

from src.index import CatalogIndex
from src.state import DialogState


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


def retrieve(
    index: CatalogIndex,
    state: DialogState,
    config: RetrievalConfig | None = None,
    route_hint: str | None = None,
) -> list[tuple[str, float]]:
    """Return a fused candidate pool, best first."""
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
