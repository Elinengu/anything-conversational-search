"""S5 - multi-route retrieval with reciprocal-rank fusion.

Routes:
  * ``terms``   - bag-of-words OR query over the whole conversation (recall)
  * ``focused`` - bag-of-words over post-override turns only (override handling)
  * ``structured`` - compact active-slot query, activated as a stagnation route

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

from dataclasses import dataclass

from src.index import CatalogIndex
from src.state import DialogState


RRF_K = 60.0


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

    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))[: config.pool_size]
