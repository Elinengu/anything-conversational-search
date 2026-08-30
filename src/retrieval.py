"""S5 - multi-route retrieval with reciprocal-rank fusion.

Routes:
  * ``terms``   - bag-of-words OR query over the whole conversation (recall)
  * ``anchor``  - bag-of-words over the opening turn only (topic drift guard)
  * ``focused`` - bag-of-words over post-override turns only (override handling)

RRF is used rather than score addition because the routes produce scores on
incomparable scales; rank fusion needs no calibration and degrades gracefully when
a route returns nothing.

Verbatim constraint matching deliberately lives in the reranker instead of being a
third route here. As an FTS5 phrase query it recalls the target in only 47 of 80
sampled sessions, so fusing it at retrieval time injects more noise than signal;
applied as a rescoring signal over a pool the terms route already fills, the same
evidence is pure gain.

A dense (bge-small sentence-embedding) route was built and measured on branch
``dense_rerank``: it recovers none of the paraphrase ``never_retrieved`` tail and
is slightly negative on the cooperative set, because the retrieved space is
dominated by the product category. See ``docs/team/dense_route.md``. This branch
is BM25 only.
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
    weight_terms: float = 1.0
    weight_anchor: float = 0.6
    weight_focused: float = 0.8
    pool_size: int = 300


def _rrf(ranked: list[tuple[str, float]], weight: float, sink: dict[str, float]) -> None:
    for position, (parent_asin, _score) in enumerate(ranked):
        sink[parent_asin] = sink.get(parent_asin, 0.0) + weight / (RRF_K + position + 1)


def retrieve(
    index: CatalogIndex,
    state: DialogState,
    config: RetrievalConfig | None = None,
) -> list[tuple[str, float]]:
    """Return a fused candidate pool, best first."""
    config = config or RetrievalConfig()
    fused: dict[str, float] = {}

    if config.use_anchor and state.opening:
        _rrf(index.search_terms(state.opening, limit=config.pool_size),
             config.weight_anchor, fused)

    if config.use_terms:
        _rrf(index.search_terms(state.full_text(), limit=config.pool_size),
             config.weight_terms, fused)

    if config.use_focused and state.override_turn is not None:
        _rrf(index.search_terms(state.focused_text(), limit=config.pool_size),
             config.weight_focused, fused)

    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))[: config.pool_size]
