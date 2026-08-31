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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.index import CatalogIndex
from src.rerank import _dense_gate_open
from src.state import DialogState

if TYPE_CHECKING:  # avoid a hard numpy dependency on the BM25-only path
    from src.embed import EmbeddingIndex


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


def _rrf(ranked: list[tuple[str, float]], weight: float, sink: dict[str, float]) -> None:
    for position, (parent_asin, _score) in enumerate(ranked):
        sink[parent_asin] = sink.get(parent_asin, 0.0) + weight / (RRF_K + position + 1)


def retrieve(
    index: CatalogIndex,
    state: DialogState,
    config: RetrievalConfig | None = None,
    route_hint: str | None = None,
    embed: "EmbeddingIndex | None" = None,
    qvec: Any = None,
    track: str | None = None,
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

    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))[: config.pool_size]
