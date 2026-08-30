"""S6b - optional neural cross-encoder reranking over ambiguous clusters.

This is the only stage in the pipeline that is not standard-library-only, and it
is **off by default**. The scored path never reaches it. Everything here is
built so that a missing dependency, missing weights, or a broken model file
costs exactly nothing: every failure returns the candidate list untouched, which
is the same degradation shape ``rerank`` already uses for an empty pool or an
empty span set (``src/rerank.py``).

Why a cross-encoder at all
--------------------------
Every shipped rerank signal is lexical: verbatim span coverage, facet agreement,
category alignment. Measured with an oracle that forces the target to rank 1
whenever it is anywhere in the pool, the ceiling for *any* reranking improvement
is +0.043 on the public set and +0.084 on the adversarial set. 70% of public
sessions already rank the target first; of the 42 that do not, 25 need a
rank-2-to-4 promotion among near-identical cluster-mates, which is precisely
where lexical evidence saturates and a semantic model might not.

Why it is gated rather than always on
-------------------------------------
A cross-encoder scores one (query, document) pair at a time, so cost is linear
in pool depth - two to three orders of magnitude above the 16.6 ms mean turn
this agent currently takes. The gate spends that budget only where the symbolic
ranking is visibly undecided. Measured at the first slate turn, the ambiguity
condition separates cleanly (public mean RR 0.774 when it fires against 0.987
when it does not), but at the originally proposed thresholds it fires on 73% of
sessions - which is an always-on stage with extra steps. The defaults here are
tighter; see ``SemanticConfig``.

Why RRF rather than adding the logit
------------------------------------
Cross-encoder logits are uncalibrated and unbounded, and the symbolic total is a
sum of hand-weighted terms where one matched span is worth about 1.12. Adding
the two puts an arbitrary scale in charge of the ranking. Rank fusion needs no
calibration, which is the same argument ``src/retrieval.py`` makes for fusing
its routes - and it is why ``_rrf`` is imported from there rather than
reimplemented.

THE MEASURED RESULT: IT LOSES
-----------------------------
Every configuration scored below the lexical reranker it was meant to help
(dev 0.9268 -> 0.9211, hard 0.7981 -> 0.7944), and lowering the semantic weight
recovers the baseline monotonically - 0.7 gives dev 0.9211, 0.3 gives 0.9249,
0.0 gives 0.9268. An optimum at zero is what a signal carrying no usable
information looks like.

The mechanism is in one number: on the 162 fired turns where the target was in
the rescored head, the fusion moved it **up 46 times and down 74**, mean rank
7.63 -> 8.77. The model is not miscalibrated, it is anti-correlated with the
target on this task. The likely reason is domain mismatch that no amount of
tuning fixes: MS MARCO pairs a natural-language question with a prose passage,
whereas here the "query" is simulator boilerplate ("For that, what matters is:
...") and the "document" is a token-joined title+features+description blob. The
job asked of it - separating near-identical cluster-mates that share every
stated facet - is also the hardest discrimination in the pool.

Cost, for completeness: mean turn latency 30.7 ms -> 389.8 ms, p95 73.7 ms ->
1347.8 ms, max 1480 ms, with the gate firing on 28% of rerank calls.

This module is kept, off, as a reproducible negative result - the measurement is
the deliverable, and ``docs/team/rerank_signals.md`` records it in full. It is
not a parked option: nothing selects it and no score depends on it.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path

from src.facets import extract
from src.index import CatalogIndex
from src.retrieval import _rrf
from src.state import DialogState


#: Where ``tools/fetch_model.py`` puts the quantised graph. Gitignored - the
#: weights are deliberately not committed, so on a fresh clone this path does
#: not exist and the stage no-ops.
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "ms-marco-MiniLM-L6-v2"


@dataclass
class SemanticConfig:
    """Everything the sweep harness is allowed to vary for this stage."""

    #: Off by default. The scored path must never depend on this stage.
    enabled: bool = False

    model_dir: Path = DEFAULT_MODEL_DIR

    # ---- the ambiguity gate ---------------------------------------------------
    # Thresholds are set from the measured rank distribution, not from session
    # counts. At tied_leaders >= 8 the gate fired on 73% of public sessions,
    # which defeats the point of gating; these are the tightened starting values
    # and tools/sweep.py brackets them.
    #: Candidates sharing the top raw span-coverage count.
    tied_leaders: int = 15
    #: Largest (category, material, colour) signature group inside the top 20.
    facet_cluster: int = 12
    #: Fire when no span is distinctive (matches <= distinctive_max candidates).
    distinctive_max: int = 3
    #: How many of the gate conditions must hold. The proposal said two; one is
    #: the looser reading. Swept.
    conditions_required: int = 2

    # ---- cost -----------------------------------------------------------------
    #: Cross-encoder passes per firing turn. Cost is linear in this.
    depth: int = 50
    #: Candidates examined when computing the gate statistics (cheap, lexical).
    gate_depth: int = 50
    #: Tokens per (query, document) pair. The model's window is 512.
    max_length: int = 320
    #: Pairs per ONNX call.
    batch_size: int = 32

    # ---- fusion ---------------------------------------------------------------
    weight_symbolic: float = 1.0
    weight_semantic: float = 0.7
    # A "protect unique lexical evidence" guard was specified and built: a
    # candidate matching a span no other candidate matched would be clamped so
    # the neural score could promote it but never demote it. Measured over the
    # public set it fired on **0 of 8750** candidates examined - inside a pool
    # retrieved by those very spans there is no such thing as a unique span -
    # and removing it left every score bit-identical. Deleted rather than kept
    # as an inert flag, per docs/team/signal_descriptions.md.


class _Scorer:
    """Lazily loaded ONNX cross-encoder. One instance per model directory."""

    def __init__(self, model_dir: Path) -> None:
        self.ok = False
        self.session = None
        self.tokenizer = None
        self._input_names: tuple[str, ...] = ()
        # Check for weights BEFORE importing the runtime. Order matters: on
        # macOS, importing onnxruntime and then exiting the interpreter can
        # abort in its teardown ("recursive_mutex lock failed", SIGABRT),
        # intermittently and long after any work is done. Nothing here needs
        # the runtime when there are no weights to feed it, and the no-weights
        # path is the common one - the scored agent and the test suite both
        # take it - so it must not touch onnxruntime at all.
        model_path = self._find_model(model_dir)
        tokenizer_path = model_dir / "tokenizer.json"
        if model_path is None or not tokenizer_path.exists():
            return
        try:
            import numpy  # noqa: F401
            import onnxruntime
            from tokenizers import Tokenizer
        except ImportError:
            # Weights present but no runtime. Silent and free, same as above.
            return
        try:
            options = onnxruntime.SessionOptions()
            # The evaluator is single-threaded and runs many short calls; letting
            # ORT spawn a pool per call costs more than it saves.
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self.session = onnxruntime.InferenceSession(
                str(model_path), options, providers=["CPUExecutionProvider"]
            )
            self._input_names = tuple(i.name for i in self.session.get_inputs())
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self.tokenizer.enable_truncation(max_length=512)
            self.tokenizer.enable_padding()
        except Exception:
            # A corrupt or incompatible export must not cost the run.
            self.session = None
            self.tokenizer = None
            return
        self.ok = True

    @staticmethod
    def _find_model(model_dir: Path) -> Path | None:
        """Prefer the quantised export; fall back to the full-precision one."""
        for name in ("model_quantized.onnx", "model.onnx"):
            candidate = model_dir / name
            if candidate.exists():
                return candidate
        return None

    def score(self, query: str, documents: list[str], config: SemanticConfig) -> list[float]:
        """Relevance logit per document. Empty list on any failure."""
        if not self.ok or not documents:
            return []
        import numpy

        self.tokenizer.enable_truncation(max_length=config.max_length)
        out: list[float] = []
        try:
            for start in range(0, len(documents), config.batch_size):
                chunk = documents[start : start + config.batch_size]
                encoded = self.tokenizer.encode_batch([(query, doc) for doc in chunk])
                feeds = {
                    "input_ids": numpy.array(
                        [e.ids for e in encoded], dtype=numpy.int64
                    ),
                    "attention_mask": numpy.array(
                        [e.attention_mask for e in encoded], dtype=numpy.int64
                    ),
                    "token_type_ids": numpy.array(
                        [e.type_ids for e in encoded], dtype=numpy.int64
                    ),
                }
                # Some exports drop token_type_ids; only feed what the graph wants.
                feeds = {k: v for k, v in feeds.items() if k in self._input_names}
                logits = self.session.run(None, feeds)[0]
                out.extend(float(value) for value in numpy.asarray(logits).reshape(-1))
        except Exception:
            return []
        return out if len(out) == len(documents) else []


#: Memoised per model directory, mirroring ``src/index.py``'s ``_CACHE`` - the
#: ONNX session costs far more to build than to call.
_CACHE: dict[str, _Scorer] = {}


def _scorer(model_dir: Path) -> _Scorer:
    key = str(model_dir)
    cached = _CACHE.get(key)
    if cached is None:
        cached = _CACHE[key] = _Scorer(model_dir)
    return cached


def gate_stats(
    index: CatalogIndex,
    state: DialogState,
    candidates: list[tuple[str, float]],
    config: SemanticConfig,
) -> dict[str, int]:
    """Cheap lexical measures of how undecided the symbolic ranking is.

    All three are computed over the same head of the pool, and none of them
    touches the model - the gate must be far cheaper than what it gates.
    """
    spans = state.query_spans()
    head = candidates[: config.gate_depth]
    if not spans or not head:
        return {"tied_leaders": 0, "facet_cluster": 0, "distinctive": 0}

    texts: list[str] = []
    for parent_asin, _score in head:
        product = index.products.get(parent_asin)
        texts.append(f" {product['text']} " if product else "")

    coverage = [sum(1 for span in spans if f" {span} " in text) for text in texts]
    best = max(coverage) if coverage else 0
    tied = sum(1 for value in coverage if value == best)

    distinctive = 0
    for span in spans:
        hits = sum(1 for text in texts if f" {span} " in text)
        if 0 < hits <= config.distinctive_max:
            distinctive += 1

    signatures: collections.Counter = collections.Counter()
    for parent_asin, _score in head[:20]:
        product = index.products.get(parent_asin)
        if product is None:
            continue
        values = extract(product)
        signatures[(values.get("category"), values.get("material"), values.get("color"))] += 1
    cluster = max(signatures.values()) if signatures else 0

    return {"tied_leaders": tied, "facet_cluster": cluster, "distinctive": distinctive}


def is_ambiguous(stats: dict[str, int], config: SemanticConfig) -> bool:
    """True when enough gate conditions hold to justify the model's cost."""
    conditions = (
        stats["tied_leaders"] >= config.tied_leaders,
        stats["facet_cluster"] >= config.facet_cluster,
        stats["distinctive"] == 0,
    )
    return sum(conditions) >= config.conditions_required


def semantic_rerank(
    index: CatalogIndex,
    state: DialogState,
    candidates: list[tuple[str, float]],
    config: SemanticConfig | None = None,
) -> list[tuple[str, float]]:
    """Re-fuse the head of the pool with cross-encoder ranks. Same signature as
    ``rerank`` so it drops into the pipeline as one more line.

    Returns ``candidates`` unchanged whenever the stage is off, the runtime or
    weights are absent, the session is not ambiguous, or scoring fails.
    """
    config = config or SemanticConfig()
    if not config.enabled or not candidates:
        return candidates

    query = state.focused_text().strip()
    if not query:
        return candidates

    if not is_ambiguous(gate_stats(index, state, candidates, config), config):
        return candidates

    scorer = _scorer(config.model_dir)
    if not scorer.ok:
        return candidates

    head = candidates[: config.depth]
    tail = candidates[config.depth :]
    documents: list[str] = []
    for parent_asin, _score in head:
        product = index.products.get(parent_asin)
        documents.append(product["text"] if product else "")

    scores = scorer.score(query, documents, config)
    if not scores:
        return candidates

    semantic_order = sorted(
        zip((asin for asin, _s in head), scores),
        key=lambda item: (-item[1], item[0]),
    )

    fused: dict[str, float] = {}
    _rrf(head, config.weight_symbolic, fused)
    _rrf(semantic_order, config.weight_semantic, fused)

    symbolic_rank = {asin: position for position, (asin, _s) in enumerate(head)}
    reordered = sorted(
        fused.items(), key=lambda item: (-item[1], symbolic_rank.get(item[0], 0))
    )
    return reordered + tail
