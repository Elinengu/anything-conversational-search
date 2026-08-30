"""S5b - dense sentence-embedding index for the retrieval route and the reranker.

A pretrained bi-encoder (default ``BAAI/bge-small-en-v1.5``, 384-dim, CLS
pooling) run through ONNX Runtime - no ``torch``. ``tools/build_embeddings.py``
produces two artifacts:

  * ``<name>.<recipe>.npz`` - the L2-normalised catalog matrix, loaded here.
  * ``<name>/``             - the vendored encoder (``model.onnx`` +
                              ``tokenizer.json`` + ``meta.json``).

The same ``EmbeddingIndex`` and the same per-turn query vector feed both stages:

  * ``search`` - a ``vectors @ q`` matmul over the 50k catalog for the S5 dense
    retrieval route (there is no vector database - out of scope).
  * ``similarities`` - a gather + dot over an arbitrary asin list for the S6
    reranker's cosine term.

**Everything here degrades to nothing.** If the artifacts, ``onnxruntime`` or
``tokenizers`` are missing, ``EmbeddingIndex.available`` is ``False`` and every
caller falls back to the BM25-only pipeline. Final scoring may run with no
network and no third-party packages installed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


DEFAULT_NPZ = "data/embeddings/bge-small-en-v1.5.v1cat.npz"
DEFAULT_MODEL_DIR = "data/embeddings/bge-small-en-v1.5"


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _OnnxEncoder:
    """Query-side encoder: tokenize -> ONNX forward -> pool -> L2-normalise.

    Matches the build script's pooling exactly (they share the same ONNX graph
    and tokenizer), so query and catalog vectors live in one space.
    """

    def __init__(self, model_dir: str | Path) -> None:
        import onnxruntime as ort  # local import - probed by EmbeddingIndex
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        self.dim = int(meta["dim"])
        self.pooling = meta.get("pooling", "cls")
        self.query_instruction = meta.get("query_instruction", "") or ""
        self.max_tokens = int(meta.get("max_tokens", 256))

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_tokens)

    def encode(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        encoding = self.tokenizer.encode(self.query_instruction + text)
        ids = np.array([encoding.ids], dtype=np.int64)
        mask = np.array([encoding.attention_mask], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feed)[0]  # (1, seq, dim)
        if self.pooling == "mean":
            weights = mask[0][:, None].astype(np.float32)
            vec = (hidden[0] * weights).sum(axis=0) / max(float(weights.sum()), 1e-9)
        else:
            vec = hidden[0, 0]
        norm = float(np.linalg.norm(vec))
        return (vec / norm if norm > 0.0 else vec).astype(np.float32)


class EmbeddingIndex:
    """The catalog matrix plus a lazily-built ONNX query encoder.

    ``available`` is the single gate every caller checks: ``True`` only when the
    ``.npz`` loaded, its catalog fingerprint matches (when checked), and the
    encoder files + ``onnxruntime`` + ``tokenizers`` are present.
    """

    def __init__(
        self,
        npz_path: str | Path = DEFAULT_NPZ,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        catalog_path: str | Path | None = None,
    ) -> None:
        self.available = False
        self.dim = 0
        self.vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.asins: list[str] = []
        self.row: dict[str, int] = {}
        self._model_dir = Path(model_dir)
        self._encoder: _OnnxEncoder | None = None
        self._load(Path(npz_path), catalog_path)

    def _load(self, npz_path: Path, catalog_path: str | Path | None) -> None:
        try:
            import onnxruntime  # noqa: F401
            import tokenizers  # noqa: F401
        except Exception:
            return
        if not npz_path.exists():
            return
        try:
            data = np.load(npz_path, allow_pickle=True)
            vectors = np.ascontiguousarray(data["vectors"], dtype=np.float32)
            asins = [str(value) for value in data["asins"]]
            meta = json.loads(str(data["meta"]))
        except Exception:
            return
        if vectors.ndim != 2 or vectors.shape[0] != len(asins):
            return
        if catalog_path is not None and meta.get("catalog_sha1"):
            try:
                if meta["catalog_sha1"] != _sha1(Path(catalog_path)):
                    return  # artifact built against a different catalog - fall back
            except OSError:
                return
        if not (self._model_dir / "model.onnx").is_file():
            return
        if not (self._model_dir / "tokenizer.json").is_file():
            return

        self.vectors = vectors
        self.asins = asins
        self.row = {asin: i for i, asin in enumerate(asins)}
        self.dim = int(meta.get("dim", vectors.shape[1]))
        self.available = True

    # ---- query side -------------------------------------------------------

    def encode_query(self, text: str) -> np.ndarray:
        if self._encoder is None:
            self._encoder = _OnnxEncoder(self._model_dir)
        return self._encoder.encode(text)

    def search(self, qvec: np.ndarray, limit: int) -> list[tuple[str, float]]:
        """Top-``limit`` catalog rows by cosine (both sides unit-norm)."""
        if not self.available or qvec is None or not np.any(qvec):
            return []
        scores = self.vectors @ qvec.astype(np.float32)
        limit = min(int(limit), scores.shape[0])
        if limit <= 0:
            return []
        top = np.argpartition(scores, -limit)[-limit:]
        top = top[np.argsort(scores[top])[::-1]]
        return [(self.asins[i], float(scores[i])) for i in top]

    def similarities(self, qvec: np.ndarray, asins: list[str]) -> dict[str, float]:
        """Cosine of ``qvec`` against each of ``asins`` present in the index."""
        if not self.available or qvec is None or not np.any(qvec):
            return {}
        rows = [(asin, self.row[asin]) for asin in asins if asin in self.row]
        if not rows:
            return {}
        scores = self.vectors[[r for _, r in rows]] @ qvec.astype(np.float32)
        return {asin: float(score) for (asin, _), score in zip(rows, scores)}


_CACHE: dict[tuple[str, str], EmbeddingIndex] = {}


def load_embedding_index(
    npz_path: str | Path = DEFAULT_NPZ,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    catalog_path: str | Path | None = None,
) -> EmbeddingIndex:
    """Return a shared ``EmbeddingIndex``, loading the matrix at most once per path."""
    key = (str(Path(npz_path).resolve()), str(Path(model_dir).resolve()))
    index = _CACHE.get(key)
    if index is None:
        index = EmbeddingIndex(npz_path, model_dir, catalog_path)
        _CACHE[key] = index
    return index
