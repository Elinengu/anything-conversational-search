"""Offline one-time build of the dense catalog-embedding artifact.

This is a **development-time** script. It runs a pretrained sentence-embedding
model (default ``BAAI/bge-small-en-v1.5``, 384-dim, CLS pooling) over the 50k
catalog via ONNX Runtime - no ``torch`` - and writes two things into ``--out``
(default ``data/embeddings/``):

  1. ``<name>.<recipe>.npz`` - the corpus matrix the agent loads at runtime:
       vectors : float16, (n_products, dim), L2-normalised
       asins   : str array, row i -> parent_asin (catalog file order)
       meta    : one JSON string (model, dim, pooling, recipe, query
                 instruction, max_tokens, catalog sha1 fingerprint, ...)

  2. ``<name>/`` - the vendored encoder, so the runtime needs only
     ``onnxruntime`` + ``numpy`` + ``tokenizers``:
       model.onnx        copied verbatim from the model repo
       tokenizer.json    copied verbatim
       meta.json         {model, dim, pooling, query_instruction, max_tokens}

Both are git-ignored and distributed as a GitHub Release asset the same way
``catalog.jsonl.gz`` is (see ``data/README.md``).

Fetch source: ``huggingface.co`` is blocked on some networks (this project's own dev
environment among them - see docs/team/dense_route.md / dense_rerank.md). ``GCS_SOURCES``
below is a fallback for known models, pointing at Qdrant's public fastembed mirror on GCS
(verified reachable: a tar.gz of exactly the files ``_Encoder`` needs -
``model_optimized.onnx`` + ``tokenizer.json`` - no huggingface_hub import required). It is
used automatically when the model is a known GCS source; pass ``--use-hf`` to force the
original ``huggingface_hub`` path instead, or ``--source-url`` to point at a different
tarball.

Usage::

    pip install onnxruntime tokenizers numpy        # GCS path - no huggingface_hub needed
    python3 tools/build_embeddings.py --recipe v1cat
    python3 tools/build_embeddings.py --recipe v1
    pip install huggingface_hub                     # only for --use-hf / an unknown model
    python3 tools/build_embeddings.py --model BAAI/bge-base-en-v1.5 --recipe v1cat --use-hf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.text import TOKEN_RE, flatten  # noqa: E402


RECIPES = ("v1", "v1cat")

# Known models: HF repo -> (onnx path, pooling, default query instruction).
KNOWN = {
    "BAAI/bge-small-en-v1.5": ("onnx/model.onnx", "cls",
                               "Represent this sentence for searching relevant passages: "),
    "BAAI/bge-base-en-v1.5": ("onnx/model.onnx", "cls",
                              "Represent this sentence for searching relevant passages: "),
    "intfloat/e5-small-v2": ("onnx/model.onnx", "mean", "query: "),
}

# Known models: HF repo -> a tarball containing (some *.onnx, tokenizer.json), fetchable
# without huggingface_hub. Verified against BAAI/bge-small-en-v1.5: the tarball holds
# model_optimized.onnx (input/output names identical to the HF onnx/model.onnx export -
# input_ids/attention_mask/token_type_ids -> last_hidden_state) and tokenizer.json. This
# is Qdrant's public fastembed packaging (github.com/qdrant/fastembed), reachable on
# networks that block huggingface.co directly.
GCS_SOURCES = {
    "BAAI/bge-small-en-v1.5":
        "https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-en-v1.5.tar.gz",
}


def _product_text(product: dict, recipe: str) -> str:
    """Text fed to the encoder for one catalog row.

    ``v1``    - the blob ``src/index.py`` builds for the reranker: title +
                features + description + details, token-joined, lower.
    ``v1cat`` - ``v1`` with the two most specific category labels prepended.
    """
    title = flatten(product.get("title"))
    features = flatten(product.get("features"))
    description = flatten(product.get("description"))
    details = flatten(product.get("details") or {})
    blob = " ".join(TOKEN_RE.findall(f"{title} {features} {description} {details}")).lower()
    if recipe == "v1cat":
        categories = product.get("categories") or []
        leaf = " ".join(str(value) for value in categories[-2:]).lower().strip()
        if leaf:
            blob = f"{leaf} {blob}"
    return blob


def _catalog_fingerprint(catalog_path: Path) -> str:
    digest = hashlib.sha1()
    with catalog_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_catalog(catalog_path: Path) -> tuple[list[str], list[dict]]:
    asins: list[str] = []
    products: list[dict] = []
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            asins.append(str(product["parent_asin"]))
            products.append(product)
    return asins, products


class _Encoder:
    def __init__(self, onnx_path: str, tokenizer_path: str, pooling: str, max_tokens: int) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.pooling = pooling
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=max_tokens)
        self.tokenizer.enable_padding()

    def encode(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feed)[0]  # (batch, seq, dim)
        if self.pooling == "mean":
            m = mask[:, :, None].astype(np.float32)
            vec = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        else:  # cls
            vec = hidden[:, 0]
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (vec / norms).astype(np.float32)


def _fetch_from_tarball(url: str, cache_dir: Path) -> tuple[str, str]:
    """Download and extract a model tarball; return (onnx_path, tokenizer_path).

    Cached under ``cache_dir`` by URL basename so re-running for a second recipe
    (``--recipe v1`` after ``v1cat``) does not redownload the ~75 MB archive.
    stdlib only (``urllib.request`` + ``tarfile``) - no ``huggingface_hub``, no
    network beyond the one GET.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_name = url.rsplit("/", 1)[-1]
    archive_path = cache_dir / archive_name
    extract_dir = cache_dir / archive_name.removesuffix(".tar.gz").removesuffix(".tgz")

    if not archive_path.exists():
        print(f"  downloading {url} ...", flush=True)
        with urllib.request.urlopen(url) as response:
            data = response.read()
        archive_path.write_bytes(data)
        print(f"  fetched {len(data) / 1e6:.1f} MB", flush=True)

    if not extract_dir.exists() or not any(extract_dir.rglob("*.onnx")):
        with tarfile.open(archive_path, "r:gz") as tar:
            # Guard against path traversal in a third-party archive.
            for member in tar.getmembers():
                target = (extract_dir.parent / member.name).resolve()
                if not str(target).startswith(str(extract_dir.parent.resolve())):
                    raise ValueError(f"unsafe path in archive: {member.name}")
            tar.extractall(extract_dir.parent)

    onnx_candidates = sorted(extract_dir.rglob("*.onnx"))
    tokenizer_candidates = sorted(extract_dir.rglob("tokenizer.json"))
    if not onnx_candidates or not tokenizer_candidates:
        raise FileNotFoundError(
            f"expected an *.onnx and a tokenizer.json under {extract_dir}, found "
            f"{len(onnx_candidates)} onnx file(s), {len(tokenizer_candidates)} tokenizer(s)"
        )
    if len(onnx_candidates) > 1:
        # Prefer a quantized/optimized export if the archive ships more than one -
        # smaller and what fastembed's own packaging actually ships as the default.
        preferred = [p for p in onnx_candidates if "optim" in p.name or "quant" in p.name]
        onnx_candidates = preferred or onnx_candidates
    return str(onnx_candidates[0]), str(tokenizer_candidates[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default=str(_REPO_ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--out", default=str(_REPO_ROOT / "data" / "embeddings"))
    parser.add_argument("--recipe", default="v1cat", choices=RECIPES)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pooling", choices=("cls", "mean"), default=None)
    parser.add_argument("--query-instruction", default=None,
                        help="prepended to queries only (not catalog text)")
    parser.add_argument("--source-url", default=None,
                        help="tar.gz containing an *.onnx + tokenizer.json, fetched "
                             "without huggingface_hub - overrides GCS_SOURCES")
    parser.add_argument("--use-hf", action="store_true",
                        help="force huggingface_hub even for a model in GCS_SOURCES "
                             "(needs `pip install huggingface_hub` and HF network access)")
    parser.add_argument("--cache-dir", default=None,
                        help="where --source-url downloads are cached (default: <out>/_dl_cache)")
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    out_root = Path(args.out)
    name = args.model.rstrip("/").split("/")[-1]
    npz_path = out_root / f"{name}.{args.recipe}.npz"
    encoder_dir = out_root / name

    onnx_rel, known_pooling, known_instruction = KNOWN.get(
        args.model, ("onnx/model.onnx", "cls", "")
    )
    pooling = args.pooling or known_pooling
    query_instruction = args.query_instruction if args.query_instruction is not None else known_instruction

    started = time.time()
    source_url = args.source_url or (None if args.use_hf else GCS_SOURCES.get(args.model))
    if source_url:
        print(f"fetching {args.model} via {source_url} ...", flush=True)
        cache_dir = Path(args.cache_dir) if args.cache_dir else out_root / "_dl_cache"
        onnx_path, tokenizer_path = _fetch_from_tarball(source_url, cache_dir)
    else:
        from huggingface_hub import hf_hub_download

        print(f"fetching {args.model} ({onnx_rel}) via huggingface_hub ...", flush=True)
        onnx_path = hf_hub_download(args.model, onnx_rel)
        tokenizer_path = hf_hub_download(args.model, "tokenizer.json")

    encoder = _Encoder(onnx_path, tokenizer_path, pooling, args.max_tokens)
    dim = encoder.encode(["probe"]).shape[1]
    print(f"  dim={dim} pooling={pooling} instruction={query_instruction!r}", flush=True)

    print(f"reading {catalog_path} ...", flush=True)
    asins, products = _load_catalog(catalog_path)
    texts = [_product_text(product, args.recipe) for product in products]
    print(f"  {len(texts)} products, recipe={args.recipe}", flush=True)

    print("encoding ...", flush=True)
    chunks: list[np.ndarray] = []
    enc_start = time.time()
    for done, start in enumerate(range(0, len(texts), args.batch_size)):
        chunks.append(encoder.encode(texts[start : start + args.batch_size]))
        rows = start + args.batch_size
        if done % 10 == 0 and rows:
            rate = rows / max(time.time() - enc_start, 1e-6)
            eta = (len(texts) - rows) / max(rate, 1e-6)
            print(f"  {min(rows, len(texts))}/{len(texts)}  {rate:.0f} rows/s  eta {eta:.0f}s",
                  flush=True)
    vectors = np.concatenate(chunks, axis=0).astype(np.float16)

    out_root.mkdir(parents=True, exist_ok=True)
    encoder_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(onnx_path, encoder_dir / "model.onnx")
    shutil.copyfile(tokenizer_path, encoder_dir / "tokenizer.json")
    encoder_meta = {
        "model": args.model,
        "dim": int(dim),
        "pooling": pooling,
        "query_instruction": query_instruction,
        "max_tokens": args.max_tokens,
    }
    (encoder_dir / "meta.json").write_text(json.dumps(encoder_meta, indent=2) + "\n", encoding="utf-8")

    meta = {
        **encoder_meta,
        "recipe": args.recipe,
        "rows": len(asins),
        "catalog_sha1": _catalog_fingerprint(catalog_path),
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    np.savez(
        npz_path,
        vectors=vectors,
        asins=np.array(asins, dtype=object),
        meta=np.array(json.dumps(meta)),
    )
    print(
        f"\nwrote {npz_path}  ({npz_path.stat().st_size / 1e6:.1f} MB, "
        f"{vectors.shape[0]}x{vectors.shape[1]} f16)"
        f"\nwrote {encoder_dir}/  (model.onnx + tokenizer.json + meta.json)"
        f"\ndone in {time.time() - started:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
