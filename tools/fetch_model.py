"""Fetch the quantised cross-encoder used by the optional S6b stage.

    pip install -r requirements.txt
    python3 tools/fetch_model.py

Downloads into ``models/ms-marco-MiniLM-L6-v2/`` (gitignored) and prints the
size. Two files are needed: one int8 ONNX graph and the tokenizer.

The upstream repo already publishes int8 ONNX exports, so there is no export
step and no torch/optimum dependency - the runtime is onnxruntime + numpy +
tokenizers. The quantised graph is 23.2 MB against 90.9 MB for the PyTorch
checkpoint, which is also what makes the "lightweight local assets" clause in
docs/submission_rules.md arguable if the weights were ever bundled. They are
not: `src/semantic.py` no-ops without them, so a clean clone still scores.

The variant is chosen by CPU architecture. int8 kernels are ISA-specific and
picking the wrong one is slow rather than wrong, so the fallback is safe.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "models" / "ms-marco-MiniLM-L6-v2"

#: Published int8 exports, by machine. All are 23.2 MB.
VARIANTS = {
    "arm64": "onnx/model_qint8_arm64.onnx",
    "aarch64": "onnx/model_qint8_arm64.onnx",
    "x86_64": "onnx/model_qint8_avx512_vnni.onnx",
    "amd64": "onnx/model_qint8_avx512_vnni.onnx",
}
FALLBACK = "onnx/model_qint8_avx512.onnx"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--variant",
        default=None,
        help="override the ONNX file, e.g. onnx/model.onnx for full precision",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "huggingface_hub is required.\n    pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    machine = platform.machine().lower()
    variant = args.variant or VARIANTS.get(machine, FALLBACK)
    if args.variant is None and machine not in VARIANTS:
        print(f"unrecognised machine {machine!r}; using {FALLBACK}", file=sys.stderr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.model_id}  ({machine} -> {variant})")

    try:
        graph = hf_hub_download(args.model_id, variant)
        tokenizer = hf_hub_download(args.model_id, "tokenizer.json")
    except Exception as error:  # noqa: BLE001 - any failure must be legible
        print(f"download failed: {error}", file=sys.stderr)
        return 1

    # src/semantic.py looks for model_quantized.onnx, then model.onnx.
    target = out / "model_quantized.onnx"
    target.write_bytes(Path(graph).read_bytes())
    (out / "tokenizer.json").write_bytes(Path(tokenizer).read_bytes())

    total = 0
    for path in sorted(out.iterdir()):
        if path.is_file():
            total += path.stat().st_size
            print(f"  {path.name:<26} {path.stat().st_size / 1e6:7.1f} MB")
    print(f"  {'total':<26} {total / 1e6:7.1f} MB   -> {out}")
    print("\nEnable with AgentConfig(semantic=SemanticConfig(enabled=True)),")
    print("or: python3 tools/sweep.py --split dev --configs semantic_off,semantic_on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
