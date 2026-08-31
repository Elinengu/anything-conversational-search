"""Method 4 - build the seed-target list for synthetic session generation.

Every ``parent_asin`` in ``data/catalog.jsonl`` MINUS every
``ground_truth.parent_asin`` in ``data/public_set.jsonl`` and
``data/hard_set.jsonl``. Excluding the test targets stops the model from
memorising the answer key; it does NOT stop it learning cooperative-simulator
ranking cues (see docs/team/lambdamart.md 2).

    python3 tools/lm_targets.py --out /path/seed_targets.txt
    python3 tools/lm_targets.py --assert-no-leak --out /path/seed_targets.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import load_jsonl  # noqa: E402


def catalog_asins(catalog: str) -> list[str]:
    import json
    out: list[str] = []
    with Path(catalog).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(str(json.loads(line)["parent_asin"]))
    return out


def test_targets() -> set[str]:
    banned: set[str] = set()
    for path in ("data/public_set.jsonl", "data/hard_set.jsonl"):
        for s in load_jsonl(path):
            banned.add(str(s["ground_truth"]["parent_asin"]))
    return banned


def build(catalog: str) -> tuple[list[str], set[str]]:
    banned = test_targets()
    seeds = [a for a in catalog_asins(catalog) if a not in banned]
    return seeds, banned


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--out", default="")
    ap.add_argument("--assert-no-leak", action="store_true",
                    help="exit non-zero if any seed target is a public/hard target")
    args = ap.parse_args()

    seeds, banned = build(args.catalog)
    seed_set = set(seeds)
    leaked = seed_set & banned

    print(f"catalog          : {len(catalog_asins(args.catalog))}")
    print(f"test targets      : {len(banned)}  (public 200 + hard 96, minus overlap)")
    print(f"seed targets      : {len(seeds)}")
    print(f"leak (seed cap test): {len(leaked)}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(seeds) + "\n")
        print(f"wrote {args.out}")

    if args.assert_no_leak and leaked:
        print(f"LEAK: {sorted(leaked)[:10]} ...", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
