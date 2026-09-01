"""Repeat one configuration N times and report the spread, not a point.

Why this exists: the LLM endpoint is not deterministic at ``temperature=0.0``.
The same ``llm_rerank_gated`` configuration scored ``0.9567`` and ``0.9558`` on
two runs of the public set - a spread of ``0.0009``, which is the size of every
effect any LLM row in this repo has claimed. A single run therefore cannot
distinguish "this layer helps" from "the API answered differently today", and
no single-run LLM number should be quoted again.

The offline rows are deterministic and will report a spread of exactly zero;
running one as a control is a cheap way to prove the harness itself is not the
source of the variance.

Usage::

    python3 tools/llm_variance.py --config llm_t1 --repeat 3
    python3 tools/llm_variance.py --config llm_t1 --repeat 5 --llm-cache runs/llm_cache
    python3 tools/llm_variance.py --config router_on --repeat 2      # control

``--llm-cache`` makes repeats nearly free, but a cached repeat replays the first
run's answers and will under-report the spread to zero. Use it while iterating;
leave it off for the number you publish.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402
from tools.sweep import build_configs, split_samples  # noqa: E402

METRICS = ("recommended_technical_score", "hit_rate_at_10", "mrr", "mttc")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--split", default="all", choices=["dev", "holdout", "all"])
    parser.add_argument("--config", required=True, help="a tools/sweep.py config name")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--llm-cache", default="", metavar="DIR",
                        help="reuse cached responses (fast, but hides the spread)")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    configs = build_configs(args.catalog)
    if args.config not in configs:
        parser.error(f"unknown config {args.config!r}")
    config = configs[args.config]
    if args.llm_cache:
        config = replace(config, llm=replace(config.llm, cache_dir=args.llm_cache))

    samples = split_samples(load_jsonl(args.dataset), args.split)
    catalog_ids, categories, products = catalog_index(args.catalog)

    print(f"config={args.config}  split={args.split}  sessions={len(samples)}  "
          f"repeat={args.repeat}  cache={args.llm_cache or 'off'}\n")
    header = f"{'run':>4} {'hit@10':>7} {'mrr':>7} {'mttc':>6} {'score':>8} {'tokens':>9} {'time':>6}"
    print(header)
    print("-" * len(header))

    runs: list[dict] = []
    totals: Counter = Counter()
    for index in range(1, args.repeat + 1):
        started = time.time()
        agent = Agent(args.catalog, config)
        result = evaluate(agent, samples, catalog_ids, categories, products)
        runs.append(result)
        totals.update(getattr(agent.llm, "stats", Counter()))
        usage = result.get("reported_token_usage", {}) or {}
        print(f"{index:>4} {result['hit_rate_at_10']:>7.3f} {result['mrr']:>7.4f} "
              f"{result['mttc']:>6.2f} {result['recommended_technical_score']:>8.6f} "
              f"{usage.get('total_tokens', 0):>9} {time.time() - started:>5.0f}s", flush=True)

    print(f"\n{'metric':<28} {'mean':>10} {'min':>10} {'max':>10} {'spread':>10}")
    print("-" * 72)
    spread_of_score = 0.0
    for metric in METRICS:
        values = [r[metric] for r in runs]
        spread = max(values) - min(values)
        if metric == "recommended_technical_score":
            spread_of_score = spread
        print(f"{metric:<28} {statistics.fmean(values):>10.6f} {min(values):>10.6f} "
              f"{max(values):>10.6f} {spread:>10.6f}")

    if totals:
        print("\nLLM call accounting (summed over runs):")
        for key, value in sorted(totals.items()):
            print(f"  {key:<20} {value}")

    print(f"\nAn effect smaller than {spread_of_score:.6f} on this config is not "
          f"distinguishable from run-to-run noise.")
    if args.llm_cache:
        print("NOTE: --llm-cache was on, so this spread is a floor, not the real one.")

    if args.output:
        Path(args.output).write_text(
            json.dumps({"config": args.config, "runs": runs, "stats": dict(totals)}, indent=2),
            encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
