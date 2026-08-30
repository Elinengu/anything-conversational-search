"""Coordinate ascent over the rerank weights, directly on the official score.

The shipped reranker is a linear feature-based model in the sense of Metzler &
Croft (Information Retrieval 10:257-274, 2007): a hand-weighted sum of a few
ranking features. Their result is that such models do better when the weights
are chosen by directly maximizing the IR metric than by hand - and coordinate
ascent is their estimator precisely because metrics like MRR are non-smooth,
so there is no gradient to follow.

Why this exists now: the near-miss anatomy (docs/team/rerank_signals.md) showed
that every session still losing rank sits in a *pure tie-break regime* - all
lexical signals exactly tied between target and impostor - where the tie is
broken by the retrieval score (wrong 33/33 on the public near-misses, a BM25
length-normalization artefact favouring thin listings) while popularity (right
31/33, the target is a real purchase) is drowned at weight 0.02 against 1.0.
The weights are mis-mixed for the regime that holds all remaining headroom.

Protocol, per the house rules in docs/team/signal_descriptions.md:

  * fit on the dev split ONLY (120 sessions);
  * holdout is a gate, never a selector - run it once, on the final vector;
  * span_weight stays fixed at 1.0: it is the definitional unit every other
    weight scales against, so fitting it would only rescale the vector;
  * what ships is a rounded, plateau-checked point, not the raw argmax.

One full dev evaluation costs ~26 s and every candidate vector needs one -
cached feature vectors cannot shortcut it because the session transcript is
itself weight-dependent (the session ends at the first hit, and the confidence
gate reads scores). A full fit is a few hundred evaluations; run it once, in
the background:

    python3 tools/fit_weights.py                     # dev split, prints trajectory
    python3 tools/fit_weights.py --max-cycles 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from src.rerank import RerankConfig  # noqa: E402
from starter.agent import Agent, AgentConfig  # noqa: E402
from tools.sweep import split_samples  # noqa: E402


#: Weights being fitted, in the fixed (deterministic) cycle order. span_weight
#: is deliberately absent - see the module docstring.
FITTED = (
    "popularity_weight",
    "retrieval_weight",
    "facet_weight",
    "category_weight",
    "tail_weight",
    "pair_weight",
    "facet_conflict_weight",
)

#: One absolute candidate grid for every weight. Absolute rather than
#: multiplicative so a weight parked near zero (popularity, 0.02) can reach the
#: informative region in one move instead of three cycles, and so the search is
#: reproducible without a seed. The current value is always added, so a cycle
#: can never make things worse.
GRID = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0)

#: Improvements smaller than this are treated as ties and do not move a weight
#: (26-second evaluations buy no re-measurement, so tiny wins are noise).
EPSILON = 1e-6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    parser.add_argument("--max-cycles", type=int, default=4)
    parser.add_argument("--output", default="", help="optional JSON path for the result")
    args = parser.parse_args()

    if args.split != "dev":
        print(
            "warning: fitting on anything but dev spends the gate split as "
            "training data - only do this to inspect, never to ship.",
            file=sys.stderr,
        )

    samples = split_samples(load_jsonl(args.dataset), args.split)
    catalog_ids, categories, products = catalog_index(args.catalog)

    weights = {name: getattr(RerankConfig(), name) for name in FITTED}
    cache: dict[tuple, float] = {}
    evaluations = 0

    def score_of(vector: dict[str, float]) -> float:
        nonlocal evaluations
        key = tuple(vector[name] for name in FITTED)
        if key in cache:
            return cache[key]
        config = AgentConfig(rerank=RerankConfig(**vector))
        result = evaluate(Agent(args.catalog, config), samples, catalog_ids, categories, products)
        cache[key] = result["recommended_technical_score"]
        evaluations += 1
        return cache[key]

    started = time.time()
    best = score_of(weights)
    print(f"start   score {best:.6f}   weights {json.dumps(weights)}", flush=True)

    for cycle in range(1, args.max_cycles + 1):
        improved = False
        for name in FITTED:
            candidates = sorted(set(GRID) | {weights[name]})
            best_value = weights[name]
            for value in candidates:
                if value == weights[name]:
                    continue
                trial = dict(weights)
                trial[name] = value
                trial_score = score_of(trial)
                if trial_score > best + EPSILON:
                    best, best_value, improved = trial_score, value, True
            if best_value != weights[name]:
                weights[name] = best_value
                print(
                    f"cycle {cycle}  {name} -> {best_value:g}   score {best:.6f}   "
                    f"({evaluations} evals, {time.time() - started:.0f}s)",
                    flush=True,
                )
        if not improved:
            print(f"cycle {cycle}: no improvement, stopping", flush=True)
            break

    print(
        f"\nfinal   score {best:.6f}   evals {evaluations}   "
        f"{time.time() - started:.0f}s\nweights {json.dumps(weights, indent=2)}",
        flush=True,
    )
    print(
        "\nThis is the DEV argmax, not what ships. Round each weight, check the\n"
        "plateau (re-evaluate at ±50% per weight), then gate on holdout and the\n"
        "hard set - docs/team/signal_descriptions.md carries the method."
    )
    if args.output:
        Path(args.output).write_text(
            json.dumps({"split": args.split, "score": best, "weights": weights}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
