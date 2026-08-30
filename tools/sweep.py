"""S0 - experiment harness.

Runs the official evaluator over several agent configurations in one process so
that the 50,000-row catalog index is built once and shared (a full evaluation
run drops from minutes to roughly twenty seconds).

Also provides the dev/holdout split. The public set has 200 sessions but the
final score is decided by 800 private ones, so every configuration is tuned on
the 120-session dev split and confirmed on the 80-session holdout. Treat
differences below ~0.02 on the holdout as noise.

Usage:
    python3 tools/sweep.py --split dev
    python3 tools/sweep.py --split holdout --configs floor,phrase
    python3 tools/sweep.py --split all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from src.facets import FacetStore  # noqa: E402
from src.index import load_index  # noqa: E402
from src.policy import FixedPolicy, InfoGainPolicy  # noqa: E402
from src.rerank import RerankConfig  # noqa: E402
from src.retrieval import RetrievalConfig  # noqa: E402
from starter.agent import Agent, AgentConfig  # noqa: E402


DEV_FRACTION = 0.6


def split_samples(samples: list[dict], split: str) -> list[dict]:
    """Deterministic split, stratified by scenario_type.

    Sorting by sample_id inside each scenario keeps the split stable across runs
    without needing a stored manifest.
    """
    if split == "all":
        return samples
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[sample["scenario_type"]].append(sample)
    selected: list[dict] = []
    for scenario in sorted(grouped):
        rows = sorted(grouped[scenario], key=lambda item: item["sample_id"])
        cut = round(len(rows) * DEV_FRACTION)
        selected.extend(rows[:cut] if split == "dev" else rows[cut:])
    return sorted(selected, key=lambda item: item["sample_id"])


def build_configs(catalog: str) -> dict[str, AgentConfig]:
    """Named configurations to compare. Add rows here when testing a change."""
    facets = FacetStore(load_index(catalog).products)
    return {
        # The committed floor: terms-only retrieval, broadest question, hold
        # recommendations until the customer has disclosed something.
        "floor": AgentConfig(
            retrieval=RetrievalConfig(use_focused=False),
            rerank=RerankConfig(enabled=False),
            policy=FixedPolicy(),
            first_recommend_turn=3,
        ),
        # Adds the post-override focused route and verbatim span reranking.
        "rerank": AgentConfig(
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(enabled=True),
            policy=FixedPolicy(),
            first_recommend_turn=3,
        ),
        # Same pipeline, but the question is chosen by expected information gain
        # over the live candidate pool instead of being fixed.
        "infogain": AgentConfig(
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(enabled=True),
            policy=InfoGainPolicy(facets),
            first_recommend_turn=3,
        ),
        # Information gain restricted to specific attributes - no broad questions.
        "infogain_specific": AgentConfig(
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(enabled=True),
            policy=InfoGainPolicy(facets, allow_broad=False),
            first_recommend_turn=3,
        ),
        # Plain: same top 10 every turn from turn 3 (no scan) - the pre-scan floor.
        "plain": AgentConfig(
            policy=FixedPolicy(), first_recommend_turn=3, elimination_scan=False
        ),
        # Elimination scan: drop everything already shown, return the top of the
        # re-ranked survivors. Sweep the turn at which the scan starts emitting.
        "elim1": AgentConfig(policy=FixedPolicy(), first_recommend_turn=1),
        "elim2": AgentConfig(policy=FixedPolicy(), first_recommend_turn=2),
        "elim3": AgentConfig(policy=FixedPolicy(), first_recommend_turn=3),
        # Elimination scan, but hold every list until disclosure has stalled.
        "elim_hold1": AgentConfig(
            policy=FixedPolicy(), first_recommend_turn=1, hold_until_stalled=True
        ),
        "elim_hold2": AgentConfig(
            policy=FixedPolicy(), first_recommend_turn=2, hold_until_stalled=True
        ),
        # Negative facet evidence ablation: the shipped default is 0.4; these
        # rows bracket it (off / shipped / upper plateau).
        "conflict00": AgentConfig(rerank=RerankConfig(facet_conflict_weight=0.0)),
        "conflict04": AgentConfig(rerank=RerankConfig(facet_conflict_weight=0.4)),
        "conflict08": AgentConfig(rerank=RerankConfig(facet_conflict_weight=0.8)),
        # Association-preserving pair spans: off / candidate / upper plateau.
        "pair00": AgentConfig(rerank=RerankConfig(pair_weight=0.0)),
        "pair08": AgentConfig(rerank=RerankConfig(pair_weight=0.8)),
        "pair15": AgentConfig(rerank=RerankConfig(pair_weight=1.5)),
        # Slate-size ramp: how many candidates the *first* slate reveals before
        # widening to 10. Narrowing it defers commitment by a turn, which buys
        # the rank the next disclosed constraint earns - the evaluator freezes
        # MRR at the position the target held when it was first shown.
        # ramp_flat is the pre-ramp floor; 3/4/5 bracket the plateau; ramp55
        # holds narrow for a second turn and regresses (see agent_changes.md).
        "ramp_flat": AgentConfig(list_size_ramp=(10,)),
        "ramp3": AgentConfig(list_size_ramp=(3, 10)),
        "ramp4": AgentConfig(list_size_ramp=(4, 10)),
        "ramp5": AgentConfig(list_size_ramp=(5, 10)),
        "ramp55": AgentConfig(list_size_ramp=(5, 5, 10)),
        # Rerank weight mixture: the near-miss anatomy (rerank_signals.md) shows
        # every remaining public rank loss sits in a tie-break regime where the
        # retrieval score picks the impostor 33/33 (BM25 length normalization
        # favours thin listings) while popularity picks the target 31/33 yet is
        # weighted 0.02 against 1.0. These rows bracket the popularity weight;
        # tools/fit_weights.py searches the full mixture.
        # pop002 is the pre-change shipped value; 0.4 is the new default, with
        # 0.3/0.5 as its measured plateau neighbours.
        "pop002": AgentConfig(rerank=RerankConfig(popularity_weight=0.02)),
        "pop010": AgentConfig(rerank=RerankConfig(popularity_weight=0.10)),
        "pop030": AgentConfig(rerank=RerankConfig(popularity_weight=0.30)),
        "pop040": AgentConfig(rerank=RerankConfig(popularity_weight=0.40)),
        "pop050": AgentConfig(rerank=RerankConfig(popularity_weight=0.50)),
        # The coordinate-ascent dev argmax (tools/fit_weights.py): higher on
        # dev/holdout/public, regresses the hard set - kept as a row so the
        # trade-off stays reproducible, not as a default.
        "weights_argmax": AgentConfig(rerank=RerankConfig(
            popularity_weight=0.8, retrieval_weight=0.1, facet_weight=0.5,
            tail_weight=1.2, facet_conflict_weight=0.0)),
        # Pool-aware clarification wording (src/phrasing.py). ask_attribute is
        # unchanged and the simulator never reads `message`, so these two must
        # score bit-for-bit identically - the row exists to prove that.
        "natural_off": AgentConfig(natural_questions=False),
        "natural_on": AgentConfig(natural_questions=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Configuration sweep over the public set")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    parser.add_argument("--configs", default="", help="comma-separated subset of config names")
    parser.add_argument("--output", default="", help="optional path for the full JSON results")
    args = parser.parse_args()

    samples = split_samples(load_jsonl(args.dataset), args.split)
    catalog_ids, categories, products = catalog_index(args.catalog)

    configs = build_configs(args.catalog)
    if args.configs:
        wanted = [name.strip() for name in args.configs.split(",") if name.strip()]
        missing = [name for name in wanted if name not in configs]
        if missing:
            parser.error(f"unknown config(s): {', '.join(missing)}")
        configs = {name: configs[name] for name in wanted}

    print(f"split={args.split}  sessions={len(samples)}\n")
    header = f"{'config':<12} {'hit@10':>7} {'mrr':>7} {'mttc':>6} {'eff':>6} {'score':>7}  {'time':>6}"
    print(header)
    print("-" * len(header))

    results: dict[str, dict] = {}
    for name, config in configs.items():
        started = time.time()
        result = evaluate(Agent(args.catalog, config), samples, catalog_ids, categories, products)
        results[name] = result
        print(
            f"{name:<12} {result['hit_rate_at_10']:>7.3f} {result['mrr']:>7.3f} "
            f"{result['mttc']:>6.2f} {result['efficiency']:>6.3f} "
            f"{result['recommended_technical_score']:>7.4f}  {time.time() - started:>5.0f}s",
            flush=True,
        )

    print("\nper-scenario score components")
    for name, result in results.items():
        parts = " ".join(
            f"{scenario[:4]}={metrics['hit_rate_at_10']:.2f}/{metrics['mrr']:.2f}"
            for scenario, metrics in sorted(result["scenario_metrics"].items())
        )
        print(f"  {name:<12} {parts}   (hit/mrr)")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
