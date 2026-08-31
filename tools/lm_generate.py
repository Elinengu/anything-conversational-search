"""Method 4 - synthesise a training session for (almost) every catalog product.

One session per seed target (tools/lm_targets.py). Each session is a dict with
the same shape the evaluator's ``materialize_hidden_fields`` needs -
``sample_id``, ``scenario_type``, ``ground_truth``, ``user_profile`` - plus
``batch`` ("coop" | "stress") and ``stress_spec`` (a tools/stress_harness.py
spec string, "" for coop).

Cooperative batch: the 4 scenario_types cycled in the public set's proportions
(buying 80 / browsing 80 / intent_override 30 / boundary 10 = 40/40/15/5 %).

Stressed batch (~22 % of targets, disjoint from coop): run later through the
dense_rerank stress harness's StressCustomer. Four specs are cycled and the
scenario is pinned to the one that makes each stressor load-bearing:

    paraphrase:heavy                 -> scenario cycled (any)
    browse-gated                     -> browsing
    decoy                            -> intent_override
    paraphrase:heavy+browse-gated    -> browsing

Nothing is rolled out here - tools/lm_snapshot.py does the heavy Agent rollout.
This step is pure, deterministic (RNG seeded per parent_asin) and fast.

    python3 tools/lm_generate.py --all   --out /scratch/sessions.jsonl
    python3 tools/lm_generate.py --limit 200 --out /scratch/sessions_smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.lm_targets import build as build_seeds  # noqa: E402

# Public-set scenario proportions (data/public_set.jsonl: 80/80/30/10).
SCENARIO_CYCLE = (
    ["buying"] * 8 + ["browsing"] * 8 + ["intent_override"] * 3 + ["boundary"] * 1
)

STRESS_SPECS = ("paraphrase:heavy", "browse-gated", "decoy",
                "paraphrase:heavy+browse-gated")
# scenario pinned so the stressor actually bites; None = cycle
STRESS_SCENARIO = {
    "paraphrase:heavy": None,
    "browse-gated": "browsing",
    "decoy": "intent_override",
    "paraphrase:heavy+browse-gated": "browsing",
}

STRESS_FRACTION = 0.22

_PREF_TAGS = ["fit", "comfort", "durability", "style", "material", "color",
             "quality", "value", "size", "warmth", "breathability"]
_FREQ = ["1-2 prior purchases", "3-4 prior purchases", "5+ prior purchases"]
_RATING_STYLE = ["usually positive", "critical", "balanced", "generous"]


def synth_profile(rng: random.Random) -> dict:
    tags = rng.sample(_PREF_TAGS, rng.choice([2, 3]))
    avg = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0])
    style = rng.choice(_RATING_STYLE)
    return {
        "average_prior_rating": avg,
        "preference_tags": tags,
        "purchase_frequency": rng.choice(_FREQ),
        "rating_style": style,
        "summary": f"Prior purchases emphasize {', '.join(tags)}; ratings are {style}.",
    }


def make_sessions(seeds: list[str], stress_fraction: float) -> list[dict]:
    sessions: list[dict] = []
    n_stress = int(round(len(seeds) * stress_fraction))
    # deterministic disjoint split: every ~1/fraction-th target is stressed
    stride = max(1, round(1.0 / stress_fraction)) if stress_fraction > 0 else 10 ** 9
    coop_i = 0
    stress_i = 0
    for idx, asin in enumerate(seeds):
        rng = random.Random(asin)
        is_stress = stress_fraction > 0 and (idx % stride == 0) and stress_i < n_stress
        if is_stress:
            spec = STRESS_SPECS[stress_i % len(STRESS_SPECS)]
            pinned = STRESS_SCENARIO[spec]
            scenario = pinned or SCENARIO_CYCLE[stress_i % len(SCENARIO_CYCLE)]
            sessions.append({
                "sample_id": f"lm_stress_{stress_i:06d}_{asin}",
                "scenario_type": scenario,
                "ground_truth": {"parent_asin": asin},
                "user_profile": synth_profile(rng),
                "batch": "stress",
                "stress_spec": spec,
            })
            stress_i += 1
        else:
            scenario = SCENARIO_CYCLE[coop_i % len(SCENARIO_CYCLE)]
            sessions.append({
                "sample_id": f"lm_coop_{coop_i:06d}_{asin}",
                "scenario_type": scenario,
                "ground_truth": {"parent_asin": asin},
                "user_profile": synth_profile(rng),
                "batch": "coop",
                "stress_spec": "",
            })
            coop_i += 1
    return sessions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stress-fraction", type=float, default=STRESS_FRACTION)
    args = ap.parse_args()

    seeds, _ = build_seeds(args.catalog)
    if not args.all:
        lim = args.limit or 200
        # spread the smoke sample across the catalog, not just the head
        step = max(1, len(seeds) // lim)
        seeds = seeds[::step][:lim]

    sessions = make_sessions(seeds, args.stress_fraction)

    from collections import Counter
    by_batch = Counter(s["batch"] for s in sessions)
    by_spec = Counter(s["stress_spec"] for s in sessions if s["batch"] == "stress")
    by_scen = Counter(s["scenario_type"] for s in sessions)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for s in sessions:
            fh.write(json.dumps(s) + "\n")

    print(f"sessions      : {len(sessions)}")
    print(f"by batch      : {dict(by_batch)}")
    print(f"stress specs  : {dict(by_spec)}")
    print(f"by scenario   : {dict(by_scen)}")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
