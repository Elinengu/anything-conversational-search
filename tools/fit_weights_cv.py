"""Coordinate ascent over the rerank weights with a VARIANCE-CONTROLLED accept rule.

This is `tools/fit_weights.py` (greedy coordinate ascent, direct on the official
score) with exactly one substantive change: the rule that decides whether a
weight move `current -> trial` is accepted. Everything else - the absolute
`GRID`, the `FITTED` cycle order, `span_weight` fixed at 1.0, the current value
always retained, `EPSILON = 1e-6`, the weight-tuple cache, the shipped-defaults
initial vector - is unchanged.

Motivation (docs/team/rerank_signals.md sec. 10, "change 12"): plain coordinate
ascent moved the dev argmax to `popularity 0.8, retrieval 0.1, facet 0.5,
tail 1.2, conflict 0`. Holdout confirmed the *direction* (+0.019) but the
adversarial hard set REGRESSED by 0.016. The accept decision had no variance
control - a dev gain of any size, however concentrated in a few sessions, was
accepted. This tool adds a variance filter on that decision.

Two rules (pick with `--bootstrap`):

  k-fold (default) - partition the 120 dev sessions into 5 stratified folds
    (group by scenario_type, sort each group by sample_id, round-robin
    session j of a group -> fold j%5). One `evaluate()` on all 120 dev
    sessions yields every per-session outcome; `scalar_from_sessions` is then
    recomputed on each fold's subset for FREE. Accept `current -> trial` iff
      (1) mean_oof(trial) > mean_oof(current) + EPSILON, AND
      (2) fold_score(trial)[i] >= fold_score(current)[i] - 1e-9 for at least
          4 of the 5 folds (no regression on >= k-1 folds).

  --bootstrap N (default off, N=20) - rng = numpy.random.default_rng(0); draw
    N index-arrays, each 120 draws with replacement from range(120). For each
    resample b, diff_b = score(trial sessions[idx_b]) - score(current
    sessions[idx_b]). Accept iff numpy.percentile(diffs, 5) > 0 (5th-percentile
    lower bound of the paired difference is positive). Also free per candidate
    (reuses the two cached `Scorer.sessions()` results).

HONESTY NOTE: this is NOT classical cross-validation. There is no per-fold model
training - the "model" is a discrete weight choice. It is a fold-agreement /
paired-subsample ACCEPTANCE RULE. Its claim is robustness of the accept
decision (the thing change 12 lacked), not an unbiased generalisation estimate.

    OMP_NUM_THREADS=1 python3 tools/fit_weights_cv.py --variant both
    OMP_NUM_THREADS=1 python3 tools/fit_weights_cv.py --variant both --bootstrap 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rerank import RerankConfig  # noqa: E402
from tools.fit_common import FITTED, Scorer, load_all, scalar_from_sessions  # noqa: E402
from tools.fit_weights import GRID  # noqa: E402  (identical absolute grid)

EPSILON = 1e-6
N_FOLDS = 5


def make_folds(samples: list[dict], n_folds: int = N_FOLDS) -> list[int]:
    """fold[i] = fold id of dev session i (samples are already sorted by sample_id).

    Group by scenario_type, sort each group by sample_id, round-robin the j-th
    member of a group into fold j % n_folds.
    """
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        grouped[s["scenario_type"]].append(i)
    fold = [0] * len(samples)
    for scenario in sorted(grouped):
        rows = sorted(grouped[scenario], key=lambda i: samples[i]["sample_id"])
        for j, i in enumerate(rows):
            fold[i] = j % n_folds
    return fold


def fold_scores(sessions: list, fold: list[int], n_folds: int = N_FOLDS) -> list[float]:
    out = []
    for f in range(n_folds):
        subset = [sessions[i] for i in range(len(sessions)) if fold[i] == f]
        out.append(scalar_from_sessions(subset)["score"])
    return out


class KFoldRule:
    name = "k-fold"

    def __init__(self, scorer: Scorer, fold: list[int]):
        self.scorer, self.fold = scorer, fold

    def diag(self, weights: dict) -> dict:
        fs = fold_scores(self.scorer.sessions(weights), self.fold)
        return {"folds": fs, "mean": float(np.mean(fs)),
                "score": self.scorer.score(weights)}

    def accept(self, trial: dict, current: dict) -> tuple[bool, dict]:
        ct = self.diag(current)
        tt = self.diag(trial)
        n_ok = sum(1 for a, b in zip(tt["folds"], ct["folds"]) if a >= b - 1e-9)
        mean_ok = tt["mean"] > ct["mean"] + EPSILON
        ok = bool(mean_ok and n_ok >= N_FOLDS - 1)
        return ok, {"trial": tt, "current": ct, "n_ok": n_ok, "mean_ok": mean_ok}


class BootstrapRule:
    name = "bootstrap"

    def __init__(self, scorer: Scorer, n: int, n_sessions: int):
        self.scorer = scorer
        rng = np.random.default_rng(0)
        self.idx = [rng.integers(0, n_sessions, size=n_sessions) for _ in range(n)]
        self.n = n

    def diag(self, weights: dict) -> dict:
        return {"score": self.scorer.score(weights)}

    def accept(self, trial: dict, current: dict) -> tuple[bool, dict]:
        ts = self.scorer.sessions(trial)
        cs = self.scorer.sessions(current)
        diffs = []
        for idx_b in self.idx:
            t = scalar_from_sessions([ts[i] for i in idx_b])["score"]
            c = scalar_from_sessions([cs[i] for i in idx_b])["score"]
            diffs.append(t - c)
        p5 = float(np.percentile(diffs, 5))
        ok = p5 > 0.0
        return ok, {"p5": p5, "mean_diff": float(np.mean(diffs)),
                    "trial": {"score": self.scorer.score(trial)},
                    "current": {"score": self.scorer.score(current)}}


def run_variant(data: dict, variant: str, rule_name: str, bootstrap_n: int,
                max_cycles: int) -> dict:
    scorer = Scorer(data, variant, "dev")
    fold = make_folds(data["dev"])
    if rule_name == "bootstrap":
        rule = BootstrapRule(scorer, bootstrap_n, len(data["dev"]))
    else:
        rule = KFoldRule(scorer, fold)

    weights = {name: getattr(RerankConfig(), name) for name in FITTED}
    started = time.time()
    trajectory = []

    start_fs = fold_scores(scorer.sessions(weights), fold)
    start_score = scalar_from_sessions(scorer.sessions(weights))["score"]
    print(f"[{variant}/{rule.name}] start   score {start_score:.6f}   "
          f"folds {[round(x, 4) for x in start_fs]}   weights {json.dumps(weights)}",
          flush=True)

    for cycle in range(1, max_cycles + 1):
        improved = False
        for name in FITTED:
            candidates = sorted(set(GRID) | {weights[name]})
            ref = dict(weights)          # running best for this weight
            best_value = weights[name]
            for value in candidates:
                if value == weights[name]:
                    continue
                trial = dict(weights)
                trial[name] = value
                ok, info = rule.accept(trial, ref)
                if not ok and _would_plain_ca_accept(info):
                    fs = info["trial"].get("folds")
                    regressed = ([i for i, (a, b) in enumerate(
                        zip(info["trial"]["folds"], info["current"]["folds"]))
                        if a < b - 1e-9] if fs is not None else None)
                    rej = {"cycle": cycle, "weight": name,
                           "from": ref[name], "to": value,
                           "trial_score": info["trial"]["score"],
                           "ref_score": info["current"]["score"],
                           "folds_regressed": regressed,
                           "p5": info.get("p5")}
                    trajectory.append({"rejected_but_plain_ca_accepts": rej})
                    print(f"[{variant}/{rule.name}] cycle {cycle}  REJECT "
                          f"{name} {ref[name]:g}->{value:g}  "
                          f"(plain CA would accept: dev {info['current']['score']:.5f}"
                          f"->{info['trial']['score']:.5f})  "
                          f"folds_regressed={regressed}  p5={info.get('p5')}",
                          flush=True)
                if ok:
                    accepted = {"cycle": cycle, "weight": name,
                                "from": ref[name], "to": value,
                                "trial_score": info["trial"]["score"],
                                "folds": info["trial"].get("folds"),
                                "p5": info.get("p5")}
                    trajectory.append({"accepted": accepted})
                    print(f"[{variant}/{rule.name}] cycle {cycle}  ACCEPT "
                          f"{name} {ref[name]:g}->{value:g}   "
                          f"score {info['trial']['score']:.6f}   "
                          f"folds {[round(x, 4) for x in info['trial']['folds']] if info['trial'].get('folds') else '-'}"
                          f"   p5={info.get('p5')}   "
                          f"({scorer.evals} evals, {time.time() - started:.0f}s)",
                          flush=True)
                    ref = trial
                    best_value = value
                    improved = True
            if best_value != weights[name]:
                weights[name] = best_value
        if not improved:
            print(f"[{variant}/{rule.name}] cycle {cycle}: no improvement, stopping",
                  flush=True)
            break

    final_sessions = scorer.sessions(weights)
    final_fs = fold_scores(final_sessions, fold)
    final_score = scalar_from_sessions(final_sessions)["score"]
    print(f"[{variant}/{rule.name}] final   score {final_score:.6f}   "
          f"folds {[round(x, 4) for x in final_fs]}   evals {scorer.evals}   "
          f"{time.time() - started:.0f}s\nweights {json.dumps(weights, indent=2)}",
          flush=True)
    return {"variant": variant, "rule": rule.name, "bootstrap_n": bootstrap_n,
            "weights_raw": weights, "score": final_score, "folds": final_fs,
            "start_score": start_score, "start_folds": start_fs,
            "evals": scorer.evals, "trajectory": trajectory}


def _would_plain_ca_accept(info: dict) -> bool:
    """Plain fit_weights.py accepts iff dev scalar improves past EPSILON."""
    return info["trial"]["score"] > info["current"]["score"] + EPSILON


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--variant", default="both",
                        choices=["plain", "default", "both"])
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="0 = k-fold rule; N>0 = bootstrap rule with N resamples")
    parser.add_argument("--out", default="", help="optional JSON path for results")
    args = parser.parse_args()

    rule_name = "bootstrap" if args.bootstrap > 0 else "k-fold"
    bootstrap_n = args.bootstrap if args.bootstrap > 0 else 20
    variants = ["plain", "default"] if args.variant == "both" else [args.variant]

    data = load_all(args.catalog)
    results = []
    for variant in variants:
        results.append(run_variant(data, variant, rule_name,
                                   bootstrap_n if rule_name == "bootstrap" else 0,
                                   args.max_cycles))

    if args.out:
        Path(args.out).write_text(json.dumps({"rule": rule_name, "results": results},
                                             indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
