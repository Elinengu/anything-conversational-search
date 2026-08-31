"""Gaussian-Process Bayesian optimization over the 7 fitted rerank weights.

Method 2 of the weight-learning experiment (branch kwongweng_fit_bo, docs/team/bo.md).

Coordinate ascent (tools/fit_weights.py) walks one axis at a time and cannot
follow a diagonal ridge - if the best move needs two weights to change together
it stalls. A GP surrogate models the whole 7-D response surface and its
acquisition function proposes joint moves. Every evaluation here is a real,
consistent `evaluate()` on the dev split, so - unlike a pairwise/listwise
surrogate fit on cached feature vectors - there is no weight-dependent-transcript
mismatch to iterate away.

The objective (dev `recommended_technical_score`) is a step function of a ranking,
so the GP kernel carries a WhiteKernel: without an explicit noise term the GP
interpolates every spike and the acquisition chases artefacts.

Protocol (docs/team/signal_descriptions.md house rules): fit on dev ONLY;
span_weight fixed at 1.0; holdout + hard are one-shot gates; what is presented
is a rounded, plateau-checked vector, never the raw dev argmax.

Thread-pin the run (the GP fit otherwise spawns ~18 BLAS threads):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      nice -n 10 python3 tools/fit_weights_bo.py --variant both --cv \
      --out /path/to/scratch/bo/result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rerank import RerankConfig  # noqa: E402
from tools.fit_common import (  # noqa: E402
    FITTED,
    Scorer,
    load_all,
    scalar_from_sessions,
)

# Box in real weight units. span_weight is not here - it stays 1.0.
BOX = {
    "popularity_weight": (0.0, 2.0),
    "retrieval_weight": (0.0, 2.0),
    "facet_weight": (0.0, 2.0),
    "category_weight": (0.0, 2.0),
    "tail_weight": (0.0, 2.0),
    "pair_weight": (0.0, 2.0),
    "facet_conflict_weight": (0.0, 1.5),
}
LO = np.array([BOX[k][0] for k in FITTED])
HI = np.array([BOX[k][1] for k in FITTED])


def to_weights(x: np.ndarray) -> dict:
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    real = LO + x * (HI - LO)
    return {k: float(real[i]) for i, k in enumerate(FITTED)}


def to_unit(weights: dict) -> np.ndarray:
    real = np.array([weights[k] for k in FITTED])
    return (real - LO) / (HI - LO)


def make_folds(dev_samples: list) -> dict:
    """sample_id -> fold in {0,1,2}. Stratify by scenario_type, sort by
    sample_id, round-robin."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for s in dev_samples:
        grouped[s["scenario_type"]].append(s["sample_id"])
    fold_of: dict[str, int] = {}
    for st in sorted(grouped):
        for i, sid in enumerate(sorted(grouped[st])):
            fold_of[sid] = i % 3
    return fold_of


def cv_score(weights: dict, scorer: Scorer, fold_of: dict) -> float:
    sess = scorer.sessions(weights)
    folds: list[list] = [[], [], []]
    for s in sess:
        folds[fold_of[s["sample_id"]]].append(s)
    return float(np.mean([scalar_from_sessions(f)["score"] for f in folds]))


def build_gp() -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0) * Matern(length_scale=[0.3] * 7, nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
    )
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=5, alpha=1e-10
    )


def expected_improvement(X: np.ndarray, gp: GaussianProcessRegressor,
                         f_best: float, xi: float = 0.01) -> np.ndarray:
    X = np.atleast_2d(X)
    mu, sigma = gp.predict(X, return_std=True)
    ei = np.zeros_like(mu)
    mask = sigma > 0.0
    imp = mu[mask] - f_best - xi
    z = imp / sigma[mask]
    ei[mask] = imp * norm.cdf(z) + sigma[mask] * norm.pdf(z)
    return ei


def propose(gp: GaussianProcessRegressor, f_best: float,
            rng: np.random.Generator) -> np.ndarray:
    cand = rng.random((5000, 7))
    eiv = expected_improvement(cand, gp, f_best)
    best_x = cand[int(np.argmax(eiv))]
    best_val = float(np.max(eiv))
    top = cand[np.argsort(eiv)[-5:]]
    for x0 in top:
        res = minimize(
            lambda x: -float(expected_improvement(x, gp, f_best)[0]),
            x0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * 7,
        )
        if -res.fun > best_val:
            best_val = -res.fun
            best_x = np.clip(res.x, 0.0, 1.0)
    return np.clip(best_x, 0.0, 1.0)


def run_bo(objective, X_init: list, y_init: list, iters: int, seed: int,
           label: str) -> dict:
    rng = np.random.default_rng(seed + 1)
    X = [np.asarray(x, float) for x in X_init]
    y = list(y_init)
    traj = []
    for i, (x, yi) in enumerate(zip(X, y)):
        traj.append({"iter": i - len(X_init) + 1, "phase": "init",
                     "score": yi, "weights": to_weights(x)})
    started = time.time()
    for it in range(1, iters + 1):
        gp = build_gp()
        gp.fit(np.array(X), np.array(y))
        f_best = max(y)
        x_next = propose(gp, f_best, rng)
        y_next = objective(to_weights(x_next))
        X.append(x_next)
        y.append(y_next)
        w = to_weights(x_next)
        traj.append({"iter": it, "phase": "ei", "score": y_next, "weights": w})
        best_so_far = max(y)
        print(f"[{label}] iter {it:3d}  score {y_next:.6f}  best {best_so_far:.6f}  "
              f"({time.time() - started:.0f}s)  weights "
              f"{ {k: round(v, 3) for k, v in w.items()} }", flush=True)
    best_idx = int(np.argmax(y))
    return {
        "label": label,
        "raw_best": {"score": y[best_idx], "weights": to_weights(X[best_idx])},
        "n_obs": len(y),
        "trajectory": traj,
        "observations": [{"y": y[j], "weights": to_weights(X[j])} for j in range(len(y))],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--variant", default="both", choices=["plain", "default", "both"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--cv", action="store_true",
                    help="also run a BO whose objective is the 3-fold stratified dev CV mean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="JSON path for the result")
    args = ap.parse_args()

    data = load_all(args.catalog)
    fold_of = make_folds(data["dev"])
    variants = ["plain", "default"] if args.variant == "both" else [args.variant]

    # Init design: 10 Latin-hypercube points in [0,1]^7 + the baseline vector.
    lhs = LatinHypercube(d=7, seed=args.seed).random(10)
    baseline_unit = to_unit({k: getattr(RerankConfig(), k) for k in FITTED})
    X_init = [row for row in lhs] + [baseline_unit]

    out: dict = {"seed": args.seed, "iters": args.iters, "box": BOX, "variants": {}}
    for variant in variants:
        scorer = Scorer(data, variant, "dev")
        t0 = time.time()
        # Evaluate the init design once; both objectives reuse the cached sessions.
        y_primary_init = [scorer.score(to_weights(x)) for x in X_init]
        print(f"[{variant}] init design evaluated ({scorer.evals} evals, "
              f"{time.time() - t0:.0f}s)", flush=True)

        runs: dict = {}
        runs["dev"] = run_bo(scorer.score, X_init, y_primary_init,
                             args.iters, args.seed, f"{variant}/dev")
        if args.cv:
            y_cv_init = [cv_score(to_weights(x), scorer, fold_of) for x in X_init]
            runs["cv"] = run_bo(
                lambda w: cv_score(w, scorer, fold_of),
                X_init, y_cv_init, args.iters, args.seed, f"{variant}/cv")

        out["variants"][variant] = {
            "total_dev_evals": scorer.evals,
            "runs": runs,
        }
        print(f"[{variant}] done: {scorer.evals} unique dev evaluations, "
              f"{time.time() - t0:.0f}s total", flush=True)

    print("\n=== RAW DEV ARGMAX ===", flush=True)
    for variant, v in out["variants"].items():
        for obj, r in v["runs"].items():
            print(f"{variant:8s} {obj:4s}  score {r['raw_best']['score']:.6f}  "
                  f"weights { {k: round(x, 4) for k, x in r['raw_best']['weights'].items()} }",
                  flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
