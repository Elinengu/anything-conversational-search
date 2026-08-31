"""Method 4 - interpret the trained LambdaMART ranker.

Gain-based feature importances, an optional SHAP summary, and a scan for the
capacity the linear reranker lacks: a feature interaction the tree ensemble uses
(e.g. "popularity matters only when span-gap ~ 0 and rating_number is high").
The interaction rule is the deliverable even on a negative gate result.

    python3 tools/lm_explain.py --queries /scratch/queries.parquet \
        --model-dir /scratch/lm_model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.fit_weights_lambdamart import FEATURES, load, session_split, zscore


def _load_model(model_dir: Path):
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(model_dir / "lambdamart.txt"))
    stats = json.loads((model_dir / "zstats.json").read_text())
    return booster, stats


def importances(booster) -> list[tuple[str, float]]:
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    rows = sorted(zip(FEATURES, gain, split), key=lambda r: -r[1])
    tot = gain.sum() or 1.0
    out = []
    print(f"\n{'feature':<20} {'gain':>12} {'gain%':>7} {'splits':>8}")
    for name, g, s in rows:
        print(f"{name:<20} {g:>12.1f} {100 * g / tot:>6.1f}% {s:>8}")
        out.append((name, float(g)))
    return out


def partial_dependence(booster, stats, df, feat, cond=None, grid=12):
    """Mean model score across the data as `feat` sweeps its range, optionally
    within a boolean row mask `cond`. Reports the score span (max-min)."""
    sub = df if cond is None else df[cond]
    if len(sub) < 50:
        return None
    Z = zscore(sub, stats, FEATURES)
    fi = FEATURES.index(feat)
    lo, hi = np.percentile(sub[feat], [2, 98])
    xs = np.linspace(lo, hi, grid)
    m, s = stats[feat]["mean"], stats[feat]["std"] or 1.0
    ys = []
    for x in xs:
        Zc = Z.copy()
        Zc[:, fi] = (x - m) / s
        ys.append(float(booster.predict(Zc).mean()))
    return {"feature": feat, "x": [float(v) for v in xs], "y": ys,
            "span": max(ys) - min(ys), "n": int(len(sub))}


def interaction_scan(booster, stats, df):
    """For each (A, B) pair, compare the partial-dependence span of A when B is
    low vs when B is high. A large gap => the ensemble's response to A depends on
    B - the non-linear capacity the linear reranker cannot express."""
    print("\n=== interaction scan (PD span of A | B-low vs B-high) ===")
    pairs = [
        ("f_popularity", "f_span_gap_to_max"),
        ("f_popularity", "f_rating_number"),
        ("f_retr_norm", "f_span_gap_to_max"),
        ("f_retr_rank", "f_rating_number"),
        ("f_retr_rank", "f_span_gap_to_max"),
        ("f_facet_agree", "f_span_gap_to_max"),
    ]
    found = []
    for a, b in pairs:
        blo, bhi = np.percentile(df[b], [25, 75])
        pd_lo = partial_dependence(booster, stats, df, a, df[b] <= blo)
        pd_hi = partial_dependence(booster, stats, df, a, df[b] >= bhi)
        if not pd_lo or not pd_hi:
            continue
        gap = abs(pd_hi["span"] - pd_lo["span"])
        print(f"  {a:<18} | {b:<18}  span(Blow)={pd_lo['span']:.3f}  "
              f"span(Bhigh)={pd_hi['span']:.3f}  |delta|={gap:.3f}")
        found.append({"A": a, "B": b, "span_B_low": pd_lo["span"],
                      "span_B_high": pd_hi["span"], "abs_delta": gap})
    found.sort(key=lambda r: -r["abs_delta"])
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--variant", default="default")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    md = Path(args.model_dir)
    booster, stats = _load_model(md)
    df = load(args.queries)
    df = df[df["variant"] == args.variant].copy()
    _tr, val = session_split(df, 0.15, 0)
    print(f"explain on val: rows={len(val)}  queries={val['qid'].nunique()}")

    result = {"importances": importances(booster)}

    print("\n=== 1-D partial dependence (score span over each feature) ===")
    pds = []
    for f in FEATURES:
        r = partial_dependence(booster, stats, val, f)
        if r:
            print(f"  {f:<20} span={r['span']:.3f}")
            pds.append(r)
    result["partial_dependence"] = pds
    result["interactions"] = interaction_scan(booster, stats, val)

    try:
        import shap
        Z = zscore(val.sample(min(4000, len(val)), random_state=0), stats, FEATURES)
        expl = shap.TreeExplainer(booster)
        sv = expl.shap_values(Z)
        order = np.argsort(-np.abs(sv).mean(0))
        print("\n=== SHAP mean|value| ===")
        for i in order:
            print(f"  {FEATURES[i]:<20} {np.abs(sv[:, i]).mean():.4f}")
        result["shap_mean_abs"] = {FEATURES[i]: float(np.abs(sv[:, i]).mean())
                                   for i in order}
    except Exception as e:  # shap optional
        print(f"\n(shap unavailable: {e})")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
