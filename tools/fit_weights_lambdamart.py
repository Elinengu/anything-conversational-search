"""Method 4 - non-linear (gradient-boosted tree) ranker on synthetic sessions.

Trains a LambdaMART ranker (LightGBM LGBMRanker, objective="lambdarank") on the
per-turn ranking queries snapshotted by tools/lm_snapshot.py over ~50k synthetic
sessions spanning the whole 50,000-product catalog (cooperative + stress-harness
customers). Tests whether non-linear capacity + a distribution-diverse training
set 400x larger than the 120-session dev split can beat the shipped linear
reranker where four linear/local methods regressed the adversarial hard set.

Also fits the SAME-data linear control (Method 1's exact pairwise LogisticRegression
on feature differences) so the write-up can separate *capacity* (LambdaMART vs
linear) from *data* (150k queries vs M1's few hundred).

Fit / model selection on synthetic data + a session-grouped val split only. The
real holdout / public / hard / stress cells are one-shot gates (tools/fit_common).

    python3 tools/fit_weights_lambdamart.py --queries /scratch/queries.parquet \
        --out-dir /scratch/lm_model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The 8 shipped reranker features + 6 extra raw / positional signals a tree can
# split on but the linear reranker cannot use.
FEATURES = [
    "f_span_cov", "f_pair_cov", "f_retr_norm", "f_popularity", "f_facet_agree",
    "f_category", "f_tail", "f_facet_conflict",
    "f_rating_number", "f_average_rating", "f_text_len",
    "f_span_gap_to_max", "f_retr_rank", "f_pool_size",
]
# M1's linear control only uses the 8 shipped features (it produces a RerankConfig
# weight vector, which has no slot for the extras).
LINEAR_FEATURES = FEATURES[:8]


def load(queries: str) -> pd.DataFrame:
    df = pd.read_parquet(queries)
    df["qid"] = df["session_id"].astype(str) + "|" + df["turn"].astype(str)
    return df


def session_split(df: pd.DataFrame, val_frac: float, seed: int):
    """85/15 by session_id, stratified by (scenario_type, batch). A session's
    queries never cross the split."""
    rng = np.random.default_rng(seed)
    sess = df.groupby("session_id")[["scenario_type", "batch"]].first()
    val_ids: set[str] = set()
    for _key, grp in sess.groupby(["scenario_type", "batch"]):
        ids = grp.index.to_numpy()
        rng.shuffle(ids)
        cut = int(round(len(ids) * val_frac))
        val_ids.update(ids[:cut].tolist())
    is_val = df["session_id"].isin(val_ids)
    return df[~is_val].copy(), df[is_val].copy()


def zstats(train: pd.DataFrame, cols: list[str]) -> dict:
    return {c: {"mean": float(train[c].mean()),
                "std": float(train[c].std() or 1.0)} for c in cols}


def zscore(df: pd.DataFrame, stats: dict, cols: list[str]) -> np.ndarray:
    return np.column_stack([
        (df[c].to_numpy() - stats[c]["mean"]) / (stats[c]["std"] or 1.0)
        for c in cols
    ])


def _group_sizes(df: pd.DataFrame) -> np.ndarray:
    # df must be ordered by qid; return contiguous run lengths
    return df.groupby("qid", sort=False).size().to_numpy()


def fit_lambdamart(train, val, zst, out_dir: Path, num_leaves: int, seed: int):
    import lightgbm as lgb

    train = train.sort_values("qid", kind="stable")
    val = val.sort_values("qid", kind="stable")
    Xtr, Xva = zscore(train, zst, FEATURES), zscore(val, zst, FEATURES)
    ytr, yva = train["is_target"].to_numpy(), val["is_target"].to_numpy()
    gtr, gva = _group_sizes(train), _group_sizes(val)

    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=1000,
        learning_rate=0.05, num_leaves=num_leaves, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
        label_gain=[0, 1], lambdarank_truncation_level=10,
        random_state=seed, n_jobs=1, verbose=-1,
    )
    model.fit(
        Xtr, ytr, group=gtr, eval_set=[(Xva, yva)], eval_group=[gva],
        eval_at=[10], callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    traj = model.evals_result_["valid_0"]["ndcg@10"]
    print(f"[lambdamart] best_iter={model.best_iteration_}  "
          f"val ndcg@10: start={traj[0]:.5f} best={max(traj):.5f}")
    model.booster_.save_model(str(out_dir / "lambdamart.txt"),
                              num_iteration=model.best_iteration_)
    return {"best_iteration": int(model.best_iteration_ or 0),
            "num_leaves": num_leaves,
            "val_ndcg10_trajectory": [round(x, 6) for x in traj],
            "val_ndcg10_best": round(max(traj), 6)}


def fit_linear_control(train, val, out_dir: Path, top_neg: int, seed: int):
    """Method 1 verbatim: pairwise LogisticRegression on phi(target)-phi(neg)
    over the 8 shipped features, on THIS (400x larger) training set."""
    from sklearn.linear_model import LogisticRegression

    diffs = []
    for _qid, g in train.groupby("qid", sort=False):
        tgt = g[g["is_target"] == 1]
        if tgt.empty:
            continue
        phi_t = tgt[LINEAR_FEATURES].to_numpy()[0]
        negs = g[g["is_target"] == 0][LINEAR_FEATURES].to_numpy()[:top_neg]
        for phi_n in negs:
            diffs.append(phi_t - phi_n)
    D = np.vstack(diffs)
    X = np.vstack([D, -D])
    y = np.array([1] * len(D) + [0] * len(D))
    best = None
    for C in (0.1, 1.0, 10.0):
        clf = LogisticRegression(fit_intercept=False, C=C, max_iter=5000,
                                 class_weight="balanced")
        clf.fit(X, y)
        c = clf.coef_[0]
        scale = c[0]
        if scale <= 0:
            print(f"[linear C={C}] degenerate span coef {scale:.3f} - skip")
            continue
        w = c / scale
        weights = {
            "popularity_weight": max(0.0, float(w[3])),
            "retrieval_weight": max(0.0, float(w[2])),
            "facet_weight": max(0.0, float(w[4])),
            "category_weight": max(0.0, float(w[5])),
            "tail_weight": max(0.0, float(w[6])),
            "pair_weight": max(0.0, float(w[1])),
            "facet_conflict_weight": max(0.0, float(-w[7])),
        }
        print(f"[linear C={C}] weights={weights}")
        if best is None:
            best = (C, weights, [float(x) for x in c])
    (out_dir / "linear_control.json").write_text(
        json.dumps({"C": best[0], "weights": best[1], "coef": best[2],
                    "n_pairs": int(len(D))}, indent=2) + "\n")
    return {"C": best[0], "weights": best[1], "n_pairs": int(len(D))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--num-leaves", type=int, default=31)
    ap.add_argument("--top-neg", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="default", choices=["default", "plain"],
                    help="which agent-config slice of the queries to train on")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(args.queries)
    df = df[df["variant"] == args.variant].copy()
    print(f"rows={len(df)}  queries={df['qid'].nunique()}  "
          f"sessions={df['session_id'].nunique()}  "
          f"target_rows={int(df['is_target'].sum())}")

    train, val = session_split(df, args.val_frac, args.seed)
    print(f"train queries={train['qid'].nunique()}  val queries={val['qid'].nunique()}")

    zst = zstats(train, FEATURES)
    (out_dir / "zstats.json").write_text(json.dumps(zst, indent=2) + "\n")

    summary = {"variant": args.variant, "rows": int(len(df)),
               "queries": int(df["qid"].nunique()),
               "sessions": int(df["session_id"].nunique())}
    summary["lambdamart"] = fit_lambdamart(train, val, zst, out_dir,
                                           args.num_leaves, args.seed)
    summary["linear_control"] = fit_linear_control(train, val, out_dir,
                                                   args.top_neg, args.seed)
    (out_dir / "fit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
