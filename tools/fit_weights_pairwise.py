"""Method 1 - pairwise logistic regression (linear RankNet) fit of the rerank weights.

The reranker (src/rerank.py) scores each candidate as a linear sum of eight
features.  tools/fit_weights.py fits those weights by coordinate ascent directly
on the non-smooth technical score.  This module fits the *same* linear model with
the classical learning-to-rank objective instead: for every session, rank the
gold product above each non-gold candidate in its own retrieved pool.  A pairwise
logistic regression on the feature *differences* (gold - impostor) is exactly a
linear RankNet, and its coefficient vector, rescaled so the span-coverage
coefficient is 1.0, is a candidate weight vector.

The data gives ONE gold product per session and NO graded per-candidate labels,
so the negatives are *synthesised*: gold vs each of the top non-gold asins in the
pool that was actually shown on a slate turn.  See docs/team/pairwise.md for the
risk (a synthesised negative that is a near-perfect substitute) and the
mitigation (only shown-slate turns, where the pool is already constraint-filtered
by retrieval; the plateau check; the hard-set gate).

    OMP_NUM_THREADS=1 python3 tools/fit_weights_pairwise.py --verify
    OMP_NUM_THREADS=1 python3 tools/fit_weights_pairwise.py --variant both
"""

from __future__ import annotations

import argparse
import json
import sys
import types
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.rerank as rr
import starter.agent as ag
from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    coarse_category,
    customer_reply,
    evaluate,
    initial_message,
    materialize_hidden_fields,
    normalize_recommendations,
)
from src.facets import extract, extract_query_facets
from src.index import load_index
from src.rerank import (
    RerankConfig,
    _category_match,
    _facet_agreement,
    _facet_conflicts,
    _popularity,
    _tail_match,
)
from starter.agent import Agent
from tools import fit_common
from tools.fit_common import FITTED, Scorer, make_config, scalar_from_sessions

# Feature order: 0..7 = [span_cov, pair_cov, retr, pop, facet, cat, tail, conflict]
LENGTH_BONUS = 0.12
DEPTH = 300
CS = (0.1, 1.0, 10.0)

_IDX_PRODS: dict = {}


def index_products(catalog: str) -> dict:
    """The reranker sees index.products (has the token-joined 'text' blob),
    NOT the evaluator's raw catalog_index() dicts. Feature helpers need 'text'."""
    if catalog not in _IDX_PRODS:
        _IDX_PRODS[catalog] = load_index(catalog).products
    return _IDX_PRODS[catalog]


class Log(list):
    """A list that echoes every appended line to stdout as it lands."""
    def append(self, item):
        print(item, flush=True)
        super().append(item)


@dataclass
class Snapshot:
    opening: str
    full: str
    focused: str
    spans: list
    pairs: list
    pool: list  # [(asin, retrieval_score), ...] pre-rerank
    target: str


# --------------------------------------------------------------------------- #
# (a) instrumented snapshot pass - a faithful copy of evaluate()'s outer loop
# --------------------------------------------------------------------------- #

_cap: dict = {}
_orig_rerank = rr.rerank


def _spy(index, state, candidates, config=None, *a, **k):
    _cap["pool"] = [(asin, sc) for asin, sc in candidates]
    _cap["opening"] = state.opening
    _cap["full"] = state.full_text()
    _cap["focused"] = state.focused_text()
    _cap["spans"] = list(state.query_spans())
    _cap["pairs"] = list(state.query_pair_spans())
    return _orig_rerank(index, state, candidates, config)


def snapshot_pass(weights: dict, variant: str, data: dict):
    """Re-implements evaluator.local_evaluator.evaluate()'s per-sample loop.

    Returns (snapshots, session_records).  session_records carry the same fields
    evaluate() builds so scalar_from_sessions() can be checked against it.
    """
    catalog_ids = data["ids"]
    categories = data["cats"]
    products = data["prods"]
    samples = data["dev"]
    agent = Agent(data["catalog"], make_config(weights, variant))

    snapshots: list[Snapshot] = []
    sessions: list[dict] = []

    rr.rerank = _spy
    ag.rerank = _spy
    try:
        for sample in samples:
            session_id = f"public_{uuid.uuid4().hex}"
            agent.reset(session_id, sample["user_profile"])
            target = str(sample["ground_truth"]["parent_asin"])
            eff_card, eff_behavior = materialize_hidden_fields(sample, products)
            eff_sample = {**sample, "intent_card": eff_card, "behavior": eff_behavior}
            disclosed: set = set()
            boundary_used = False
            override_applied = sample["scenario_type"] != "intent_override"
            user_message = initial_message(
                eff_sample, coarse_category(categories.get(target, [])), disclosed
            )
            hit_turn = None
            best_rank = None
            for turn in range(1, MAX_TURNS + 1):
                _cap.clear()
                try:
                    response = agent.respond(session_id, user_message, turn, TOP_K)
                except Exception:
                    response = {"message": "", "ask_attribute": None, "recommendations": []}
                if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                    response = {"message": "", "ask_attribute": None, "recommendations": []}

                if response.get("recommendations") and "pool" in _cap:
                    snapshots.append(Snapshot(
                        opening=_cap["opening"], full=_cap["full"], focused=_cap["focused"],
                        spans=_cap["spans"][:], pairs=_cap["pairs"][:], pool=_cap["pool"][:],
                        target=target,
                    ))

                ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
                if override_applied and target in ranked:
                    best_rank = ranked.index(target) + 1
                    hit_turn = turn
                    break
                if turn == MAX_TURNS:
                    break
                override = eff_sample.get("behavior", {}).get("override") or {}
                if not override_applied and turn + 1 == int(override.get("turn", 3)):
                    override_applied = True
                    new_value = str(override.get("new_value", ""))
                    if new_value:
                        disclosed.add(new_value)
                    user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
                else:
                    user_message, boundary_used = customer_reply(
                        eff_sample, response.get("ask_attribute"), disclosed, boundary_used
                    )
            sessions.append({
                "sample_id": sample["sample_id"],
                "scenario_type": sample["scenario_type"],
                "hit": hit_turn is not None,
                "first_hit_turn": hit_turn,
                "best_rank": best_rank,
                "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
            })
    finally:
        rr.rerank = _orig_rerank
        ag.rerank = _orig_rerank
    return snapshots, sessions


# --------------------------------------------------------------------------- #
# (b) offline features
# --------------------------------------------------------------------------- #

def phi(product: dict, retrieval_norm: float, sn, cf: dict, af: dict,
        spans: list, pairs: list) -> np.ndarray:
    padded = f" {product['text']} "
    coverage = 0.0
    for span in spans:
        if f" {span} " in padded:
            coverage += 1.0 + LENGTH_BONUS * len(span.split())
    pair_coverage = 0.0
    for span in pairs:
        if f" {span} " in padded:
            pair_coverage += 1.0
    pf = extract(product)
    return np.array([
        coverage,
        pair_coverage,
        retrieval_norm,
        _popularity(product),
        _facet_agreement(cf, pf),
        _category_match(sn, product),
        _tail_match(sn, product),
        _facet_conflicts(af, pf, product["text"]),
    ], dtype=float)


def build_pairs(snapshots: list[Snapshot], prods: dict, top_neg: int):
    """Return (diffs, n_skipped) where diffs is a list of phi(target)-phi(neg)."""
    diffs = []
    skipped = 0
    for snap in snapshots:
        head = snap.pool[:DEPTH]
        if not head:
            skipped += 1
            continue
        head_asins = [a for a, _ in head]
        if snap.target not in head_asins:
            skipped += 1
            continue
        top_norm = max((sc for _, sc in head), default=0.0) or 1.0
        sn = types.SimpleNamespace(opening=snap.opening)
        cf = extract_query_facets(snap.full)
        af = extract_query_facets(snap.focused)
        score_by_asin = dict(head)
        tgt_prod = prods.get(snap.target)
        if tgt_prod is None:
            skipped += 1
            continue
        phi_t = phi(tgt_prod, score_by_asin[snap.target] / top_norm, sn, cf, af,
                    snap.spans, snap.pairs)
        negs = [a for a, _ in head if a != snap.target][:top_neg]
        for na in negs:
            np_prod = prods.get(na)
            if np_prod is None:
                continue
            phi_n = phi(np_prod, score_by_asin[na] / top_norm, sn, cf, af,
                        snap.spans, snap.pairs)
            diffs.append(phi_t - phi_n)
    return diffs, skipped


# --------------------------------------------------------------------------- #
# (c) pairwise fit
# --------------------------------------------------------------------------- #

def coef_to_weights(c: np.ndarray) -> dict:
    scale = c[0]
    w = c / scale
    return {
        "popularity_weight": max(0.0, w[3]),
        "retrieval_weight": max(0.0, w[2]),
        "facet_weight": max(0.0, w[4]),
        "category_weight": max(0.0, w[5]),
        "tail_weight": max(0.0, w[6]),
        "pair_weight": max(0.0, w[1]),
        "facet_conflict_weight": max(0.0, -w[7]),
    }


def fit_one_C(diffs: list, C: float):
    D = np.vstack(diffs)
    m = D.shape[0]
    X = np.vstack([D, -D])
    y = np.array([1] * m + [0] * m)
    clf = LogisticRegression(fit_intercept=False, C=C, max_iter=5000,
                             class_weight="balanced")
    clf.fit(X, y)
    return clf.coef_[0]


def fit_select_C(diffs: list, scorer: Scorer, log: list):
    """Try each C, convert to weights, pick the C with the best dev Scorer score.

    Run once, on iteration 0's synthesized pairs; the chosen C is then reused for
    the weight-dependent re-fits so the iterate loop costs one dev pass per step
    (dev is the fitting set, so selecting C on it is allowed - house rules)."""
    best = None
    for C in CS:
        c = fit_one_C(diffs, C)
        scale = float(c[0])
        if scale <= 0:
            log.append(f"  C={C}: DEGENERATE span_cov coef {scale:.4f} <= 0 - skipped")
            continue
        w = coef_to_weights(c)
        neg_wanted = _negatives_wanted(c)
        sc = scorer.score(w)
        log.append(f"  C={C}: scale(span)={scale:.4f}  dev={sc:.6f}  "
                   f"weights={_fmt(w)}  raw_neg={neg_wanted}")
        if best is None or sc > best[0]:
            best = (sc, C, w, c, neg_wanted)
    if best is None:
        raise RuntimeError("all C values degenerate - pairwise signal disagrees "
                           "with span coverage as the unit")
    return best  # (dev_score, C, weights, coef, negatives_wanted)


def fit_fixed_C(diffs: list, C: float, scorer: Scorer, log: list):
    c = fit_one_C(diffs, C)
    scale = float(c[0])
    if scale <= 0:
        raise RuntimeError(f"C={C}: degenerate span_cov coef {scale:.4f} <= 0")
    w = coef_to_weights(c)
    neg = _negatives_wanted(c)
    sc = scorer.score(w)
    return sc, C, w, c, neg


def _negatives_wanted(c: np.ndarray) -> dict:
    scale = c[0]
    w = c / scale
    names = ["span_cov", "pair_cov", "retr", "pop", "facet", "cat", "tail", "conflict"]
    out = {}
    # conflict is healthy when w[7] < 0 (subtracted term); everything else when >= 0
    for i, n in enumerate(names):
        if i == 7:
            if w[7] > 0:
                out[n] = round(float(w[7]), 4)
        elif i == 0:
            continue
        else:
            if w[i] < 0:
                out[n] = round(float(w[i]), 4)
    return out


def _fmt(w: dict) -> str:
    return "{" + ", ".join(f"{k.split('_')[0]}={v:.3f}" for k, v in w.items()) + "}"


# --------------------------------------------------------------------------- #
# (d) iterate
# --------------------------------------------------------------------------- #

def norm_rel(a: dict, b: dict) -> float:
    va = np.array([a[k] for k in FITTED])
    vb = np.array([b[k] for k in FITTED])
    denom = np.linalg.norm(vb) or 1.0
    return float(np.linalg.norm(va - vb) / denom)


def run_variant(variant: str, data: dict, top_neg: int, iters: int, log: list) -> dict:
    log.append(f"\n===== variant={variant} =====")
    scorer = Scorer(data, variant, "dev")
    baseline = {k: getattr(RerankConfig(), k) for k in FITTED}
    trajectory = []
    w_prev = baseline
    chosen = None
    fixed_C = None
    for i in range(iters + 1):
        src_w = baseline if i == 0 else w_prev
        snaps, sess = snapshot_pass(src_w, variant, data)
        # fidelity of this transcript's own scalar (informational)
        snap_scalar = scalar_from_sessions(sess)["score"]
        diffs, skipped = build_pairs(snaps, index_products(data["catalog"]), top_neg)
        log.append(f"iter {i}: snapshots={len(snaps)} (skipped {skipped} no-target-in-head)  "
                   f"pairs={len(diffs)}  transcript-from={_fmt(src_w) if i else 'baseline'}  "
                   f"snap_dev_scalar={snap_scalar:.6f}")
        if fixed_C is None:
            dev_score, C, w, coef, neg = fit_select_C(diffs, scorer, log)
            fixed_C = C
        else:
            dev_score, C, w, coef, neg = fit_fixed_C(diffs, fixed_C, scorer, log)
        log.append(f"iter {i}: C={C}  dev={dev_score:.6f}  weights={_fmt(w)}")
        trajectory.append({"iter": i, "C": C, "dev": dev_score, "weights": w,
                           "coef": [float(x) for x in coef], "neg_wanted": neg,
                           "snapshots": len(snaps), "pairs": len(diffs), "skipped": skipped})
        if i > 0:
            rel = norm_rel(w, w_prev)
            log.append(f"iter {i}: ||w_i - w_(i-1)|| / ||w_(i-1)|| = {rel:.4f}")
            w_prev = w
            chosen = w
            if rel < 0.05:
                log.append(f"iter {i}: converged (rel < 0.05)")
                break
        else:
            w_prev = w
            chosen = w
    return {"trajectory": trajectory, "w_final": chosen, "scorer": scorer}


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #

def verify(data: dict) -> bool:
    ok = True
    baseline = {k: getattr(RerankConfig(), k) for k in FITTED}
    for variant in ("plain", "default"):
        _, sess = snapshot_pass(baseline, variant, data)
        ref = evaluate(Agent(data["catalog"], make_config(baseline, variant)),
                       data["dev"], data["ids"], data["cats"], data["prods"])
        ref_sess = ref["sessions"]
        # exact per-session record match (strict loop-fidelity check)
        mism = 0
        by_id = {s["sample_id"]: s for s in ref_sess}
        for s in sess:
            r = by_id.get(s["sample_id"])
            if r is None or (s["hit"], s["first_hit_turn"], s["best_rank"],
                             round(s["reciprocal_rank"], 12)) != (
                    r["hit"], r["first_hit_turn"], r["best_rank"],
                    round(r["reciprocal_rank"], 12)):
                mism += 1
        mine = scalar_from_sessions(sess)["score"]
        theirs = scalar_from_sessions(ref_sess)["score"]
        d_helper = abs(mine - theirs)
        d_rts = abs(mine - ref["recommended_technical_score"])
        status = "PASS" if (mism == 0 and d_helper < 1e-9) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"[verify {variant}] session-record mismatches={mism}  "
              f"scalar(mine)={mine:.9f}  scalar(evaluate.sessions)={theirs:.9f}  "
              f"|delta_helper|={d_helper:.2e}  |delta_vs_recommended_technical_score|={d_rts:.2e}  "
              f"-> {status}")
    return ok


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--variant", default="both", choices=["plain", "default", "both"])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--top-neg", type=int, default=20)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data = fit_common.load_all(args.catalog)

    if args.verify:
        ok = verify(data)
        return 0 if ok else 1

    variants = ["plain", "default"] if args.variant == "both" else [args.variant]
    log: list = Log()
    result: dict = {}
    for v in variants:
        r = run_variant(v, data, args.top_neg, args.iters, log)
        w = r["w_final"]
        snapped = fit_common.snap(w)
        scorer = r["scorer"]
        pl_snap = fit_common.plateau(snapped, scorer)
        raw_base = scorer.score(w)
        log.append(f"\n--- {v}: w_final (raw argmax) = {_fmt(w)}  dev={raw_base:.6f}")
        log.append(f"--- {v}: snapped (rounded)     = {_fmt(snapped)}")
        log.append(f"--- {v}: plateau(snapped) base={pl_snap['_base']:.6f}  (dev score at +/-50% per weight)")
        for k in FITTED:
            log.append(f"      {k:22s} x0.5={pl_snap[k][0.5]:.6f}  x1.5={pl_snap[k][1.5]:.6f}")
        result[v] = {"trajectory": r["trajectory"], "w_final": w, "w_final_dev": raw_base,
                     "snapped": snapped,
                     "plateau_snap": {k: pl_snap[k] for k in FITTED} | {"_base": pl_snap["_base"]}}

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
