"""Method 4 - roll out synthetic sessions and snapshot per-turn ranking queries.

For every synthetic session (tools/lm_generate.py) and each of two agent configs
("default" = AgentConfig(); "plain" = use_router=False + FixedPolicy), drive the
REAL Agent through a faithful copy of the stress-harness / evaluator loop, with a
spy monkey-patched over src.rerank.rerank / starter.agent.rerank that captures the
pre-rerank pool plus state.opening / full_text / focused_text / query_spans /
query_pair_spans.

On every turn that emits a slate (and, for intent_override, only once the override
has landed), write one ranking query:

  group key : (session_id, turn)
  candidates: the depth-300 retrieval head
  label     : 1 for the target, 0 otherwise
  features  : the 8 reranker features computed EXACTLY as src/rerank.py does,
              plus raw rating_number, raw average_rating, text length,
              span-coverage gap-to-pool-max, 1-indexed retrieval rank, pool size
  slices    : scenario_type, batch, stress_spec

Queries whose target is not in the depth-300 head are SKIPPED and COUNTED - that
count per batch is the retrieval recall ceiling no reranker can touch.

    python3 tools/lm_snapshot.py --sessions S.jsonl --variant both --out Q.parquet
    python3 tools/lm_snapshot.py --verify         # loop fidelity vs evaluate()
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.rerank as rr  # noqa: E402
import starter.agent as ag  # noqa: E402
from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    catalog_index,
    evaluate,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from src.facets import extract, extract_query_facets  # noqa: E402
from src.index import load_index  # noqa: E402
from src.policy import FixedPolicy  # noqa: E402
from src.rerank import (  # noqa: E402
    GENERIC_CATEGORY_PARTS,
    RerankConfig,
    _facet_agreement,
    _facet_conflicts,
    _popularity,
)
from src.text import terms  # noqa: E402
from starter.agent import Agent, AgentConfig  # noqa: E402
from tools.stress_harness import build_customer, parse_spec  # noqa: E402

LENGTH_BONUS = 0.12
DEPTH = 300
# Candidates featurized per query. The reranker reorders the full depth-300 head,
# but negatives past rank ~150 carry no span/pair coverage and near-zero facet
# signal - they are trivial negatives that only inflate the dataset. The target
# is always included even if it sits deeper. The target-not-in-pool (recall)
# check is still made against the full DEPTH.
FEATURIZE_TOP = 150

FEATURE_COLS = [
    "f_span_cov", "f_pair_cov", "f_retr_norm", "f_popularity", "f_facet_agree",
    "f_category", "f_tail", "f_facet_conflict",
    "f_rating_number", "f_average_rating", "f_text_len",
    "f_span_gap_to_max", "f_retr_rank", "f_pool_size",
]

_cap: dict = {}
_orig_rerank = rr.rerank


def _spy(index, state, candidates, config=None, *a, **k):
    _cap["pool"] = [(asin, sc) for asin, sc in candidates]
    _cap["opening"] = state.opening
    _cap["full"] = state.full_text()
    _cap["focused"] = state.focused_text()
    _cap["spans"] = list(state.query_spans())
    _cap["pairs"] = list(state.query_pair_spans())
    out = _orig_rerank(index, state, candidates, config)
    _cap["reranked"] = {asin: i + 1 for i, (asin, _s) in enumerate(out)}
    _cap["reranked_score"] = {asin: sc for asin, sc in out}
    return out


def make_config(variant: str) -> AgentConfig:
    if variant == "default":
        return AgentConfig()
    if variant == "plain":
        return AgentConfig(use_router=False, policy=FixedPolicy())
    raise ValueError(variant)


def _coverage(text_padded: str, spans: list[str]) -> float:
    cov = 0.0
    for span in spans:
        if f" {span} " in text_padded:
            cov += 1.0 + LENGTH_BONUS * len(span.split())
    return cov


def _pair_coverage(text_padded: str, pairs: list[str]) -> float:
    pc = 0.0
    for span in pairs:
        if f" {span} " in text_padded:
            pc += 1.0
    return pc


_PROD_CACHE: dict[str, dict] = {}
_PADDED_CACHE: dict[str, str] = {}


def _prod(asin: str, index_products: dict) -> dict | None:
    """Per-product invariants, computed once and reused across every query whose
    pool contains this asin. extract() / _popularity() / the category token sets
    are pure functions of the catalog record."""
    hit = _PROD_CACHE.get(asin)
    if hit is not None:
        return hit
    product = index_products.get(asin)
    if product is None:
        return None
    text = product.get("text") or ""
    cats = product.get("categories", []) or []
    # _category_match dedupes via a set of lowercased category strings before
    # tokenising; _tail_match walks the raw list. Match both exactly.
    cat_sets = [s for s in (set(terms(c)) for c in {str(v).lower() for v in cats}) if s]
    cleaned: list[str] = []
    for value in cats:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in GENERIC_CATEGORY_PARTS:
                cleaned.append(part)
    tail_sets = [s for s in (set(terms(p)) for p in cleaned[-2:]) if s]
    rec = {
        "padded": f" {text} ",
        "text": text,
        "facets": extract(product),
        "pop": _popularity(product),
        "cat_sets": cat_sets,
        "tail_sets": tail_sets,
        "rating_number": float(product.get("rating_number") or 0),
        "average_rating": float(product.get("average_rating") or 0.0),
        "text_len": float(len(text)),
    }
    _PROD_CACHE[asin] = rec
    return rec


def featurize_query(snap: dict, index_products: dict) -> list[dict] | None:
    """Return one row-dict per featurized candidate, or None if the target is not
    in the depth-300 head (recall miss)."""
    head = snap["pool"][:DEPTH]
    if not head or not snap["spans"]:
        return None
    head_asins = [a for a, _ in head]
    if snap["target"] not in head_asins:
        return None
    top_norm = max((sc for _, sc in head), default=0.0) or 1.0
    cf = extract_query_facets(snap["full"])
    af = extract_query_facets(snap["focused"])
    spans, pairs = snap["spans"], snap["pairs"]
    pool_size = len(head)
    opening_terms = set(terms(snap["opening"], drop_boilerplate=True))

    tgt_rank = head_asins.index(snap["target"]) + 1
    subset = list(enumerate(head[:FEATURIZE_TOP], start=1))
    if tgt_rank > FEATURIZE_TOP:
        subset.append((tgt_rank, head[tgt_rank - 1]))

    # span coverage over the FULL depth-300 head so span-gap-to-pool-max is the
    # true pool max, not the subset max. Only a padded-text lookup - no facet
    # extraction for the rank 151-300 tail we never featurize.
    pool_max_cov = 0.0
    for asin, _sc in head:
        pad = _PADDED_CACHE.get(asin)
        if pad is None:
            p = index_products.get(asin)
            if p is None:
                continue
            pad = _PADDED_CACHE[asin] = f" {p.get('text') or ''} "
        c = _coverage(pad, spans)
        if c > pool_max_cov:
            pool_max_cov = c

    rows: list[dict] = []
    for rank, (asin, sc) in subset:
        rec = _prod(asin, index_products)
        if rec is None:
            continue
        padded = rec["padded"]
        cov = _coverage(padded, spans)
        pf = rec["facets"]
        cat_score = sum(1.0 for s in rec["cat_sets"] if opening_terms & s) if opening_terms else 0.0
        tail_score = sum(1.0 for s in rec["tail_sets"] if s <= opening_terms) if opening_terms else 0.0
        rows.append({
            "session_id": snap["session_id"],
            "turn": snap["turn"],
            "asin": asin,
            "is_target": int(asin == snap["target"]),
            "scenario_type": snap["scenario_type"],
            "batch": snap["batch"],
            "stress_spec": snap["stress_spec"],
            "variant": snap["variant"],
            "f_span_cov": cov,
            "f_pair_cov": _pair_coverage(padded, pairs),
            "f_retr_norm": sc / top_norm,
            "f_popularity": rec["pop"],
            "f_facet_agree": _facet_agreement(cf, pf),
            "f_category": cat_score,
            "f_tail": tail_score,
            "f_facet_conflict": _facet_conflicts(af, pf, rec["text"]),
            "f_rating_number": rec["rating_number"],
            "f_average_rating": rec["average_rating"],
            "f_text_len": rec["text_len"],
            "f_span_gap_to_max": pool_max_cov - cov,
            "f_retr_rank": float(rank),
            "f_pool_size": float(pool_size),
            "shipped_rerank_rank": int(snap.get("reranked", {}).get(asin, 0)),
        })
    return rows


def run_session(agent, sample, customer, catalog_ids, target, variant,
                capture: bool):
    """Faithful copy of stress_harness.run_session's loop. When capture=True,
    returns (session_record, [snapshot dicts]); else ([], session_record)."""
    sid = "lm_" + str(sample["sample_id"]) + "_" + variant
    agent.reset(sid, sample["user_profile"])
    msg = customer.opening()
    hit_turn = best_rank = None
    slate_turns: dict[int, dict] = {}
    for turn in range(1, MAX_TURNS + 1):
        _cap.clear()
        try:
            resp = agent.respond(sid, msg, turn, TOP_K)
        except Exception:
            resp = {"message": "", "ask_attribute": None, "recommendations": []}
        if not isinstance(resp, dict) or not isinstance(resp.get("message"), str):
            resp = {"message": "", "ask_attribute": None, "recommendations": []}
        # Only capture turns where the reranker actually ran: src/rerank.py
        # returns the pool untouched when there are no query_spans (e.g. the
        # override turn before any post-override constraint is disclosed), so
        # such a turn carries only retrieval order - nothing a reranker learns.
        if (capture and resp.get("recommendations") and _cap.get("pool")
                and _cap.get("spans") and customer.override_applied):
            slate_turns[turn] = {
                "session_id": sid, "turn": turn, "target": target,
                "scenario_type": sample["scenario_type"], "batch": sample["batch"],
                "stress_spec": sample["stress_spec"], "variant": variant,
                "opening": _cap["opening"], "full": _cap["full"],
                "focused": _cap["focused"], "spans": _cap["spans"][:],
                "pairs": _cap["pairs"][:], "pool": _cap["pool"][:],
                "reranked": dict(_cap.get("reranked", {})),
                "reranked_score": dict(_cap.get("reranked_score", {})),
            }
        ranked = normalize_recommendations(resp.get("recommendations"), catalog_ids)
        if customer.override_applied and target in ranked:
            best_rank = ranked.index(target) + 1
            hit_turn = turn
            break
        if turn == MAX_TURNS:
            break
        msg = customer.reply(turn, resp.get("ask_attribute"))

    record = {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "hit": hit_turn is not None,
        "first_hit_turn": hit_turn,
        "best_rank": best_rank,
        "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
    }
    # keep first + last slate turn only (under-informed and fully-informed regimes)
    snaps: list[dict] = []
    if slate_turns:
        keys = sorted(slate_turns)
        for k in dict.fromkeys([keys[0], keys[-1]]):
            snaps.append(slate_turns[k])
    return record, snaps


def _drive(sessions, variant, catalog, ids, cats, prods, index_products,
           writer_cb, progress_every):
    agent = Agent(catalog, make_config(variant))
    rr.rerank = _spy
    ag.rerank = _spy
    miss = Counter()          # (batch, stress_spec, scenario) -> target-not-in-pool
    seen = Counter()          # (batch, stress_spec, scenario) -> total queries
    n_rows = n_q = 0
    try:
        for i, s in enumerate(sessions, 1):
            target = str(s["ground_truth"]["parent_asin"])
            card, behavior = materialize_hidden_fields(s, prods)
            spec = parse_spec(s["stress_spec"] or "official")
            cust = build_customer(spec, s, card, behavior, cats, target,
                                  index_products)
            _rec, snaps = run_session(agent, s, cust, ids, target, variant,
                                      capture=True)
            for snap in snaps:
                key = (snap["batch"], snap["stress_spec"] or "official",
                       snap["scenario_type"])
                seen[key] += 1
                rows = featurize_query(snap, index_products)
                if rows is None:
                    miss[key] += 1
                    continue
                n_q += 1
                n_rows += len(rows)
                writer_cb(rows)
            if progress_every and i % progress_every == 0:
                print(f"[{variant}] {i}/{len(sessions)} sessions  "
                      f"{n_q} queries  {n_rows} rows  "
                      f"recall_miss {sum(miss.values())}/{sum(seen.values())}",
                      flush=True)
    finally:
        rr.rerank = _orig_rerank
        ag.rerank = _orig_rerank
    return {"rows": n_rows, "queries": n_q,
            "miss": {f"{b}|{sp}|{sc}": c for (b, sp, sc), c in miss.items()},
            "seen": {f"{b}|{sp}|{sc}": c for (b, sp, sc), c in seen.items()}}


def cmd_run(args) -> int:
    ids, cats, prods = catalog_index(args.catalog)
    index_products = load_index(args.catalog).products
    sessions = load_jsonl(args.sessions)
    variants = ["default", "plain"] if args.variant == "both" else [args.variant]

    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [("session_id", pa.string()), ("turn", pa.int32()), ("asin", pa.string()),
         ("is_target", pa.int8()), ("scenario_type", pa.string()),
         ("batch", pa.string()), ("stress_spec", pa.string()),
         ("variant", pa.string())]
        + [(c, pa.float64()) for c in FEATURE_COLS]
        + [("shipped_rerank_rank", pa.int32())]
    )
    writer = pq.ParquetWriter(out, schema, compression="zstd")
    buf: list[dict] = []

    def flush():
        nonlocal buf
        if not buf:
            return
        cols = {name: [r[name] for r in buf] for name in schema.names}
        writer.write_table(pa.table(cols, schema=schema))
        buf = []

    def writer_cb(rows):
        buf.extend(rows)
        if len(buf) >= 120_000:
            flush()

    meta = {"variants": {}}
    try:
        for v in variants:
            print(f"=== variant {v} ===", flush=True)
            stats = _drive(sessions, v, args.catalog, ids, cats, prods,
                           index_products, writer_cb, args.progress_every)
            meta["variants"][v] = stats
            flush()
    finally:
        flush()
        writer.close()

    # aggregate recall-miss per batch/spec
    agg = defaultdict(lambda: [0, 0])
    for v in meta["variants"].values():
        for k, c in v["seen"].items():
            agg[k][1] += c
        for k, c in v["miss"].items():
            agg[k][0] += c
    meta["recall_miss_by_batch_spec_scenario"] = {
        k: {"miss": m, "seen": s, "rate": (m / s if s else 0.0)}
        for k, (m, s) in sorted(agg.items())
    }
    total_miss = sum(m for m, _ in agg.values())
    total_seen = sum(s for _, s in agg.values())
    meta["recall_miss_overall"] = {
        "miss": total_miss, "seen": total_seen,
        "rate": (total_miss / total_seen if total_seen else 0.0)}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


def cmd_verify(args) -> int:
    """Drive the base (official) customer over the dev split; the reconstructed
    session records must match evaluate()'s exactly."""
    from tools.sweep import split_samples

    ids, cats, prods = catalog_index(args.catalog)
    index_products = load_index(args.catalog).products
    dev = split_samples(load_jsonl("data/public_set.jsonl"), "dev")
    # give the synthetic-shape fields the driver expects
    dev = [{**s, "batch": "coop", "stress_spec": ""} for s in dev]

    ok = True
    for variant in ("plain", "default"):
        agent = Agent(args.catalog, make_config(variant))
        rr.rerank = _spy
        ag.rerank = _spy
        recs = []
        try:
            for s in dev:
                target = str(s["ground_truth"]["parent_asin"])
                card, behavior = materialize_hidden_fields(s, prods)
                cust = build_customer(parse_spec("official"), s, card, behavior,
                                      cats, target, index_products)
                rec, _ = run_session(agent, s, cust, ids, target, variant,
                                     capture=False)
                recs.append(rec)
        finally:
            rr.rerank = _orig_rerank
            ag.rerank = _orig_rerank

        ref = evaluate(Agent(args.catalog, make_config(variant)), dev, ids, cats, prods)
        by_id = {r["sample_id"]: r for r in ref["sessions"]}
        mism = 0
        for r in recs:
            g = by_id.get(r["sample_id"])
            if g is None or (r["hit"], r["first_hit_turn"], r["best_rank"],
                             round(r["reciprocal_rank"], 12)) != (
                    g["hit"], g["first_hit_turn"], g["best_rank"],
                    round(g["reciprocal_rank"], 12)):
                mism += 1
        mine = _scalar(recs)
        theirs = _scalar(ref["sessions"])
        d = abs(mine - theirs)
        status = "PASS" if (mism == 0 and d < 1e-9) else "FAIL"
        ok = ok and status == "PASS"
        print(f"[verify {variant}] mismatches={mism}  scalar(mine)={mine:.9f}  "
              f"scalar(evaluate)={theirs:.9f}  |delta|={d:.2e}  -> {status}")

    ok = ok and _feature_fidelity(args.catalog, ids, cats, prods, index_products)
    return 0 if ok else 1


_W = RerankConfig()  # shipped defaults


def _feature_fidelity(catalog, ids, cats, prods, index_products) -> bool:
    """The 8 reranker features this script recomputes offline must reproduce
    src/rerank.py's own ranking. For a sample of captured pools, score the
    featurized subset with the shipped weights and assert the induced order
    matches the order those same asins hold in rerank()'s real output."""
    from tools.sweep import split_samples
    dev = [{**s, "batch": "coop", "stress_spec": ""}
           for s in split_samples(load_jsonl("data/public_set.jsonl"), "dev")[:40]]
    agent = Agent(catalog, make_config("default"))
    rr.rerank = _spy
    ag.rerank = _spy
    checked = bad_total = bad_order = 0
    worst = 0.0
    try:
        for s in dev:
            target = str(s["ground_truth"]["parent_asin"])
            card, behavior = materialize_hidden_fields(s, prods)
            cust = build_customer(parse_spec("official"), s, card, behavior, cats,
                                  target, index_products)
            _rec, snaps = run_session(agent, s, cust, ids, target, "default",
                                      capture=True)
            for snap in snaps:
                rows = featurize_query(snap, index_products)
                if not rows:
                    continue
                real_score = snap["reranked_score"]
                scored = []
                for r in rows:
                    tot = (_W.span_weight * r["f_span_cov"]
                           + _W.pair_weight * r["f_pair_cov"]
                           + _W.retrieval_weight * r["f_retr_norm"]
                           + _W.popularity_weight * r["f_popularity"]
                           + _W.facet_weight * r["f_facet_agree"]
                           + _W.category_weight * r["f_category"]
                           + _W.tail_weight * r["f_tail"]
                           - _W.facet_conflict_weight * r["f_facet_conflict"])
                    scored.append((r["asin"], tot))
                    rs = real_score.get(r["asin"])
                    if rs is not None:
                        worst = max(worst, abs(tot - rs))
                        if abs(tot - rs) > 1e-9:
                            bad_total += 1
                mine_order = [a for a, _ in sorted(scored, key=lambda x: (-x[1], x[0]))]
                real = snap["reranked"]
                real_order = sorted((a for a, _ in scored),
                                    key=lambda a: real.get(a, 10 ** 9))
                checked += 1
                if mine_order != real_order:
                    bad_order += 1
    finally:
        rr.rerank = _orig_rerank
        ag.rerank = _orig_rerank
    ok = bad_total == 0 and bad_order == 0
    print(f"[verify features] pools={checked}  per-cand total mismatches={bad_total}"
          f"  order mismatches={bad_order}  worst|dtotal|={worst:.2e}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def _scalar(sessions: list[dict]) -> float:
    n = len(sessions)
    if not n:
        return 0.0
    hit = sum(int(s["hit"]) for s in sessions) / n
    mrr = sum(s["reciprocal_rank"] for s in sessions) / n
    mttc = sum((s["first_hit_turn"] if s["first_hit_turn"] is not None
                else MAX_TURNS + 1) for s in sessions) / n
    eff = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return 0.5 * hit + 0.3 * mrr + 0.2 * eff


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--sessions", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--variant", default="both", choices=["default", "plain", "both"])
    ap.add_argument("--progress-every", type=int, default=1000)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        return cmd_verify(args)
    if not args.sessions or not args.out:
        ap.error("--sessions and --out are required unless --verify")
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
