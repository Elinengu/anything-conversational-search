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
from src.llm import LLMConfig  # noqa: E402
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
        # The pre-sniper shipped defaults, kept so before/after can be measured
        # in one process on any dataset (Change 17).
        "pre_sniper": AgentConfig(first_recommend_turn=3, list_size_ramp=(4, 10)),
        # Sniper sizing (Change 17): one candidate per turn until a wide
        # safety-net turn. The evaluator ends a session the moment the target
        # appears and scores its position *within that turn's list only*, so a
        # 1-item slate converts every eventual hit into rank 1. Rank r -> 1 is
        # worth 0.30*(1 - 1/r); a turn of MTTC costs 0.20/10 = 0.02, so rank is
        # worth ~13x a turn. The elimination scan makes the singles cumulative:
        # eight singles plus a wide turn walk 18+ distinct candidates, deeper
        # than one 10-item slate. sniperN widens at turn N.
        "sniper5": AgentConfig(first_recommend_turn=1, list_size_ramp=(1,) * 4 + (10,)),
        "sniper6": AgentConfig(first_recommend_turn=1, list_size_ramp=(1,) * 5 + (10,)),
        "sniper7": AgentConfig(first_recommend_turn=1, list_size_ramp=(1,) * 6 + (10,)),
        "sniper8": AgentConfig(first_recommend_turn=1, list_size_ramp=(1,) * 7 + (10,)),
        "sniper9": AgentConfig(first_recommend_turn=1, list_size_ramp=(1,) * 8 + (10,)),
        "sniper10": AgentConfig(first_recommend_turn=1, list_size_ramp=(1,) * 9 + (10,)),
        # Isolates the two halves of sniper9: singles but still holding turns
        # 1-2 back (sniper9_t3), and guessing from turn 1 at the shipped widths
        # (elim1 is that row without the ramp change).
        "sniper9_t3": AgentConfig(first_recommend_turn=3, list_size_ramp=(1,) * 6 + (10,)),
        # The STAGNATING orchestration phase (context_programming Phase 3)
        # overrides the ramp with its own wide slate, so sniper9 above still
        # emits 10 on a stalled turn - which is where its remaining rank losses
        # sit. These rows carry the singles through stagnation too.
        "sniper9_stag1": AgentConfig(
            first_recommend_turn=1, list_size_ramp=(1,) * 8 + (10,), stagnation_slate_size=1),
        "sniper7_stag1": AgentConfig(
            first_recommend_turn=1, list_size_ramp=(1,) * 6 + (10,), stagnation_slate_size=1),
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
        # Re-swept under sniper sizing: with a one-item slate the popularity
        # prior stops being a tie-break and becomes the decision, so the 0.4
        # fitted against a 4-wide slate is not automatically still right.
        "pop070": AgentConfig(rerank=RerankConfig(popularity_weight=0.70)),
        "pop100": AgentConfig(rerank=RerankConfig(popularity_weight=1.00)),
        "pop140": AgentConfig(rerank=RerankConfig(popularity_weight=1.40)),
        "pop180": AgentConfig(rerank=RerankConfig(popularity_weight=1.80)),
        "pop250": AgentConfig(rerank=RerankConfig(popularity_weight=2.50)),
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
        # router_off is the flat single-track pipeline - it must score
        # bit-for-bit like the pre-routing agent. router_on is the shipped
        # default: the state-machine's _route_for drives policy, rerank and
        # timing per intent_track (see src/state.py, src/context_programming.py).
        # The dual_tracking branch's per-track config knobs (buying_rerank,
        # route_policies, ...) and its own router_on_hardfilter row are
        # superseded by that state-tracked routing and were dropped here rather
        # than kept unwired.
        "router_off": AgentConfig(use_router=False),
        "router_on": AgentConfig(use_router=True),
        # Dense sentence-embedding cosine as an S6 rerank signal (branch
        # dense_rerank). The only S6 term that scores meaning rather than exact
        # tokens - the paraphrase hypothesis. 0.0 is router_on; these bracket the
        # weight. `_spans` encodes the disclosed spans instead of full_text.
        "dense_rr_02": AgentConfig(use_router=True, rerank=RerankConfig(dense_weight=0.2)),
        "dense_rr_05": AgentConfig(use_router=True, rerank=RerankConfig(dense_weight=0.5)),
        "dense_rr_10": AgentConfig(use_router=True, rerank=RerankConfig(dense_weight=1.0)),
        "dense_rr_15": AgentConfig(use_router=True, rerank=RerankConfig(dense_weight=1.5)),
        "dense_rr_05_spans": AgentConfig(
            use_router=True, rerank=RerankConfig(dense_weight=0.5, dense_query="spans")),
        "dense_rr_05_rns": AgentConfig(   # also rescore when no verbatim span exists
            use_router=True,
            rerank=RerankConfig(dense_weight=0.5, rescore_without_spans=True)),
        # Step 3.2/3.3 (branch state-encoder-eval): the 21-session generic-tail
        # sanity check on dense_rr_10 came back net -0.016, scenario-split -
        # buying/override up, browsing down hard - see
        # docs/team/branch_state_encoder_eval_changes.md. These gate/query-text
        # variants test whether that split is fixable rather than intrinsic.
        # dense_rr_gate: fire only when state.over_general (pool has stopped
        # discriminating) - the pool-shape hypothesis alone.
        "dense_rr_gate": AgentConfig(
            use_router=True,
            rerank=RerankConfig(dense_weight=1.0, dense_gate_over_general=True)),
        # dense_rr_nobrowse: withhold on the browsing track only, no pool-shape
        # gate - isolates whether avoiding the track that collapsed is
        # sufficient on its own.
        "dense_rr_nobrowse": AgentConfig(
            use_router=True,
            rerank=RerankConfig(dense_weight=1.0, dense_gate_exclude_browsing=True)),
        # dense_rr_gate_nobrowse: both gates together - fire only on a stalled
        # pool, and never on browsing even then.
        "dense_rr_gate_nobrowse": AgentConfig(
            use_router=True,
            rerank=RerankConfig(dense_weight=1.0, dense_gate_over_general=True,
                                dense_gate_exclude_browsing=True)),
        # dense_rr_slots: same unconditional dense_weight as dense_rr_10, but
        # query state.authoritative_text() (the state machine's compact
        # active-slot text) instead of full_text() - no simulator boilerplate.
        "dense_rr_slots": AgentConfig(
            use_router=True,
            rerank=RerankConfig(dense_weight=1.0, dense_query="slots")),
        # Dense sentence-embedding cosine as an S5 retrieval route (branch
        # dense_rerank). Originally trialled per-track (browsing only, since the
        # paraphrase recall tail concentrates there - docs/team/dense_route.md);
        # that per-track config surface (buying_retrieval/browsing_retrieval) is
        # dropped on this branch in favour of the state machine's own routing
        # (see router_on's comment above), so this row is the both-tracks form.
        "dense_route_all": AgentConfig(
            use_router=True, retrieval=RetrievalConfig(use_dense=True)),
        # dense_route_all is a confirmed trade-off (branch state-encoder-eval):
        # +0.0263 under paraphrase:heavy+browse-gated stress, -0.0042/-0.0065 on
        # the cooperative official/holdout sets - see
        # docs/team/branch_state_encoder_eval_changes.md §3d. These gate it the
        # same way dense_rr_gate/dense_rr_nobrowse gate the S6 rerank term.
        "dense_route_gate": AgentConfig(
            use_router=True,
            retrieval=RetrievalConfig(use_dense=True, dense_gate_over_general=True)),
        "dense_route_nobrowse": AgentConfig(
            use_router=True,
            retrieval=RetrievalConfig(use_dense=True, dense_gate_exclude_browsing=True)),
        "dense_route_gate_nobrowse": AgentConfig(
            use_router=True,
            retrieval=RetrievalConfig(use_dense=True, dense_gate_over_general=True,
                                      dense_gate_exclude_browsing=True)),
        # Tier-2 opt-in LLM reranking layer (src/llm.py, DeepSeek), fused into
        # the top llm_depth of the lexical order - see
        # docs/team/ideas_to_integrate_llm.md #3 and RerankConfig.llm_weight's
        # docstring. Needs DEEPSEEK_API_KEY in the environment; with no key,
        # LLMReranker.available is False and both rows fall back byte-identical
        # to router_on. llm_rerank_always fires every turn (llm_gate_margin=0.0);
        # llm_rerank_gated fires only when the previous turn's pool had no clear
        # lexical leader (state.leader_margin < 0.05, the shipped default) -
        # see docs/team/agent_changes.md Change 16 for the measured trade-off.
        "llm_rerank_always": AgentConfig(
            use_router=True,
            llm=LLMConfig(enabled=True),
            rerank=RerankConfig(llm_weight=1.0, llm_gate_margin=0.0)),
        "llm_rerank_gated": AgentConfig(
            use_router=True,
            llm=LLMConfig(enabled=True),
            rerank=RerankConfig(llm_weight=1.0, llm_gate_margin=0.05)),
        # ---- Arm A: let the LLM rank on turn 1 --------------------------
        # Turn 1 is the only turn where S6 is otherwise inert (rerank returns
        # early with no spans) and the turn sniper sizing stakes a full slate
        # on, so it is the one place the LLM supplies the FIRST evidence-aware
        # pass rather than second-guessing a lexical one.
        "llm_t1": AgentConfig(
            use_router=True, llm=LLMConfig(enabled=True),
            rerank=RerankConfig(llm_weight=1.0, llm_gate_margin=0.05,
                                llm_without_spans=True)),
        # Turn 1's slots may be a single word ("alloy"); the opening also
        # carries the category ("Jewelry Necklaces").
        "llm_t1_opening": AgentConfig(
            use_router=True, llm=LLMConfig(enabled=True),
            rerank=RerankConfig(llm_weight=1.0, llm_gate_margin=0.05,
                                llm_without_spans=True, llm_query="opening")),
        # 6.5% of hard constraints fall beyond the 220-char candidate window.
        "llm_t1_wide": AgentConfig(
            use_router=True, llm=LLMConfig(enabled=True, candidate_chars=700),
            rerank=RerankConfig(llm_weight=1.0, llm_gate_margin=0.05,
                                llm_without_spans=True)),
        # Separates "turn 1 helps" from "the gate was the constraint".
        "llm_t1_nogate": AgentConfig(
            use_router=True, llm=LLMConfig(enabled=True),
            rerank=RerankConfig(llm_weight=1.0, llm_gate_margin=0.0,
                                llm_without_spans=True)),
        # ---- Arm B: the LLM as a parser feeding retrieval ----------------
        # Judged on the paraphrase stress customer, not the public set: a
        # cooperative customer quotes the catalog verbatim, so there is nothing
        # there for a parser to add.
        "llm_terms": AgentConfig(
            use_router=True, llm=LLMConfig(enabled=True),
            retrieval=RetrievalConfig(use_llm_terms=True)),
        "llm_terms_w04": AgentConfig(
            use_router=True, llm=LLMConfig(enabled=True),
            retrieval=RetrievalConfig(use_llm_terms=True, weight_llm_terms=0.4)),
        "llm_terms_w12": AgentConfig(
            use_router=True, llm=LLMConfig(enabled=True),
            retrieval=RetrievalConfig(use_llm_terms=True, weight_llm_terms=1.2)),
        # Both arms - they act at different stages and may compose the way
        # changes 18 and 19 did.
        "llm_t1_terms": AgentConfig(
            use_router=True, llm=LLMConfig(enabled=True),
            retrieval=RetrievalConfig(use_llm_terms=True),
            rerank=RerankConfig(llm_weight=1.0, llm_gate_margin=0.05,
                                llm_without_spans=True)),
        # Both optional layers at once, on the merged (change 18 + 19) default.
        # They are independent signals over the same head: the dense term scores
        # meaning per candidate, the LLM term reorders the top llm_depth
        # listwise. Measured together because each was only ever measured alone,
        # and both fire on ambiguous pools - which change 19 made rarer.
        "dense_llm": AgentConfig(
            use_router=True,
            llm=LLMConfig(enabled=True),
            rerank=RerankConfig(dense_weight=1.0, llm_weight=1.0, llm_gate_margin=0.05)),
        # The pre-pool lexical-only retrieval, kept as the ablation baseline.
        "catpool_off": AgentConfig(
            retrieval=RetrievalConfig(use_category_pool=False),
            rerank=RerankConfig(depth=300)),
        # Both changes off - the pre-change-18 agent, the floor both are measured
        # from. `pre_sniper` is the sizing ablation on its own; `catpool_off` is
        # the retrieval ablation on its own.
        "no_sniper_no_catpool": AgentConfig(
            first_recommend_turn=3, list_size_ramp=(4, 10),
            retrieval=RetrievalConfig(use_category_pool=False),
            rerank=RerankConfig(depth=300)),
        # ---- coarse-category pool route (S5) ----------------------------
        # Measured motivation: at turn 1 the target is inside our 300-candidate
        # lexical pool only 80.5% of the time, at median rank 51, and only ~66%
        # of those 300 are in the target's category at all. The evaluator's
        # opening message names coarse_category(target's categories), so that
        # string keys a pool the target is inside 200/200 (median 182 members).
        # `catpool` unions that pool into the candidate set; depth=0 makes the
        # reranker score all of it.
        "catpool": AgentConfig(
            retrieval=RetrievalConfig(use_category_pool=True),
            rerank=RerankConfig(depth=0)),
        # Does the pool make the reranker's own category signals redundant? The
        # union is not category-pure - it still carries the 300 lexical
        # candidates, a third of which are out-of-category - so these should
        # still be doing work. Measured, not assumed.
        "catpool_nocat": AgentConfig(rerank=RerankConfig(category_weight=0.0)),
        "catpool_notail": AgentConfig(rerank=RerankConfig(tail_weight=0.0)),
        "catpool_nocat_notail": AgentConfig(rerank=RerankConfig(category_weight=0.0, tail_weight=0.0)),
        # Popularity re-swept *inside* the pool. Raising it over a lexical pool
        # was rejected on branch claude/techjam-agent-analysis-hzm14g (gains on
        # both public splits, loses on both generated sets). Inside a
        # category-correct pool it is a different measurement.
        "catpool_pop070": AgentConfig(
            retrieval=RetrievalConfig(use_category_pool=True),
            rerank=RerankConfig(depth=0, popularity_weight=0.70)),
        "catpool_pop100": AgentConfig(
            retrieval=RetrievalConfig(use_category_pool=True),
            rerank=RerankConfig(depth=0, popularity_weight=1.00)),
        # Sniper list sizing is not on main yet (branch
        # claude/techjam-agent-analysis-hzm14g, public 0.923487 -> 0.940083).
        # These rows pin it explicitly so the pool route can be measured against
        # the sizing it will actually ship alongside, in one process.
        # Pool-route RRF weight bracket.
        "sn_cp_w20": AgentConfig(
            first_recommend_turn=1, list_size_ramp=(1, 1, 1, 1, 10),
            retrieval=RetrievalConfig(use_category_pool=True, weight_category_pool=2.0),
            rerank=RerankConfig(depth=0)),
        "sn_cp_w30": AgentConfig(
            first_recommend_turn=1, list_size_ramp=(1, 1, 1, 1, 10),
            retrieval=RetrievalConfig(use_category_pool=True, weight_category_pool=3.0),
            rerank=RerankConfig(depth=0)),
        "sn_cp_w03": AgentConfig(
            first_recommend_turn=1, list_size_ramp=(1, 1, 1, 1, 10),
            retrieval=RetrievalConfig(use_category_pool=True, weight_category_pool=0.3),
            rerank=RerankConfig(depth=0)),
        "sn_cp_w05": AgentConfig(
            first_recommend_turn=1, list_size_ramp=(1, 1, 1, 1, 10),
            retrieval=RetrievalConfig(use_category_pool=True, weight_category_pool=0.5),
            rerank=RerankConfig(depth=0)),
        "sn_cp_w07": AgentConfig(
            first_recommend_turn=1, list_size_ramp=(1, 1, 1, 1, 10),
            retrieval=RetrievalConfig(use_category_pool=True, weight_category_pool=0.7),
            rerank=RerankConfig(depth=0)),
        "sn_cp_w15": AgentConfig(
            first_recommend_turn=1, list_size_ramp=(1, 1, 1, 1, 10),
            retrieval=RetrievalConfig(use_category_pool=True, weight_category_pool=1.5),
            rerank=RerankConfig(depth=0)),
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
