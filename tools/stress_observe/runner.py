"""Trace stress-harness conversations and render the existing observer UI.

This module intentionally lives outside ``tools/observe.py`` and
``tools/stress_harness.py``. It composes their public building blocks so the two
actively developed tools do not need to be edited to support one another.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import tools.observe as observe_module
from evaluator.local_evaluator import catalog_index, load_jsonl, materialize_hidden_fields
from starter.agent import Agent
from tools.observe import (
    DIAGNOSES,
    TracingAgent,
    diagnose,
    render_index_markdown,
    render_session_markdown,
    render_viewer,
)
from tools.stress_harness import (
    StressCustomer,
    build_customer,
    parse_spec,
    run_session,
    score,
    select_generic,
)


def install_stress_probes():
    """Install observer probes compatible with the track-aware agent.

    ``tools.observe.install_probes`` predates the reranker's ``track=`` keyword.
    This local implementation forwards that keyword and also observes
    ``detect_turn_intent`` so the viewer shows the active per-turn route rather
    than only the opening classification.
    """
    import starter.agent as agent_module

    names = ["classify", "retrieve", "rerank"]
    if hasattr(agent_module, "detect_turn_intent"):
        names.append("detect_turn_intent")
    original = {name: getattr(agent_module, name) for name in names}

    def classify_probe(opening):
        route = original["classify"](opening)
        observe_module._PROBE["route"] = route
        return route

    def detect_probe(*args, **kwargs):
        route = original["detect_turn_intent"](*args, **kwargs)
        observe_module._PROBE["route"] = route
        return route

    # **kwargs rather than a fixed signature: Agent._respond() passes track=,
    # embed= and qvec= on every retrieve()/rerank() call (the dual-track routing
    # and the dense sentence-embedding signal), plus route_hint= on the
    # orchestrator's retrieval reroute. A probe that cannot accept them raises
    # TypeError inside Agent.respond()'s catch-all handler, so the turn returns
    # an empty envelope and every traced session silently scores 0.000 - the
    # same bug docs/team/agent_changes.md change 15 fixed once in
    # tools/observe.py, which this sibling tool's separate copy never picked up.
    def retrieve_probe(index, state, config=None, **kwargs):
        started = time.perf_counter()
        pool = original["retrieve"](index, state, config, **kwargs)
        observe_module._PROBE["retrieve_ms"] = (time.perf_counter() - started) * 1000.0
        observe_module._PROBE["pool"] = pool
        observe_module._PROBE["retrieval_route"] = kwargs.get("route_hint") or "terms"
        return pool

    def rerank_probe(index, state, candidates, config=None, **kwargs):
        started = time.perf_counter()
        ranked = original["rerank"](index, state, candidates, config, **kwargs)
        observe_module._PROBE["rerank_ms"] = (time.perf_counter() - started) * 1000.0
        observe_module._PROBE["ranked"] = ranked
        return ranked

    agent_module.classify = classify_probe
    agent_module.retrieve = retrieve_probe
    agent_module.rerank = rerank_probe
    if "detect_turn_intent" in original:
        agent_module.detect_turn_intent = detect_probe

    def undo() -> None:
        for name, function in original.items():
            setattr(agent_module, name, function)

    return undo


class StressTracingAgent(TracingAgent):
    """The existing tracing wrapper with stress-runner compatibility.

    The stress harness reads a few diagnostic attributes directly from ``Agent``;
    these properties delegate them without exposing the hidden target to the
    wrapped production agent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._next_annotation: dict[str, Any] | None = None
        self._bound_customer = None

    @property
    def index(self):
        return self.inner.index

    @property
    def config(self):
        return self.inner.config

    @property
    def _states(self):
        return self.inner._states

    def bind_customer(self, customer) -> None:
        self._bound_customer = customer

    def set_next_annotation(self, annotation: dict[str, Any]) -> None:
        self._next_annotation = annotation

    def reset(self, session_id: str, user_profile: dict) -> None:
        super().reset(session_id, user_profile)
        # A decoy stressor can alter the effective behavior after the dataset's
        # hidden fields were materialized. Render what this customer actually uses.
        if self._current is not None and self._bound_customer is not None:
            self._current["intent_card"] = self._bound_customer.card
            self._current["behavior"] = self._bound_customer.behavior
            self._current["stressor"] = {
                "paraphrase": self._bound_customer.paraphrase,
                "browse_gated": self._bound_customer.browse_gated,
            }

    def _record_turn(self, *args, **kwargs):
        super()._record_turn(*args, **kwargs)
        if self._next_annotation is not None and self._current is not None:
            turn_input = self._current["turns"][-1]["in"]
            message = turn_input["message"]
            turn_input.clear()
            turn_input.update({"message": message, **self._next_annotation})
            self._next_annotation = None


class ObservedCustomer:
    """Attach exact stress-customer events to the next traced input message."""

    def __init__(self, inner, tracer: StressTracingAgent):
        self.inner = inner
        self.tracer = tracer

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def opening(self) -> str:
        before = set(self.inner.disclosed)
        message = self.inner.opening()
        revealed = sorted(set(self.inner.disclosed) - before)
        self.tracer.set_next_annotation({"kind": "opening", "revealed": revealed})
        return message

    def reply(self, turn: int, ask_attribute: object) -> str:
        before_disclosed = set(self.inner.disclosed)
        before_override = self.inner.override_applied
        before_boundary = self.inner.boundary_used
        message = self.inner.reply(turn, ask_attribute)
        revealed = sorted(set(self.inner.disclosed) - before_disclosed)

        if not before_override and self.inner.override_applied:
            annotation = {"kind": "override", "revealed": revealed}
        elif revealed:
            annotation = {"kind": "disclosed", "revealed": revealed}
        elif not before_boundary and self.inner.boundary_used:
            annotation = {
                "kind": "boundary_decline",
                "attribute": ask_attribute,
                "revealed": [],
            }
        elif "still just browsing" in message.lower():
            annotation = {"kind": "stalled", "revealed": []}
        elif "don't have" in message.lower() or "do not have" in message.lower():
            annotation = {
                "kind": "no_preference",
                "attribute": ask_attribute,
                "revealed": [],
            }
        else:
            annotation = {"kind": "stalled", "revealed": []}

        self.tracer.set_next_annotation(annotation)
        return message


def _outcome(sample: dict, result: dict) -> dict:
    reciprocal_rank = float(result["reciprocal_rank"])
    best_rank = round(1.0 / reciprocal_rank) if reciprocal_rank > 0.0 else None
    return {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "hit": bool(result["hit"]),
        "first_hit_turn": result["first_hit_turn"],
        "best_rank": best_rank,
        "reciprocal_rank": reciprocal_rank,
        "token_coverage": result["token_coverage"],
        "pool_rank": result["pool_rank"],
        "ranked_rank": result["ranked_rank"],
    }


def run_traced(
    tracer: StressTracingAgent,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict,
    products: dict,
    spec: dict,
) -> list[dict]:
    StressCustomer.decoys_injected = StressCustomer.decoys_eligible = 0
    outcomes: list[dict] = []
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        customer = build_customer(
            spec, sample, card, behavior, categories, target, tracer.index.products
        )
        tracer.bind_customer(customer)
        observed = ObservedCustomer(customer, tracer)
        result = run_session(tracer, sample, observed, catalog_ids, target)
        outcomes.append(_outcome(sample, result))
    return outcomes


def _scenario_metrics(outcomes: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for outcome in outcomes:
        grouped[outcome["scenario_type"]].append(outcome)
    return {
        name: {
            "hit_rate_at_10": score(items)["hit"],
            "mrr": score(items)["mrr"],
            "mttc": score(items)["mttc"],
            "recommended_technical_score": score(items)["score"],
        }
        for name, items in sorted(grouped.items())
    }


def _write_outputs(
    run_dir: Path,
    tracer: StressTracingAgent,
    summary: dict,
    tag: str,
    markdown: bool,
) -> None:
    (run_dir / "sessions").mkdir(parents=True, exist_ok=True)
    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for record in tracer.records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (run_dir / "index.md").write_text(
        render_index_markdown(tracer.records, summary, tag), encoding="utf-8"
    )
    if markdown:
        for record in tracer.records:
            (run_dir / "sessions" / f"{record['sample_id']}.md").write_text(
                render_session_markdown(record), encoding="utf-8"
            )
    (run_dir / "viewer.html").write_text(
        render_viewer(tracer.records, summary, tag), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--customer",
        default="official",
        help="stressor spec, e.g. paraphrase:heavy+browse-gated",
    )
    parser.add_argument("--config", default="", help="optional tools/sweep.py config name")
    parser.add_argument("--out", default="runs/stress-observe", help="run-folder root")
    parser.add_argument("--tag", default="", help="run label; defaults to the customer spec")
    parser.add_argument("--scenario", help="only this scenario_type")
    parser.add_argument("--only", help="comma-separated sample_ids")
    parser.add_argument("--limit", type=int, help="first N sessions after filtering")
    parser.add_argument("--top", type=int, default=10, help="reranked candidates saved per turn")
    parser.add_argument("--targets", choices=["all", "generic"], default="all")
    parser.add_argument("--no-markdown", action="store_true")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.scenario:
        samples = [s for s in samples if s["scenario_type"] == args.scenario]
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        samples = [s for s in samples if s["sample_id"] in wanted]
    if args.limit is not None:
        samples = samples[: max(0, args.limit)]
    if not samples:
        raise SystemExit("no sessions matched the filters")

    spec = parse_spec(args.customer)
    print(f"loading catalog {args.catalog} ...", flush=True)
    catalog_ids, categories, products = catalog_index(args.catalog)

    if args.config:
        from tools.sweep import build_configs

        configs = build_configs(args.catalog)
        if args.config not in configs:
            raise SystemExit(
                f"unknown config {args.config!r}; choose from: {', '.join(sorted(configs))}"
            )
        agent = Agent(args.catalog, configs[args.config])
    else:
        agent = Agent(args.catalog)

    if args.targets == "generic":
        samples = select_generic(samples, products, agent.index)
        if not samples:
            raise SystemExit("no generic targets matched the filtered sessions")

    tracer = StressTracingAgent(agent, samples, products, top_n=args.top)
    print(f"tracing {len(samples)} stress sessions ({args.customer}) ...", flush=True)
    started = time.perf_counter()
    undo = install_stress_probes()
    try:
        outcomes = run_traced(tracer, samples, catalog_ids, categories, products, spec)
    finally:
        undo()
    wall = time.perf_counter() - started

    by_id = {outcome["sample_id"]: outcome for outcome in outcomes}
    for record in tracer.records:
        record["outcome"] = by_id[record["sample_id"]]
        record["diagnosis"] = diagnose(record, record["outcome"])

    counts = Counter(record["diagnosis"]["label"] for record in tracer.records)
    aggregate = score(outcomes)
    tag = args.tag or args.customer.replace(":", "-").replace("+", "_")
    summary = {
        "tag": tag,
        "customer": args.customer,
        "config": args.config or "branch-default",
        "dataset": args.dataset,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "wall_seconds": round(wall, 2),
        "sample_count": len(outcomes),
        "hit_rate_at_10": aggregate["hit"],
        "mrr": aggregate["mrr"],
        "mttc": aggregate["mttc"],
        "efficiency": max(0.0, min(1.0, (11.0 - aggregate["mttc"]) / 10.0)),
        "recommended_technical_score": aggregate["score"],
        "token_coverage": aggregate["tok_cov"],
        "scenario_metrics": _scenario_metrics(outcomes),
        "diagnosis_counts": {label: counts[label] for label in DIAGNOSES if counts[label]},
        "turns_left_on_table": sum(
            record["diagnosis"]["turns_left_on_table"] or 0 for record in tracer.records
        ),
        "mean_turn_latency_ms": round(
            sum(t["latency_ms"] for r in tracer.records for t in r["turns"])
            / max(1, sum(len(r["turns"]) for r in tracer.records)),
            2,
        ),
    }

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out) / f"{tag}-{stamp}"
    _write_outputs(run_dir, tracer, summary, tag, not args.no_markdown)

    latest = Path(args.out) / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink() if latest.is_symlink() else shutil.rmtree(latest)
    try:
        latest.symlink_to(run_dir.name)
    except OSError:
        pass

    print(f"\n{run_dir}")
    print(
        f"  {len(outcomes)} sessions | hit {aggregate['hit']:.3f} | "
        f"MRR {aggregate['mrr']:.4f} | MTTC {aggregate['mttc']:.3f} | "
        f"score {aggregate['score']:.6f}"
    )
    for label, count in summary["diagnosis_counts"].items():
        print(f"  {label:<18} {count:>4}  {DIAGNOSES[label]}")
    print(f"\n  open {run_dir / 'viewer.html'}")
