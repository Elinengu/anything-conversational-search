"""Dual-track realism harness - a stricter customer for the browsing track.

The official simulator (evaluator/local_evaluator.py) hands over *every*
undisclosed constraint the moment the agent asks ``ask_attribute="other"``,
regardless of scenario. So a decided buyer and a vague browser are drained
identically and ``FixedPolicy("other")`` is unbeatable - there is nothing for
buying-vs-browsing routing to do (docs/team/future_steps.md).

This harness keeps the buying / intent_override / boundary customers exactly as
the organizer wrote them and makes only the **browsing** customer behave like a
real one: with no shopping list to recite, they disclose a constraint only when
the agent asks a *pointed* question whose attribute matches
(``classify_constraint``), never on a broad "anything else?". Now the browsing
track needs a real clarification policy, and routing has a measurable job.

It never edits evaluator/ or data/. It wraps ``local_evaluator.customer_reply``
(and, for --misroute-matrix, ``starter.agent`` routing) around an unmodified
``evaluate()`` and restores everything in a ``finally`` - the same technique
tools/observe.py uses.

Usage:
    python3 tools/dual_track_harness.py --dataset data/public_set.jsonl
    python3 tools/dual_track_harness.py --split dev --configs router_off,router_on
    python3 tools/dual_track_harness.py --misroute-matrix
    python3 tools/dual_track_harness.py --verify
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator import local_evaluator  # noqa: E402
from evaluator.local_evaluator import (  # noqa: E402
    ALLOWED_ATTRIBUTES,
    catalog_index,
    classify_constraint,
    evaluate,
    load_jsonl,
)
from src.router import BROWSING, BUYING  # noqa: E402
from starter import agent as agent_module  # noqa: E402
from starter.agent import Agent  # noqa: E402
from tools.sweep import build_configs, split_samples  # noqa: E402


# How many constraints a scenario's customer discloses per kind of question.
# "broad" = ask_attribute in (None, "other"); "targeted" = a named attribute.
# Buying is the organizer's behaviour (unchanged). Browsing is the realism edit.
DISCLOSURE = {
    "buying": {"broad": 2, "targeted": 2},
    "browsing": {"broad": 0, "targeted": 1},
}


def _make_reply(disclosure: dict[str, dict[str, int]]):
    """A customer_reply wrapper. Buying / override / boundary are the organizer's
    behaviour verbatim; browsing discloses per its per-question-kind budget."""
    original = local_evaluator.customer_reply

    def customer_reply(sample, ask_attribute, disclosed, boundary_used):
        scenario = sample["scenario_type"]
        rule = disclosure.get(scenario)
        if scenario != "browsing" or rule == DISCLOSURE_NEUTRAL[scenario]:
            return original(sample, ask_attribute, disclosed, boundary_used)

        broad = not isinstance(ask_attribute, str) or ask_attribute == "other"
        budget = rule["broad"] if broad else rule["targeted"]
        if broad and budget <= 0:
            return (
                "I'm still just browsing - ask me about one particular thing "
                "and I'll tell you.",
                boundary_used,
            )

        attribute = ask_attribute if isinstance(ask_attribute, str) else None
        if attribute is None:
            return original(sample, ask_attribute, disclosed, boundary_used)
        if attribute not in ALLOWED_ATTRIBUTES:
            attribute = "other"
        constraints = [
            *[str(v) for v in sample["intent_card"].get("hard_constraints", [])],
            *[str(v) for v in sample["intent_card"].get("soft_preferences", [])],
        ]
        matches = [
            v for v in constraints
            if v not in disclosed
            and (attribute == "other" or classify_constraint(v) == attribute)
        ][:budget]
        if not matches:
            return (f"I don't have an additional preference for {attribute}.", boundary_used)
        disclosed.update(matches)
        return ("For that, what matters is: " + "; ".join(matches) + ".", boundary_used)

    return customer_reply


#: The organizer's own per-question disclosure, used by --verify to make the
#: wrapper a no-op and by the browsing short-circuit above.
DISCLOSURE_NEUTRAL = {
    "buying": {"broad": 2, "targeted": 2},
    "browsing": {"broad": 2, "targeted": 2},
    "intent_override": {"broad": 2, "targeted": 2},
    "boundary": {"broad": 2, "targeted": 2},
}


@contextlib.contextmanager
def _patched_simulator(disclosure: dict[str, dict[str, int]]):
    saved = local_evaluator.customer_reply
    local_evaluator.customer_reply = _make_reply(disclosure)
    try:
        yield
    finally:
        local_evaluator.customer_reply = saved


@contextlib.contextmanager
def _forced_route(track: str | None):
    """Pin the agent's router to one track (for --misroute-matrix)."""
    if track is None:
        yield
        return
    route = BUYING if track == "buying" else BROWSING
    saved_c, saved_d = agent_module.classify, agent_module.detect_turn_intent
    agent_module.classify = lambda _opening: route
    agent_module.detect_turn_intent = lambda *a, **k: route
    try:
        yield
    finally:
        agent_module.classify, agent_module.detect_turn_intent = saved_c, saved_d


def _run(config, samples, catalog_ids, categories, products, disclosure, forced=None):
    with _patched_simulator(disclosure), _forced_route(forced):
        return evaluate(Agent("data/catalog.jsonl", config), samples, catalog_ids, categories, products)


def _scen_row(result: dict) -> str:
    parts = []
    for scenario, m in sorted(result["scenario_metrics"].items()):
        parts.append(f"{scenario[:4]}={m['hit_rate_at_10']:.2f}/{m['mrr']:.2f}/{m['mttc']:.1f}")
    return "  ".join(parts)


def _print_result(name: str, result: dict) -> None:
    print(
        f"{name:<22} hit={result['hit_rate_at_10']:.3f} mrr={result['mrr']:.3f} "
        f"mttc={result['mttc']:.2f} score={result['recommended_technical_score']:.4f}"
    )
    print(f"  scenario (hit/mrr/mttc): {_scen_row(result)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--split", default="all", choices=["dev", "holdout", "all"])
    parser.add_argument("--configs", default="router_off,router_on",
                        help="comma-separated build_configs() names")
    parser.add_argument("--misroute-matrix", action="store_true",
                        help="force each track and tabulate true x routed (router_on config)")
    parser.add_argument("--verify", action="store_true",
                        help="disable the browsing intercept and assert parity with the official evaluator")
    args = parser.parse_args()

    samples = split_samples(load_jsonl(args.dataset), args.split)
    catalog_ids, categories, products = catalog_index(args.catalog)
    configs = build_configs(args.catalog)

    print(f"dataset={args.dataset} split={args.split} sessions={len(samples)}")
    print(f"disclosure={DISCLOSURE}\n")

    if args.verify:
        neutral = {k: DISCLOSURE["buying"] for k in DISCLOSURE}
        patched = _run(configs["router_off"], samples, catalog_ids, categories, products, neutral)
        with _forced_route(None):
            official = evaluate(Agent(args.catalog, configs["router_off"]),
                                samples, catalog_ids, categories, products)
        d = abs(patched["recommended_technical_score"] - official["recommended_technical_score"])
        print(f"patched(neutral)={patched['recommended_technical_score']:.9f}  "
              f"official={official['recommended_technical_score']:.9f}  delta={d:.2e}")
        print("PASS" if d < 1e-9 else "FAIL")
        sys.exit(0 if d < 1e-9 else 1)

    if args.misroute_matrix:
        config = configs["router_on"]
        for forced in ("buying", "browsing"):
            result = _run(config, samples, catalog_ids, categories, products, DISCLOSURE, forced=forced)
            sm = result["scenario_metrics"]
            print(f"routed as {forced:<9}: "
                  + "  ".join(
                      f"true-{s}={sm[s]['hit_rate_at_10']:.2f}/{sm[s]['mrr']:.2f}/mttc{sm[s]['mttc']:.1f}"
                      for s in ("buying", "browsing") if s in sm))
        return

    wanted = [n.strip() for n in args.configs.split(",") if n.strip()]
    for name in wanted:
        if name not in configs:
            parser.error(f"unknown config: {name}")
        result = _run(configs[name], samples, catalog_ids, categories, products, DISCLOSURE)
        _print_result(name, result)


if __name__ == "__main__":
    main()
