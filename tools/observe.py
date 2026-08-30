"""Session observability - see exactly what happened in every evaluated session.

Why this is not a modified copy of the evaluator
------------------------------------------------
``evaluator/local_evaluator.evaluate()`` takes the agent as a parameter, so every
message the simulated customer sends and every response the agent returns already
passes through ``Agent.respond``. Wrapping the agent therefore captures the whole
transcript while the *official, unmodified* evaluator drives the session - the
reported metrics stay byte-identical to ``python3 -m evaluator.local_evaluator``,
and this tool asserts that before writing anything.

A forked evaluator would have been worse on both counts: it duplicates ~100 lines
of protocol that can silently drift from the organizer's copy, and it still only
sees the response envelope. The interesting question is never "what did the agent
reply" but "why", so this tool also probes the pipeline stages from the outside
(no production code is edited - the probes are installed on ``starter.agent``'s
module namespace for the duration of the run and removed afterwards).

What you get per turn
---------------------
  * the customer message, and what it disclosed (constraint / decline / override)
  * the routing decision, with the cues and facets that produced it
  * the dialog state: accumulated spans, dead attributes, override turn
  * where the target sat in the retrieval pool, and where after reranking
  * what was actually shown, and - when nothing was - whether the confidence
    gate or the turn floor held it back

The last one is the payoff. A miss reads very differently depending on whether
the target was never retrieved (a recall problem, stage S1/S5) or sat at pool
rank 14 for six turns (a ranking problem, stage S6).

Usage
-----
    python3 tools/observe.py                       # full public set -> runs/<stamp>/
    python3 tools/observe.py --scenario intent_override
    python3 tools/observe.py --only public_0008,public_0042
    python3 tools/observe.py --dataset data/hard_set.jsonl --tag hard
    python3 tools/observe.py --limit 20 --no-markdown   # quick triage

Outputs (under ``runs/<tag>-<timestamp>/``, also linked as ``runs/latest``):

    index.md              one row per session, sorted worst-first
    sessions/<id>.md      full annotated transcript, one file per session
    trace.jsonl           machine-readable, one JSON record per turn
    summary.json          aggregate metrics + failure-mode breakdown
    viewer.html           self-contained offline browser for all of the above
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluator.local_evaluator import (  # noqa: E402  (read-only import; never modified)
    catalog_index,
    coarse_category,
    evaluate,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.agent import Agent  # noqa: E402


# --------------------------------------------------------------------------------
# Stage probes
# --------------------------------------------------------------------------------
# A scratch dict the probes write into. The tracing agent clears it before each
# turn and reads it afterwards, so no pipeline signature changes.
_PROBE: dict = {}


def install_probes():
    """Wrap the stage functions in ``starter.agent``'s namespace. Returns an undo."""
    import starter.agent as agent_module

    original = {
        name: getattr(agent_module, name)
        for name in (
            "classify", "detect_turn_intent", "retrieve", "rerank",
            "AdaptiveOrchestrator",
        )
    }

    def classify_probe(opening):
        route = original["classify"](opening)
        _PROBE["route"] = route
        return route

    def detect_probe(*args, **kwargs):
        route = original["detect_turn_intent"](*args, **kwargs)
        _PROBE["route"] = route
        return route

    def retrieve_probe(index, state, config=None, route_hint=None):
        started = time.perf_counter()
        pool = original["retrieve"](index, state, config, route_hint=route_hint)
        _PROBE["retrieve_ms"] = (time.perf_counter() - started) * 1000.0
        _PROBE["pool"] = pool
        _PROBE["retrieval_route"] = route_hint or "terms"
        return pool

    def rerank_probe(index, state, candidates, config=None):
        started = time.perf_counter()
        ranked = original["rerank"](index, state, candidates, config)
        _PROBE["rerank_ms"] = (time.perf_counter() - started) * 1000.0
        _PROBE["ranked"] = ranked
        return ranked

    class OrchestratorProbe:
        @staticmethod
        def align_strategy(*args, **kwargs):
            plan = original["AdaptiveOrchestrator"].align_strategy(*args, **kwargs)
            _PROBE["plan"] = plan
            return plan

        @staticmethod
        def compute_pool_entropy(*args, **kwargs):
            return original["AdaptiveOrchestrator"].compute_pool_entropy(*args, **kwargs)

    agent_module.classify = classify_probe
    agent_module.detect_turn_intent = detect_probe
    agent_module.retrieve = retrieve_probe
    agent_module.rerank = rerank_probe
    agent_module.AdaptiveOrchestrator = OrchestratorProbe

    def undo() -> None:
        for name, function in original.items():
            setattr(agent_module, name, function)

    return undo


# --------------------------------------------------------------------------------
# Customer-side annotation
# --------------------------------------------------------------------------------
# The simulator's replies are generated by fixed templates in
# evaluator/local_evaluator.py (initial_message / customer_reply). Recognising
# them lets each turn be labelled with what it actually gave the agent. This is
# display only - nothing here feeds the agent.

_DISCLOSE_RE = re.compile(r"^For that, what matters is:\s*(.+?)\.?$", re.S)
_NO_PREF_RE = re.compile(r"^I don't have an additional preference for (\w+)\.?$")
_BOUNDARY_RE = re.compile(r"^I don't have a preference for (\w+); please use your judgment\.?$")
_OVERRIDE_RE = re.compile(r"^Actually, ignore my earlier preference\. What I need is:\s*(.+?)\.?$", re.S)
_STALL_RE = re.compile(r"^Those options are not quite right yet")


def annotate_message(text: str) -> dict:
    """Classify one simulated customer message into a labelled event."""
    text = (text or "").strip()
    match = _DISCLOSE_RE.match(text)
    if match:
        revealed = [part.strip() for part in match.group(1).split(";") if part.strip()]
        return {"kind": "disclosed", "revealed": revealed}
    match = _OVERRIDE_RE.match(text)
    if match:
        return {"kind": "override", "revealed": [match.group(1).strip()]}
    match = _BOUNDARY_RE.match(text)
    if match:
        return {"kind": "boundary_decline", "attribute": match.group(1), "revealed": []}
    match = _NO_PREF_RE.match(text)
    if match:
        return {"kind": "no_preference", "attribute": match.group(1), "revealed": []}
    if _STALL_RE.match(text):
        return {"kind": "stalled", "revealed": []}
    return {"kind": "opening", "revealed": []}


# --------------------------------------------------------------------------------
# Tracing agent
# --------------------------------------------------------------------------------

class TracingAgent:
    """Records every reset/respond while delegating to the real Agent.

    ``evaluate()`` assigns each session a random uuid, so sessions are matched to
    samples by reset order - it calls ``reset`` exactly once per sample, in order.
    The target ``parent_asin`` is used only to annotate the trace; it is never
    passed to the wrapped agent.
    """

    def __init__(self, inner: Agent, samples: list[dict], products: dict, top_n: int = 10):
        self.inner = inner
        self.samples = samples
        self.products = products
        self.top_n = top_n
        self.records: list[dict] = []
        self._index = -1
        self._current: dict | None = None

    # -- official interface ------------------------------------------------------

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._index += 1
        sample = self.samples[self._index]
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, self.products)
        self._current = {
            "sample_id": sample["sample_id"],
            "session_id": session_id,
            "scenario_type": sample["scenario_type"],
            "difficulty_bucket": sample.get("difficulty_bucket"),
            "category_bucket": sample.get("category_bucket"),
            "user_profile": user_profile,
            "target": target,
            "target_title": (self.products.get(target) or {}).get("title", ""),
            "target_price": (self.products.get(target) or {}).get("price"),
            "target_categories": (self.products.get(target) or {}).get("categories") or [],
            "coarse_category": coarse_category(
                [str(v) for v in (self.products.get(target) or {}).get("categories") or []]
            ),
            "intent_card": card,
            "behavior": behavior,
            "turns": [],
        }
        self.records.append(self._current)
        self.inner.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        _PROBE.clear()
        started = time.perf_counter()
        error = None
        try:
            response = self.inner.respond(session_id, user_message, turn, top_k)
        except Exception as exc:  # recorded, then re-raised so behaviour is unchanged
            self._record_turn(session_id, user_message, turn, {}, 0.0, repr(exc))
            raise
        elapsed = (time.perf_counter() - started) * 1000.0
        self._record_turn(session_id, user_message, turn, response, elapsed, error)
        return response

    # -- internals ---------------------------------------------------------------

    def _record_turn(self, session_id, user_message, turn, response, elapsed_ms, error):
        assert self._current is not None
        target = self._current["target"]

        pool = _PROBE.get("pool") or []
        ranked = _PROBE.get("ranked") or []
        route = _PROBE.get("route")
        plan = _PROBE.get("plan")

        shown = [
            str(item.get("parent_asin"))
            for item in (response.get("recommendations") or [])
            if isinstance(item, dict)
        ]

        state = getattr(self.inner, "_states", {}).get(session_id)
        state_view = None
        if state is not None:
            state_view = state.snapshot()

        record = {
            "turn": turn,
            "in": {"message": user_message, **annotate_message(user_message)},
            "route": None
            if route is None
            else {
                "name": route.name,
                "confidence": route.confidence,
                "buying_score": route.buying_score,
                "browsing_score": route.browsing_score,
                "scenario_hint": route.scenario_hint,
                "cues": list(route.detected_cues),
                "facets": {k: list(v) for k, v in (route.detected_facets or {}).items()},
            },
            "state": state_view,
            "plan": None if plan is None else {
                "phase": plan.phase.value,
                "retrieval_route": plan.retrieval_route,
                "recommendation_cutoff": plan.recommendation_cutoff,
                "recommended_slate_size": plan.recommended_slate_size,
                "guidance_action": plan.guidance_action,
            },
            "retrieval": {
                "pool_size": len(pool),
                "target_pool_rank": _rank_of(target, pool),
                "ms": round(_PROBE.get("retrieve_ms", 0.0), 2),
                "route": _PROBE.get("retrieval_route", "terms"),
            },
            "rerank": {
                "target_rank": _rank_of(target, ranked),
                "top": [
                    {
                        "rank": position + 1,
                        "parent_asin": parent_asin,
                        "score": round(float(score), 4),
                        "title": _title(self.products, parent_asin),
                        "is_target": parent_asin == target,
                    }
                    for position, (parent_asin, score) in enumerate(ranked[: self.top_n])
                ],
                "ms": round(_PROBE.get("rerank_ms", 0.0), 2),
            },
            "out": {
                "message": response.get("message"),
                "ask_attribute": response.get("ask_attribute"),
                "shown_count": len(shown),
                "shown": shown,
                "target_shown_rank": _rank_of(target, [(a, 0.0) for a in shown]),
                "withheld": len(shown) == 0,
            },
            "latency_ms": round(elapsed_ms, 2),
            "error": error,
        }
        self._current["turns"].append(record)


def _rank_of(target: str, ranked) -> int | None:
    for position, entry in enumerate(ranked):
        parent_asin = entry[0] if isinstance(entry, tuple) else entry
        if parent_asin == target:
            return position + 1
    return None


def _title(products: dict, parent_asin: str, limit: int = 90) -> str:
    title = str((products.get(parent_asin) or {}).get("title") or "")
    title = re.sub(r"\s+", " ", title).strip()
    return title[:limit] + ("..." if len(title) > limit else "")


# --------------------------------------------------------------------------------
# Diagnosis
# --------------------------------------------------------------------------------

DIAGNOSES = {
    "hit": "target found and scored",
    "never_retrieved": "target never entered the retrieval pool - a recall problem (S1/S5)",
    "ranked_out": "target was in the pool every turn but never reached the shown top 10 - a ranking problem (S6)",
    "withheld_only": "target reached the top 10 but the list was held back on every such turn (S7)",
    "override_locked": "target was shown before the override fired, so the hit could not be counted",
    "exhausted": "10 turns elapsed with the target outside the top 10",
}


def diagnose(record: dict, outcome: dict) -> dict:
    """Explain a session's result from the per-turn evidence."""
    turns = record["turns"]
    pool_ranks = [t["retrieval"]["target_pool_rank"] for t in turns]
    ranked_ranks = [t["rerank"]["target_rank"] for t in turns]
    in_pool = [r for r in pool_ranks if r is not None]
    best_ranked = min([r for r in ranked_ranks if r is not None], default=None)

    # Earliest turn where the target was already inside the top 10 of the ranked
    # list, whether or not it was shown. Compared against the actual hit turn this
    # is the MTTC the timing gate left on the table.
    # For intent_override sessions the evaluator ignores any hit before the
    # override fires (local_evaluator.py:252), so a top-10 placement earlier than
    # that was never convertible and must not be counted as a wasted turn.
    override = (record.get("behavior") or {}).get("override") or {}
    convertible_from = int(override.get("turn", 1)) if override else 1
    earliest_top10 = next(
        (
            t["turn"]
            for t in turns
            if (t["rerank"]["target_rank"] or 99) <= 10 and t["turn"] >= convertible_from
        ),
        None,
    )
    shown_before_override = any(
        t["out"]["target_shown_rank"] is not None for t in turns
    ) and not outcome["hit"]

    if outcome["hit"]:
        label = "hit"
    elif not in_pool:
        label = "never_retrieved"
    elif shown_before_override:
        label = "override_locked"
    elif earliest_top10 is not None and all(
        t["out"]["withheld"] for t in turns if (t["rerank"]["target_rank"] or 99) <= 10
    ):
        label = "withheld_only"
    elif best_ranked is not None and best_ranked > 10:
        label = "ranked_out"
    else:
        label = "exhausted"

    turns_wasted = None
    if outcome["hit"] and earliest_top10 is not None and outcome["first_hit_turn"]:
        turns_wasted = max(0, int(outcome["first_hit_turn"]) - int(earliest_top10))

    return {
        "label": label,
        "explanation": DIAGNOSES[label],
        "best_pool_rank": min(in_pool, default=None),
        "best_ranked_rank": best_ranked,
        "earliest_top10_turn": earliest_top10,
        "turns_left_on_table": turns_wasted,
        "disclosures": sum(len(t["in"].get("revealed") or []) for t in turns),
        "dead_ends": sum(
            1 for t in turns if t["in"]["kind"] in ("no_preference", "stalled")
        ),
    }


# --------------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------------

def render_session_markdown(record: dict) -> str:
    outcome = record["outcome"]
    diagnosis = record["diagnosis"]
    card = record["intent_card"]
    lines: list[str] = []

    verdict = (
        f"HIT at turn {outcome['first_hit_turn']}, position {outcome['best_rank']}"
        if outcome["hit"]
        else "MISS"
    )
    lines += [
        f"# {record['sample_id']} - {verdict}",
        "",
        "| | |",
        "|---|---|",
        f"| Scenario | `{record['scenario_type']}` |",
        f"| Difficulty | `{record.get('difficulty_bucket')}` |",
        f"| Turns used | {len(record['turns'])} |",
        f"| Reciprocal rank | {outcome['reciprocal_rank']:.4f} |",
        f"| Diagnosis | **{diagnosis['label']}** - {diagnosis['explanation']} |",
        f"| Best pool rank / best ranked rank | {diagnosis['best_pool_rank']} / {diagnosis['best_ranked_rank']} |",
    ]
    if diagnosis["turns_left_on_table"]:
        lines.append(
            f"| Turns left on the table | {diagnosis['turns_left_on_table']} "
            f"(target was already top-10 at turn {diagnosis['earliest_top10_turn']}) |"
        )
    lines += [
        "",
        "## Hidden truth (never visible to the agent)",
        "",
        f"- **Target** `{record['target']}` - {record['target_title']}",
        f"- **Price** {record['target_price']}   **Coarse category** {record['coarse_category']}",
        f"- **Hard constraints** {json.dumps(card.get('hard_constraints', []), ensure_ascii=False)}",
        f"- **Soft preferences** {json.dumps(card.get('soft_preferences', []), ensure_ascii=False)}",
    ]
    override = (record.get("behavior") or {}).get("override")
    if override:
        lines.append(
            f"- **Override** fires on turn {override.get('turn')}: "
            f"drops _{override.get('old_value')}_ for _{override.get('new_value')}_"
        )
    profile = record.get("user_profile") or {}
    lines += [
        f"- **Profile** {profile.get('summary', '-')}",
        "",
        "## Transcript",
        "",
    ]

    for turn in record["turns"]:
        lines.append(f"### Turn {turn['turn']}")
        lines.append("")
        kind = turn["in"]["kind"]
        badge = {
            "opening": "opening message",
            "disclosed": "DISCLOSED a new constraint",
            "override": "INTENT OVERRIDE",
            "no_preference": "dead end - no further preference",
            "boundary_decline": "boundary - declined the attribute",
            "stalled": "wasted turn - agent asked nothing",
        }[kind]
        lines.append(f"**Customer** ({badge})")
        lines.append("")
        lines.append(f"> {turn['in']['message']}")
        lines.append("")
        if turn["in"].get("revealed"):
            for item in turn["in"]["revealed"]:
                lines.append(f"  - revealed: `{item}`")
            lines.append("")

        route = turn["route"]
        if route:
            cues = ", ".join(route["cues"]) or "none"
            lines.append(
                f"**Route** `{route['name']}` (confidence {route['confidence']}, "
                f"buying {route['buying_score']} vs browsing {route['browsing_score']}) - cues: {cues}"
            )
            lines.append("")

        state = turn["state"]
        if state:
            lines.append(
                f"**State** phase: `{state['phase']}` ({state['phase_reason']}) | "
                f"intent: `{state['intent']['track']}` | asked: {state['asked'] or '-'} | "
                f"dead: {state['dead_attributes'] or '-'}"
            )
            lines.append("")
            lines.append(
                f"**Progress** productive turns: {state['productive_turns']} | "
                f"unproductive streak: {state['unproductive_streak']} | "
                f"pool entropy: {state['pool']['entropy']} | "
                f"stable pool turns: {state['pool']['stable_turns']}"
            )
            if state["active_slots"]:
                lines.append("")
                lines.append(
                    "**Active slots** `"
                    + json.dumps(state["active_slots"], ensure_ascii=False)
                    + "`"
                )
            if state["superseded_slots"]:
                lines.append("")
                lines.append(
                    "**Superseded slots** `"
                    + json.dumps(state["superseded_slots"], ensure_ascii=False)
                    + "`"
                )
            if state["spans"]:
                lines.append("")
                lines.append(
                    "**Constraint spans in play** "
                    + ", ".join(f"`{span}`" for span in state["spans"])
                )
            lines.append("")

        retrieval, rerank_info = turn["retrieval"], turn["rerank"]
        lines.append(
            f"**Retrieval** pool {retrieval['pool_size']} candidates, "
            f"target at pool rank **{retrieval['target_pool_rank'] or 'not in pool'}** "
            f"-> after rerank **{rerank_info['target_rank'] or 'not in pool'}**"
        )
        lines.append("")
        if rerank_info["top"]:
            lines.append("| # | asin | score | product |")
            lines.append("|---:|---|---:|---|")
            for entry in rerank_info["top"]:
                marker = " **<- TARGET**" if entry["is_target"] else ""
                lines.append(
                    f"| {entry['rank']} | `{entry['parent_asin']}` | {entry['score']} | "
                    f"{entry['title']}{marker} |"
                )
            lines.append("")

        out = turn["out"]
        lines.append(f"**Agent** asks `{out['ask_attribute']}`")
        lines.append("")
        lines.append(f"> {out['message']}")
        lines.append("")
        if out["withheld"]:
            lines.append(
                "  - _list withheld this turn_ - showing an uncertain list ends the "
                "session at whatever rank the target happens to hold"
            )
        else:
            position = out["target_shown_rank"]
            lines.append(
                f"  - showed {out['shown_count']} products; "
                + (f"**target at position {position}**" if position else "target not among them")
            )
        lines.append(f"  - latency {turn['latency_ms']} ms")
        if turn["error"]:
            lines.append(f"  - **EXCEPTION** {turn['error']}")
        lines.append("")

    return "\n".join(lines) + "\n"


def render_index_markdown(records: list[dict], summary: dict, tag: str) -> str:
    lines = [
        f"# Session observations - {tag}",
        "",
        f"{summary['sample_count']} sessions | hit rate {summary['hit_rate_at_10']:.3f} | "
        f"MRR {summary['mrr']:.4f} | MTTC {summary['mttc']:.3f} | "
        f"score **{summary['recommended_technical_score']:.6f}**",
        "",
        "## Failure modes",
        "",
        "| diagnosis | sessions | meaning |",
        "|---|---:|---|",
    ]
    for label, count in summary["diagnosis_counts"].items():
        lines.append(f"| `{label}` | {count} | {DIAGNOSES[label]} |")
    lines += [
        "",
        f"Turns left on the table across all hits: **{summary['turns_left_on_table']}** "
        "(the target was already in the top 10 before the list was shown).",
        "",
        "## Sessions (misses first, then worst rank)",
        "",
        "| session | scenario | outcome | turn | rank | diagnosis | best pool | target |",
        "|---|---|---|---:|---:|---|---:|---|",
    ]
    ordered = sorted(
        records,
        key=lambda r: (
            r["outcome"]["hit"],
            -(r["outcome"]["best_rank"] or 0),
            r["sample_id"],
        ),
    )
    for record in ordered:
        outcome, diagnosis = record["outcome"], record["diagnosis"]
        lines.append(
            f"| [{record['sample_id']}](sessions/{record['sample_id']}.md) "
            f"| {record['scenario_type']} "
            f"| {'hit' if outcome['hit'] else 'MISS'} "
            f"| {outcome['first_hit_turn'] or '-'} "
            f"| {outcome['best_rank'] or '-'} "
            f"| `{diagnosis['label']}` "
            f"| {diagnosis['best_pool_rank'] or '-'} "
            f"| {record['target_title'][:60]} |"
        )
    return "\n".join(lines) + "\n"


VIEWER_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Session observations - __TAG__</title>
<style>
  :root {
    --bg:#fbfaf8; --panel:#ffffff; --ink:#1b1a18; --muted:#6b6660; --line:#e3ded7;
    --hit:#1c7a4a; --miss:#b3261e; --accent:#0b5fa5; --target:#fff4c2; --code:#f3f0eb;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16151a; --panel:#1e1d24; --ink:#eceaf2; --muted:#a09aa8; --line:#33313c;
            --hit:#5fd39a; --miss:#ff8a80; --accent:#7db6f0; --target:#4a4020; --code:#26252e; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }
  header { padding:14px 20px; border-bottom:1px solid var(--line); background:var(--panel);
           position:sticky; top:0; z-index:5; }
  header h1 { margin:0 0 6px; font-size:16px; letter-spacing:.01em; }
  .metrics { color:var(--muted); font-size:13px; }
  .metrics b { color:var(--ink); }
  .wrap { display:grid; grid-template-columns:minmax(300px,380px) 1fr; height:calc(100vh - 74px); }
  @media (max-width:860px) { .wrap { grid-template-columns:1fr; height:auto; } }
  .list { border-right:1px solid var(--line); overflow:auto; background:var(--panel); }
  .filters { padding:10px 12px; border-bottom:1px solid var(--line); display:flex; flex-wrap:wrap; gap:6px; }
  .filters select, .filters input {
    font:inherit; padding:5px 7px; border:1px solid var(--line); border-radius:6px;
    background:var(--bg); color:var(--ink); flex:1 1 110px; min-width:0; }
  .row { padding:9px 12px; border-bottom:1px solid var(--line); cursor:pointer; }
  .row:hover { background:var(--code); }
  .row.sel { background:var(--code); box-shadow:inset 3px 0 0 var(--accent); }
  .row .id { font-weight:600; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  .row .meta { color:var(--muted); font-size:12px; margin-top:2px; }
  .hit { color:var(--hit); font-weight:600; } .miss { color:var(--miss); font-weight:600; }
  .detail { overflow:auto; padding:20px 24px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
          padding:14px 16px; margin-bottom:14px; }
  .turn { border-left:3px solid var(--line); padding-left:14px; margin:16px 0; }
  .turn h4 { margin:0 0 8px; font-size:13px; color:var(--muted); text-transform:uppercase;
             letter-spacing:.06em; }
  blockquote { margin:6px 0; padding:8px 12px; background:var(--code); border-radius:6px; }
  table { border-collapse:collapse; width:100%; margin:8px 0; font-size:13px; }
  th,td { text-align:left; padding:4px 8px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; }
  tr.tgt td { background:var(--target); font-weight:600; }
  code, .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; }
  .tag { display:inline-block; padding:1px 7px; border-radius:999px; background:var(--code);
         color:var(--muted); font-size:12px; margin-right:5px; }
  .withheld { color:var(--muted); font-style:italic; }
  .empty { color:var(--muted); padding:40px; text-align:center; }
</style></head><body>
<header>
  <h1>Session observations &mdash; __TAG__</h1>
  <div class="metrics" id="metrics"></div>
</header>
<div class="wrap">
  <div class="list">
    <div class="filters">
      <select id="f-outcome"><option value="">all outcomes</option><option value="hit">hits</option><option value="miss">misses</option></select>
      <select id="f-scenario"><option value="">all scenarios</option></select>
      <select id="f-diagnosis"><option value="">all diagnoses</option></select>
      <input id="f-text" placeholder="search id / title">
    </div>
    <div id="rows"></div>
  </div>
  <div class="detail" id="detail"><div class="empty">Select a session.</div></div>
</div>
<script>
const DATA = __DATA__;
const SUMMARY = __SUMMARY__;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
document.getElementById("metrics").innerHTML =
  `<b>${SUMMARY.sample_count}</b> sessions &middot; hit rate <b>${SUMMARY.hit_rate_at_10.toFixed(3)}</b>`
  + ` &middot; MRR <b>${SUMMARY.mrr.toFixed(4)}</b> &middot; MTTC <b>${SUMMARY.mttc.toFixed(3)}</b>`
  + ` &middot; score <b>${SUMMARY.recommended_technical_score.toFixed(6)}</b>`
  + ` &middot; ${SUMMARY.turns_left_on_table} turns left on the table`;

const fill = (id, values) => { const el = document.getElementById(id);
  [...new Set(values)].sort().forEach(v => el.insertAdjacentHTML("beforeend", `<option>${esc(v)}</option>`)); };
fill("f-scenario", DATA.map(d => d.scenario_type));
fill("f-diagnosis", DATA.map(d => d.diagnosis.label));

function visible() {
  const o = f("f-outcome"), s = f("f-scenario"), d = f("f-diagnosis"), t = f("f-text").toLowerCase();
  return DATA.filter(r =>
    (!o || (o === "hit") === !!r.outcome.hit) &&
    (!s || r.scenario_type === s) && (!d || r.diagnosis.label === d) &&
    (!t || (r.sample_id + " " + r.target_title).toLowerCase().includes(t)));
}
const f = id => document.getElementById(id).value;

function renderRows() {
  const rows = visible();
  document.getElementById("rows").innerHTML = rows.length ? rows.map(r => `
    <div class="row" data-id="${esc(r.sample_id)}">
      <div class="id">${esc(r.sample_id)}
        <span class="${r.outcome.hit ? "hit" : "miss"}">${r.outcome.hit
          ? "hit t" + r.outcome.first_hit_turn + " #" + r.outcome.best_rank : "MISS"}</span></div>
      <div class="meta">${esc(r.scenario_type)} &middot; ${esc(r.diagnosis.label)}
        &middot; best rank ${r.diagnosis.best_ranked_rank ?? "-"}</div>
      <div class="meta">${esc(r.target_title.slice(0, 70))}</div>
    </div>`).join("") : '<div class="empty">No sessions match.</div>';
  document.querySelectorAll(".row").forEach(el =>
    el.onclick = () => { document.querySelectorAll(".row").forEach(x => x.classList.remove("sel"));
                         el.classList.add("sel"); renderDetail(el.dataset.id); });
}

function renderDetail(id) {
  const r = DATA.find(x => x.sample_id === id);
  const c = r.intent_card, ov = (r.behavior || {}).override;
  let h = `<div class="card"><h2 style="margin:0 0 8px">${esc(r.sample_id)}
    <span class="${r.outcome.hit ? "hit" : "miss"}">${r.outcome.hit
      ? "HIT turn " + r.outcome.first_hit_turn + ", position " + r.outcome.best_rank : "MISS"}</span></h2>
    <div><span class="tag">${esc(r.scenario_type)}</span><span class="tag">${esc(r.difficulty_bucket)}</span>
         <span class="tag">RR ${r.outcome.reciprocal_rank.toFixed(4)}</span>
         <span class="tag">${r.turns.length} turns</span></div>
    <p><b>${esc(r.diagnosis.label)}</b> &mdash; ${esc(r.diagnosis.explanation)}<br>
    best pool rank ${r.diagnosis.best_pool_rank ?? "-"}, best ranked rank ${r.diagnosis.best_ranked_rank ?? "-"}
    ${r.diagnosis.turns_left_on_table ? `, <b>${r.diagnosis.turns_left_on_table} turn(s) left on the table</b>
      (top-10 already at turn ${r.diagnosis.earliest_top10_turn})` : ""}</p></div>
    <div class="card"><h3 style="margin:0 0 6px">Hidden truth</h3>
    <div class="mono">${esc(r.target)}</div><div>${esc(r.target_title)}</div>
    <p>price ${esc(r.target_price)} &middot; category ${esc(r.coarse_category)}</p>
    <p><b>hard</b> ${esc(JSON.stringify(c.hard_constraints))}<br>
       <b>soft</b> ${esc(JSON.stringify(c.soft_preferences))}
       ${ov ? `<br><b>override</b> turn ${ov.turn}: drops "${esc(ov.old_value)}" for "${esc(ov.new_value)}"` : ""}</p>
    <p class="mono">${esc((r.user_profile || {}).summary || "")}</p></div>`;

  for (const t of r.turns) {
    const st = t.state || {}, rr = t.rerank, out = t.out;
    h += `<div class="turn"><h4>Turn ${t.turn} &middot; ${esc(t.in.kind)}</h4>
      <blockquote>${esc(t.in.message)}</blockquote>
      ${(t.in.revealed || []).map(v => `<div class="mono">+ revealed: ${esc(v)}</div>`).join("")}
      ${t.route ? `<p><span class="tag">route ${esc(t.route.name)} @ ${t.route.confidence}</span>
        <span class="tag">cues: ${esc((t.route.cues || []).join(", ") || "none")}</span></p>` : ""}
      <p><span class="tag">phase ${esc(st.phase || "-")}</span>
         <span class="tag">intent ${esc((st.intent || {}).track || "-")}</span>
         <span class="tag">streak ${st.unproductive_streak ?? 0}</span>
         <span class="tag">asked ${esc((st.asked || []).join(", ") || "-")}</span>
         <span class="tag">dead ${esc((st.dead_attributes || []).join(", ") || "-")}</span>
         <span class="tag">override turn ${st.override_turn ?? "-"}</span></p>
      <p>${esc(st.phase_reason || "")}</p>
      ${st.active_slots && Object.keys(st.active_slots).length
        ? `<p class="mono">active slots: ${esc(JSON.stringify(st.active_slots))}</p>` : ""}
      ${(st.superseded_slots || []).length
        ? `<p class="mono">superseded: ${esc(JSON.stringify(st.superseded_slots))}</p>` : ""}
      ${(st.spans || []).length ? `<p class="mono">spans: ${esc(st.spans.join(" | "))}</p>` : ""}
      <p>pool ${t.retrieval.pool_size} via ${esc(t.retrieval.route || "terms")} &middot; target at pool rank
         <b>${t.retrieval.target_pool_rank ?? "not in pool"}</b> &rarr; after rerank
         <b>${rr.target_rank ?? "not in pool"}</b></p>
      <table><tr><th>#</th><th>asin</th><th>score</th><th>product</th></tr>
        ${rr.top.map(e => `<tr class="${e.is_target ? "tgt" : ""}"><td>${e.rank}</td>
          <td class="mono">${esc(e.parent_asin)}</td><td>${e.score}</td><td>${esc(e.title)}</td></tr>`).join("")}
      </table>
      <p><b>Agent</b> asks <code>${esc(out.ask_attribute)}</code></p>
      <blockquote>${esc(out.message)}</blockquote>
      ${out.withheld
        ? `<p class="withheld">list withheld this turn</p>`
        : `<p>showed ${out.shown_count} &middot; ${out.target_shown_rank
            ? "<b>target at position " + out.target_shown_rank + "</b>" : "target not among them"}</p>`}
      <p class="mono">${t.latency_ms} ms${t.error ? " &middot; EXCEPTION " + esc(t.error) : ""}</p></div>`;
  }
  document.getElementById("detail").innerHTML = h;
  document.getElementById("detail").scrollTop = 0;
}

["f-outcome", "f-scenario", "f-diagnosis", "f-text"].forEach(id =>
  document.getElementById(id).addEventListener("input", renderRows));
renderRows();
</script></body></html>
"""


def render_viewer(records: list[dict], summary: dict, tag: str) -> str:
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    return (
        VIEWER_TEMPLATE.replace("__DATA__", payload)
        .replace("__SUMMARY__", json.dumps(summary, ensure_ascii=False).replace("</", "<\\/"))
        .replace("__TAG__", html.escape(tag))
    )


# --------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--out", default="runs", help="root directory for run folders")
    parser.add_argument("--tag", default="public", help="label for this run")
    parser.add_argument("--scenario", help="only sessions of this scenario_type")
    parser.add_argument("--only", help="comma-separated sample_ids")
    parser.add_argument("--limit", type=int, help="first N sessions after filtering")
    parser.add_argument("--top", type=int, default=10, help="candidates recorded per turn")
    parser.add_argument("--no-markdown", action="store_true", help="skip the per-session .md files")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="re-run the official evaluator untraced and assert the score is identical",
    )
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.scenario:
        samples = [s for s in samples if s["scenario_type"] == args.scenario]
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        samples = [s for s in samples if s["sample_id"] in wanted]
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("no sessions matched the filters")

    print(f"loading catalog {args.catalog} ...", flush=True)
    catalog_ids, categories, products = catalog_index(args.catalog)

    agent = Agent(args.catalog)
    tracer = TracingAgent(agent, samples, products, top_n=args.top)

    print(f"tracing {len(samples)} sessions ...", flush=True)
    started = time.perf_counter()
    undo = install_probes()
    try:
        result = evaluate(tracer, samples, catalog_ids, categories, products)
    finally:
        undo()
    wall = time.perf_counter() - started

    if args.verify:
        print("verifying against an untraced official run ...", flush=True)
        control = evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)
        traced_score = result["recommended_technical_score"]
        control_score = control["recommended_technical_score"]
        if abs(traced_score - control_score) > 1e-9:
            raise SystemExit(
                f"tracing changed behaviour: {traced_score} vs {control_score}"
            )
        print(f"  identical: {traced_score:.6f}")

    outcomes = {session["sample_id"]: session for session in result["sessions"]}
    for record in tracer.records:
        record["outcome"] = outcomes[record["sample_id"]]
        record["diagnosis"] = diagnose(record, record["outcome"])

    counts = Counter(record["diagnosis"]["label"] for record in tracer.records)
    summary = {
        "tag": args.tag,
        "dataset": args.dataset,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "wall_seconds": round(wall, 2),
        "sample_count": result["sample_count"],
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "efficiency": result["efficiency"],
        "recommended_technical_score": result["recommended_technical_score"],
        "scenario_metrics": result["scenario_metrics"],
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
    run_dir = Path(args.out) / f"{args.tag}-{stamp}"
    (run_dir / "sessions").mkdir(parents=True, exist_ok=True)

    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for record in tracer.records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (run_dir / "index.md").write_text(
        render_index_markdown(tracer.records, summary, args.tag), encoding="utf-8"
    )
    if not args.no_markdown:
        for record in tracer.records:
            (run_dir / "sessions" / f"{record['sample_id']}.md").write_text(
                render_session_markdown(record), encoding="utf-8"
            )
    (run_dir / "viewer.html").write_text(
        render_viewer(tracer.records, summary, args.tag), encoding="utf-8"
    )

    latest = Path(args.out) / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink() if latest.is_symlink() else shutil.rmtree(latest)
    try:
        latest.symlink_to(run_dir.name)
    except OSError:
        pass

    print(f"\n{run_dir}")
    print(
        f"  {summary['sample_count']} sessions | hit {summary['hit_rate_at_10']:.3f} "
        f"| MRR {summary['mrr']:.4f} | MTTC {summary['mttc']:.3f} "
        f"| score {summary['recommended_technical_score']:.6f}"
    )
    for label, count in summary["diagnosis_counts"].items():
        print(f"  {label:<18} {count:>4}  {DIAGNOSES[label]}")
    print(f"  turns left on the table: {summary['turns_left_on_table']}")
    print(f"\n  open {run_dir / 'viewer.html'}")


if __name__ == "__main__":
    main()
