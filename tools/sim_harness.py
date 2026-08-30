"""Realism harness - evaluate the agent against customer behaviours the official
simulator cannot produce.

`evaluator/local_evaluator.py` is a fully-cooperative, templated, deterministic
simulator: it always answers when asked, discloses constraints copied verbatim
from the target's own metadata, and its "intent override" never retracts anything
real. So it cannot measure Pillar I routing, Pillar II proactive guidance /
genuine override, or Pillar III adaptation - there is no difficult customer to
handle.

This harness drives the *unmodified* `Agent` through the *same* session protocol
(a faithful copy of `evaluate()`), but swaps in customer policies that stress the
things the official metric is blind to:

  official   - byte-identical replay of the official simulator (control; --verify
               asserts it reproduces `python3 -m evaluator.local_evaluator`).
  paraphrase - the same constraints, reworded so they are no longer verbatim
               catalog text. Directly probes the verbatim-span reranker against
               the private set's "may paraphrase" clause. --level light|medium.
  decoy      - intent_override sessions where the pre-override preference is a
               GENUINE decoy (a facet value the target does not have), so the
               override is a real retraction the agent must recover from.

Nothing here modifies the agent or the evaluator. Reported numbers are NOT the
official score - they are a robustness probe.

    python3 tools/sim_harness.py --customer official --verify
    python3 tools/sim_harness.py --customer paraphrase --level medium
    python3 tools/sim_harness.py --customer decoy --dataset data/public_set.jsonl
    python3 tools/sim_harness.py --all            # every customer, side by side
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluator.local_evaluator import (  # noqa: E402  (read-only import)
    ALLOWED_ATTRIBUTES,
    MAX_TURNS,
    TOP_K,
    catalog_index,
    classify_constraint,
    coarse_category,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from src.facets import VOCABULARIES, extract  # noqa: E402
from src.text import terms  # noqa: E402
from starter.agent import Agent  # noqa: E402


# --------------------------------------------------------------------------------
# Paraphrasing (rule-based, deterministic per session)
# --------------------------------------------------------------------------------

_LEADINS = [
    "I'd also want it to be {}.",
    "It should be {}.",
    "Also important: {}.",
    "One more thing - {}.",
    "{} matters to me as well.",
    "Ideally it's {}.",
]

_HYPHEN = {
    "stainless steel": "stainless-steel", "full grain": "full-grain",
    "long sleeve": "long-sleeved", "short sleeve": "short-sleeved",
    "high waisted": "high-waisted", "slim fit": "slim-fit",
    "water resistant": "water-resistant",
}


def _reword_one(value: str, rng: random.Random) -> str:
    """Reword a single constraint string so it is no longer verbatim catalog text,
    while a human would still read it as the same requirement."""
    v = value.strip()
    low = v.lower()

    if low.startswith("color:"):
        col = low.split(":", 1)[1].strip()
        return rng.choice([f"in {col}", f"{col} coloured", f"the {col} one", f"{col} in colour"])

    if "budget around $" in low:
        amt = low.split("$", 1)[1].strip()
        return rng.choice([f"around ${amt}", f"roughly {amt} dollars", f"my budget is about ${amt}",
                           f"nothing over ${amt} really"])

    import re as _re
    m = _re.match(r"(\d+)%\s+(.+)", v)
    if m:
        pct, mat = m.group(1), m.group(2).lower()
        return rng.choice([f"all {mat}", f"{mat} through and through", f"{pct} percent {mat}",
                           f"mostly {mat}"])

    m = _re.match(r"(.+?)\s+closure$", low)
    if m:
        kind = m.group(1)
        return rng.choice([f"a {kind} fastening", f"it does up with a {kind}", f"{kind}-fastened"])

    for phrase, hyph in _HYPHEN.items():
        if phrase in low:
            return low.replace(phrase, hyph)

    # default: strip an incidental leading article / reorder lightly
    toks = v.split()
    if len(toks) >= 3 and rng.random() < 0.5:
        return " ".join(toks[1:] + toks[:1]).lower()
    return low


def paraphrase_disclosure(matches: list[str], level: str, rng: random.Random) -> str:
    """Turn the official 'For that, what matters is: A; B.' into reworded prose."""
    if level == "light":
        parts = [rng.choice(_LEADINS).format(m) for m in matches]  # verbatim m, new frame
    else:  # medium
        parts = [rng.choice(_LEADINS).format(_reword_one(m, rng)) for m in matches]
    return " ".join(parts)


# --------------------------------------------------------------------------------
# Customer policies
# --------------------------------------------------------------------------------

class Customer:
    """Base = a faithful copy of the official simulator's disclosure logic."""

    def __init__(self, sample: dict, card: dict, behavior: dict,
                 categories: dict, target: str, rng: random.Random):
        self.sample = {**sample, "intent_card": card, "behavior": behavior}
        self.card = card
        self.behavior = behavior
        self.scenario = sample["scenario_type"]
        self.category = coarse_category(categories.get(target, []))
        self.rng = rng
        self.disclosed: set[str] = set()
        self.boundary_used = False
        self.override_applied = self.scenario != "intent_override"
        self.override_turn = int((behavior.get("override") or {}).get("turn", 3))

    # -- opening --------------------------------------------------------------
    def opening(self) -> str:
        if self.scenario == "buying" and self.card.get("hard_constraints"):
            c = str(self.card["hard_constraints"][0])
            self.disclosed.add(c)
            return f"I'm looking for {self.category}. A key requirement is: {c}."
        if self.scenario == "intent_override":
            return f"I'm looking for {self.category}. {self.behavior['override']['old_value']}"
        return f"I'm looking for {self.category}, but I'm still exploring."

    # -- per-turn reply ----------------------------------------------------------
    def reply(self, turn: int, ask_attribute: object) -> str:
        # Override injection: mirrors evaluate() - fires when turn+1 == override_turn.
        if not self.override_applied and turn + 1 == self.override_turn:
            self.override_applied = True
            nv = str(self.behavior["override"].get("new_value", ""))
            if nv:
                self.disclosed.add(nv)
            return str(self.behavior["override"]["message"])
        return self._disclose(ask_attribute)

    def _pick_matches(self, ask_attribute: object) -> tuple[str, str | None, list[str]]:
        attribute = ask_attribute if isinstance(ask_attribute, str) else None
        if self.scenario == "boundary" and not self.boundary_used and attribute:
            self.boundary_used = True
            return "BOUNDARY", attribute, []
        if not attribute:
            return "STALL", None, []
        if attribute not in ALLOWED_ATTRIBUTES:
            attribute = "other"
        constraints = [
            *[str(v) for v in self.card.get("hard_constraints", [])],
            *[str(v) for v in self.card.get("soft_preferences", [])],
        ]
        matches = [
            v for v in constraints
            if v not in self.disclosed
            and (attribute == "other" or classify_constraint(v) == attribute)
        ][:2]
        return "DISCLOSE", attribute, matches

    def _disclose(self, ask_attribute: object) -> str:
        kind, attribute, matches = self._pick_matches(ask_attribute)
        if kind == "BOUNDARY":
            return f"I don't have a preference for {attribute}; please use your judgment."
        if kind == "STALL":
            return "Those options are not quite right yet. Ask me about one specific attribute."
        if not matches:
            return f"I don't have an additional preference for {attribute}."
        self.disclosed.update(matches)
        return self._render(matches)

    def _render(self, matches: list[str]) -> str:
        return "For that, what matters is: " + "; ".join(matches) + "."


class OfficialCustomer(Customer):
    pass


class ParaphraseCustomer(Customer):
    def __init__(self, *a, level: str = "medium", **kw):
        super().__init__(*a, **kw)
        self.level = level

    def _render(self, matches: list[str]) -> str:
        return paraphrase_disclosure(matches, self.level, self.rng)

    def opening(self) -> str:
        text = super().opening()
        # rephrase the "A key requirement is: X." tail on buying openings
        if "A key requirement is:" in text:
            head, c = text.split("A key requirement is:", 1)
            c = c.strip().rstrip(".")
            reworded = c if self.level == "light" else _reword_one(c, self.rng)
            return f"{head.strip()} I really need it to be {reworded}."
        return text


class DecoyOverrideCustomer(Customer):
    """intent_override with a genuine decoy as the pre-override preference."""

    injected = 0
    eligible = 0

    def __init__(self, *a, index_products: dict | None = None, target: str = "", **kw):
        super().__init__(*a, target=target, **kw)
        self.decoy_value: str | None = None
        if self.scenario == "intent_override":
            DecoyOverrideCustomer.eligible += 1
            if index_products and target in index_products:
                decoy = self._decoy(index_products[target])
                if decoy:
                    self.decoy_value = decoy
                    DecoyOverrideCustomer.injected += 1
                    self.behavior = {**self.behavior,
                                     "override": {**self.behavior["override"], "old_value": decoy}}
                    self.sample["behavior"] = self.behavior

    @staticmethod
    def _decoy(product: dict) -> str | None:
        facets = extract(product)
        text = product.get("text", "")
        for attr in ("color", "material"):
            if attr not in facets:
                continue
            have = facets[attr]
            for cand in VOCABULARIES[attr]:
                if cand != have and cand not in text:
                    return f"color: {cand}" if attr == "color" else cand
        return None


# --------------------------------------------------------------------------------
# Session runner (faithful copy of evaluate()'s loop)
# --------------------------------------------------------------------------------

def run_session(agent: Agent, sample: dict, customer: Customer,
                catalog_ids: set[str], target: str) -> dict:
    sid = "sess_" + str(sample.get("sample_id", id(sample)))
    agent.reset(sid, sample["user_profile"])
    msg = customer.opening()
    hit_turn = best_rank = None
    for turn in range(1, MAX_TURNS + 1):
        try:
            resp = agent.respond(sid, msg, turn, TOP_K)
        except Exception:
            resp = {"message": "", "ask_attribute": None, "recommendations": []}
        if not isinstance(resp, dict) or not isinstance(resp.get("message"), str):
            resp = {"message": "", "ask_attribute": None, "recommendations": []}
        ranked = normalize_recommendations(resp.get("recommendations"), catalog_ids)
        if customer.override_applied and target in ranked:
            best_rank = ranked.index(target) + 1
            hit_turn = turn
            break
        if turn == MAX_TURNS:
            break
        msg = customer.reply(turn, resp.get("ask_attribute"))

    # extraction-coverage proxy: how many ground-truth constraint tokens the agent
    # actually accumulated (paraphrasing erodes this).
    gt_tokens: set[str] = set()
    for c in [*customer.card.get("hard_constraints", []), *customer.card.get("soft_preferences", [])]:
        gt_tokens |= set(terms(str(c)))
    state = agent._states.get(sid)
    seen = set(terms(state.full_text())) if state is not None else set()
    coverage = len(gt_tokens & seen) / len(gt_tokens) if gt_tokens else 1.0

    return {
        "scenario_type": sample["scenario_type"],
        "hit": hit_turn is not None,
        "first_hit_turn": hit_turn,
        "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        "token_coverage": coverage,
    }


def score(sessions: list[dict]) -> dict:
    n = len(sessions)
    hit = sum(s["hit"] for s in sessions) / n
    mrr = statistics.fmean(s["reciprocal_rank"] for s in sessions)
    mttc = statistics.fmean(s["first_hit_turn"] or (MAX_TURNS + 1) for s in sessions)
    eff = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {
        "hit": hit, "mrr": mrr, "mttc": mttc, "eff": eff,
        "score": 0.50 * hit + 0.30 * mrr + 0.20 * eff,
        "tok_cov": statistics.fmean(s["token_coverage"] for s in sessions),
    }


def build_customer(kind: str, level: str, sample, card, behavior, categories,
                   target, index_products):
    seed = f"{sample.get('sample_id','')}\0{kind}\0{level}"
    rng = random.Random(seed)
    common = dict(sample=sample, card=card, behavior=behavior,
                  categories=categories, target=target, rng=rng)
    if kind == "official":
        return OfficialCustomer(**common)
    if kind == "paraphrase":
        return ParaphraseCustomer(level=level, **common)
    if kind == "decoy":
        return DecoyOverrideCustomer(index_products=index_products, **common)
    raise ValueError(kind)


def run_all(agent, samples, catalog_ids, categories, products, kind, level):
    out = []
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        cust = build_customer(kind, level, sample, card, behavior, categories,
                              target, agent.index.products)
        out.append(run_session(agent, sample, cust, catalog_ids, target))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--dataset", default="data/public_set.jsonl")
    ap.add_argument("--customer", default="official",
                    choices=["official", "paraphrase", "decoy"])
    ap.add_argument("--level", default="medium", choices=["light", "medium"])
    ap.add_argument("--all", action="store_true", help="run every customer variant")
    ap.add_argument("--verify", action="store_true",
                    help="assert the 'official' customer reproduces local_evaluator")
    args = ap.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)

    if args.verify:
        from evaluator.local_evaluator import evaluate
        ref = evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)
        mine = score(run_all(agent, samples, catalog_ids, categories, products, "official", "light"))
        d = abs(mine["score"] - ref["recommended_technical_score"])
        print(f"verify: harness official = {mine['score']:.5f}   "
              f"evaluator = {ref['recommended_technical_score']:.5f}   |delta| = {d:.5f}")
        print("  OK" if d < 1e-4 else "  MISMATCH - harness does not faithfully replay the simulator")
        return

    runs = ([("official", "-"), ("paraphrase", "light"), ("paraphrase", "medium"), ("decoy", "-")]
            if args.all else [(args.customer, args.level)])

    print(f"dataset={args.dataset}  sessions={len(samples)}\n")
    hdr = f"{'customer':<20} {'hit@10':>7} {'mrr':>8} {'mttc':>6} {'score':>8} {'tok_cov':>8}"
    print(hdr); print("-" * len(hdr))
    base = None
    last_sessions: list[dict] = []
    for kind, level in runs:
        DecoyOverrideCustomer.injected = DecoyOverrideCustomer.eligible = 0
        last_sessions = run_all(agent, samples, catalog_ids, categories, products, kind, level)
        s = score(last_sessions)
        name = kind if level == "-" else f"{kind}:{level}"
        delta = "" if base is None else f"  ({s['score'] - base:+.4f})"
        if base is None:
            base = s["score"]
        note = ""
        if kind == "decoy":
            note = f"  [decoys injected {DecoyOverrideCustomer.injected}/{DecoyOverrideCustomer.eligible} override sessions]"
        print(f"{name:<20} {s['hit']:>7.3f} {s['mrr']:>8.4f} {s['mttc']:>6.2f} "
              f"{s['score']:>8.5f} {s['tok_cov']:>8.3f}{delta}{note}")

    # per-scenario for the last customer run
    by: dict[str, list[dict]] = {}
    for x in last_sessions:
        by.setdefault(x["scenario_type"], []).append(x)
    print(f"\n{runs[-1][0]} per scenario:")
    for sc in sorted(by):
        s = score(by[sc])
        print(f"  {sc:<18} hit={s['hit']:.3f} mrr={s['mrr']:.4f} score={s['score']:.4f} (n={len(by[sc])})")


if __name__ == "__main__":
    main()
