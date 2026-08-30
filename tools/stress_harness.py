"""Stress harness - evaluate the agent against customers the official simulator
cannot produce, and see *why* it fails when it does.

`evaluator/local_evaluator.py` is a fully-cooperative, templated, deterministic
customer: it always answers, discloses constraints copied **verbatim** from the
target's own metadata, drains every constraint on `ask_attribute="other"`, and
its "intent override" never retracts anything real. So it cannot measure Pillar I
routing, Pillar II proactive guidance / genuine override, or Pillar III
adaptation - there is no difficult customer to handle, and `FixedPolicy("other")`
is unbeatable.

This harness drives the **unmodified** `Agent` through a faithful copy of
`evaluate()`'s loop with composable customer stressors:

  paraphrase:light   - same constraints, verbatim tokens kept, carrier sentence
                       reworded ("For that, what matters is: X" -> "It should be X.")
  paraphrase:medium  - the constraint itself reworded ("color: blue" -> "in blue")
  paraphrase:heavy   - medium + broad synonym substitution (leather -> genuine
                       hide, waterproof -> water-repellent, gym -> working out ...)
                       + clause shuffle/fusion + spoken filler. Rule-based; an
                       offline LLM rewriter would be the real "heavy".
  browse-gated       - the *browsing* customer discloses a constraint only when
                       asked a pointed question whose classify_constraint bucket
                       matches - never on the broad "anything else?". Makes
                       Buying/Browsing routing load-bearing.
  decoy              - intent_override sessions where the pre-override preference
                       is a GENUINE decoy (a facet value the target lacks).

Stressors compose: `--customer paraphrase:medium+browse-gated` is a vague browser
who also rewords everything - the closest thing to the feared private simulator.

It also reports, per scenario, whether a failure was **retrieval** (target never
entered the 300-pool, or sits deep in it) or **ranking** (in the pool, never
surfaced) - so "route retrieval by track" claims can be checked against evidence.

Nothing here modifies the agent or the evaluator. `--verify` asserts the
un-stressed path reproduces `python3 -m evaluator.local_evaluator` (delta 0).
Reported numbers are a robustness probe, NOT the official score.

    python3 tools/stress_harness.py --verify
    python3 tools/stress_harness.py --all
    python3 tools/stress_harness.py --customer paraphrase:medium+browse-gated
    python3 tools/stress_harness.py --customer browse-gated --configs router_off,router_on
    python3 tools/stress_harness.py --customer browse-gated --misroute-matrix
    python3 tools/stress_harness.py --all --targets generic     # hard-to-retrieve targets only
"""

from __future__ import annotations

import argparse
import random
import re
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
from src.index import DEFAULT_WEIGHTS  # noqa: E402
from src.rerank import rerank  # noqa: E402
from src.retrieval import retrieve  # noqa: E402
from src.text import terms  # noqa: E402
from starter import agent as agent_module  # noqa: E402
from starter.agent import Agent  # noqa: E402
from src.router import BROWSING, BUYING  # noqa: E402


# --------------------------------------------------------------------------------
# Paraphrasing (rule-based, deterministic per session)
# --------------------------------------------------------------------------------

_LEADINS = [
    "I'd also want it to be {}.", "It should be {}.", "Also important: {}.",
    "One more thing - {}.", "{} matters to me as well.", "Ideally it's {}.",
]
#: Looser, more spoken frames, used only at level=heavy.
_LEADINS_HEAVY = _LEADINS + [
    "honestly I just want {}.", "the main thing is {}.", "oh and {} if possible.",
    "gotta be {} for me.", "leaning towards {}.", "something {} would be great.",
]
_HYPHEN = {
    "stainless steel": "stainless-steel", "full grain": "full-grain",
    "long sleeve": "long-sleeved", "short sleeve": "short-sleeved",
    "high waisted": "high-waisted", "slim fit": "slim-fit",
    "water resistant": "water-resistant",
}

#: Whole-phrase synonyms for level=heavy. Applied by longest key first so
#: "full grain leather" wins over "leather". A human reads the substitute as the
#: same requirement; none of them is verbatim catalog text.
_SYNONYMS: dict[str, list[str]] = {
    # materials - at least one option per entry departs from the key token, so
    # heavy erodes FTS5 recall and not only the verbatim-span signal.
    "full grain leather": ["full-grain hide", "top-grain cowhide"],
    "genuine leather": ["real hide", "actual cowhide"],
    "leather": ["genuine hide", "cowhide", "a tanned-hide build"],
    "suede": ["brushed nap", "a soft napped finish"],
    "cotton": ["a pure natural-fibre weave", "an all-cotton knit", "combed 100 percent cotton"],
    "polyester": ["a synthetic blend", "man-made fibre"],
    "nylon": ["a ripstop synthetic", "a technical shell fabric"],
    "denim": ["jean material", "a sturdy indigo twill"],
    "wool": ["merino knit", "a woollen weave"],
    "cashmere": ["a soft luxury knit", "fine goat-hair yarn"],
    "stainless steel": ["surgical-grade metal", "a brushed silver alloy"],
    "sterling silver": ["925 fine metal", "solid precious metal"],
    "rubber": ["a flexible compound", "moulded gum"],
    "canvas": ["heavy woven cotton", "a duck-cloth upper"],
    "mesh": ["an open knit", "a breathable net weave"],
    # colours
    "black": ["jet black", "matte black", "a dark black"],
    "white": ["off-white", "a clean white", "bright white"],
    "grey": ["gray", "charcoal", "a heather grey"],
    "gray": ["grey", "charcoal", "slate"],
    "blue": ["a mid blue", "royal blue"],
    "navy": ["navy blue", "a deep navy"],
    "red": ["a true red", "crimson"],
    "green": ["forest green", "a muted green"],
    "brown": ["chocolate brown", "a tan brown"],
    "beige": ["tan", "sand", "a beige tone"],
    "pink": ["blush pink", "a soft pink"],
    "gold": ["a gold tone", "yellow gold"],
    "silver": ["a silver finish", "brushed silver"],
    # features / construction
    "waterproof": ["water-resistant", "water-repellent", "weatherproof"],
    "adjustable": ["a customizable fit", "resizable", "adjusts to fit"],
    "lightweight": ["light", "barely any weight", "featherweight"],
    "breathable": ["airy", "well-ventilated", "it breathes well"],
    "wireless": ["cordless", "no wires"],
    "buckle": ["clasp", "snap fastener"],
    "zipper": ["zip", "zip fastener"],
    "pockets": ["pouches", "storage compartments"],
    "insulated": ["thermal", "keeps warmth in"],
    "non slip": ["grippy", "a no-slip sole"],
    "quick dry": ["fast-drying", "dries quickly"],
    "hypoallergenic": ["skin-safe", "gentle on sensitive skin"],
    # use cases
    "hiking": ["trail use", "trekking", "the trails"],
    "running": ["jogging", "road running"],
    "workout": ["training", "the gym"],
    "gym": ["the gym", "working out"],
    "yoga": ["yoga practice", "a yoga class"],
    "travel": ["trips", "traveling", "the road"],
    "office": ["work", "the office", "day-to-day work"],
    "wedding": ["a formal event", "a black-tie thing"],
    "outdoor": ["being outside", "outdoors"],
    "winter": ["the cold months", "cold weather"],
    "summer": ["hot days", "the warmer months"],
}
_SYNONYM_KEYS = sorted(_SYNONYMS, key=len, reverse=True)


def _synonym_sub(text: str, rng: random.Random, rate: float = 0.75) -> str:
    """Replace recognised whole-word phrases with a spoken synonym."""
    for key in _SYNONYM_KEYS:
        if re.search(rf"\b{re.escape(key)}\b", text) and rng.random() < rate:
            text = re.sub(rf"\b{re.escape(key)}\b", rng.choice(_SYNONYMS[key]), text, count=1)
    return text


_FILLERS = ["", "", "something ", "ideally ", "a bit ", "really ", "kind of "]


def _reword_one(value: str, rng: random.Random) -> str:
    """Reword one constraint so it is no longer verbatim catalog text, while a
    human still reads it as the same requirement."""
    v = value.strip()
    low = v.lower()
    if low.startswith("color:"):
        col = low.split(":", 1)[1].strip()
        return rng.choice([f"in {col}", f"{col} coloured", f"the {col} one", f"{col} in colour"])
    if "budget around $" in low:
        amt = low.split("$", 1)[1].strip()
        return rng.choice([f"around ${amt}", f"roughly {amt} dollars",
                           f"my budget is about ${amt}", f"nothing over ${amt} really"])
    m = re.match(r"(\d+)%\s+(.+)", v)
    if m:
        pct, mat = m.group(1), m.group(2).lower()
        return rng.choice([f"all {mat}", f"{mat} through and through",
                           f"{pct} percent {mat}", f"mostly {mat}"])
    m = re.match(r"(.+?)\s+closure$", low)
    if m:
        kind = m.group(1)
        return rng.choice([f"a {kind} fastening", f"it does up with a {kind}", f"{kind}-fastened"])
    for phrase, hyph in _HYPHEN.items():
        if phrase in low:
            return low.replace(phrase, hyph)
    toks = v.split()
    if len(toks) >= 3 and rng.random() < 0.5:
        return " ".join(toks[1:] + toks[:1]).lower()
    return low


def _tidy(text: str) -> str:
    """Kill the compounding artefacts of stacked rewrites - a real messy customer
    is disfluent, not ungrammatical to the point of noise."""
    text = re.sub(r"\bit does up with a\b", "with a", text)
    text = re.sub(r"\b(a|an|the) (a|an|the)\b", r"\1", text)
    text = re.sub(r"\b(\w+) \1\b", r"\1", text)          # "fastening fastening"
    return re.sub(r"\s+", " ", text).strip()


def _reword_heavy(value: str, rng: random.Random) -> str:
    """medium's pattern rewrites, then broad synonym substitution, then a filler."""
    v = _synonym_sub(_reword_one(value, rng), rng)
    if rng.random() < 0.4:
        v = rng.choice(_FILLERS) + v
    return _tidy(v)


def paraphrase_disclosure(matches: list[str], level: str, rng: random.Random) -> str:
    if level == "light":
        return " ".join(rng.choice(_LEADINS).format(m) for m in matches)
    if level == "medium":
        return " ".join(rng.choice(_LEADINS).format(_reword_one(m, rng)) for m in matches)
    # heavy: reword + synonyms, shuffle the clauses, and often fuse them into one
    # reordered sentence instead of one crisp statement per constraint.
    pieces = [_reword_heavy(m, rng) for m in matches]
    rng.shuffle(pieces)
    if len(pieces) > 1 and rng.random() < 0.6:
        body = rng.choice([", and ", ", plus ", " - also ", ", and honestly "]).join(pieces)
        out = rng.choice(["I'm after something {}.", "Ideally {}.",
                          "What I care about: {}.", "Looking for {} really.",
                          "So, {} - that's the gist."]).format(body)
    else:
        out = " ".join(rng.choice(_LEADINS_HEAVY).format(p) for p in pieces)
    return _tidy(out)


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

    def opening(self) -> str:
        if self.scenario == "buying" and self.card.get("hard_constraints"):
            c = str(self.card["hard_constraints"][0])
            self.disclosed.add(c)
            return f"I'm looking for {self.category}. A key requirement is: {c}."
        if self.scenario == "intent_override":
            return f"I'm looking for {self.category}. {self.behavior['override']['old_value']}"
        return f"I'm looking for {self.category}, but I'm still exploring."

    def reply(self, turn: int, ask_attribute: object) -> str:
        if not self.override_applied and turn + 1 == self.override_turn:
            self.override_applied = True
            nv = str(self.behavior["override"].get("new_value", ""))
            if nv:
                self.disclosed.add(nv)
            return str(self.behavior["override"]["message"])
        return self._disclose(ask_attribute)

    def _constraints(self) -> list[str]:
        return [
            *[str(v) for v in self.card.get("hard_constraints", [])],
            *[str(v) for v in self.card.get("soft_preferences", [])],
        ]

    def _match(self, attribute: str, budget: int) -> list[str]:
        return [
            v for v in self._constraints()
            if v not in self.disclosed
            and (attribute == "other" or classify_constraint(v) == attribute)
        ][:budget]

    def _disclose(self, ask_attribute: object) -> str:
        attribute = ask_attribute if isinstance(ask_attribute, str) else None
        if self.scenario == "boundary" and not self.boundary_used and attribute:
            self.boundary_used = True
            return f"I don't have a preference for {attribute}; please use your judgment."
        if not attribute:
            return "Those options are not quite right yet. Ask me about one specific attribute."
        if attribute not in ALLOWED_ATTRIBUTES:
            attribute = "other"
        matches = self._match(attribute, 2)
        if not matches:
            return f"I don't have an additional preference for {attribute}."
        self.disclosed.update(matches)
        return self._render(matches)

    def _render(self, matches: list[str]) -> str:
        return "For that, what matters is: " + "; ".join(matches) + "."


def _decoy_value(product: dict) -> str | None:
    """A facet value the target does not have and whose token is absent from its
    text - so an override away from it is a real retraction."""
    facets = extract(product)
    text = product.get("text", "")
    for attr in ("color", "material"):
        have = facets.get(attr)
        if not have:
            continue
        for cand in VOCABULARIES[attr]:
            if cand != have and cand not in text:
                return f"color: {cand}" if attr == "color" else cand
    return None


class StressCustomer(Customer):
    """The base customer plus any combination of independent stressors."""

    decoys_injected = 0
    decoys_eligible = 0

    def __init__(self, *a, paraphrase: str = "", browse_gated: bool = False,
                 decoy: bool = False, index_products: dict | None = None,
                 target: str = "", **kw):
        super().__init__(*a, target=target, **kw)
        self.paraphrase = paraphrase          # "" | "light" | "medium"
        self.browse_gated = browse_gated
        if decoy and self.scenario == "intent_override":
            StressCustomer.decoys_eligible += 1
            prod = (index_products or {}).get(target)
            d = _decoy_value(prod) if prod else None
            if d:
                StressCustomer.decoys_injected += 1
                self.behavior = {**self.behavior,
                                 "override": {**self.behavior["override"], "old_value": d}}
                self.sample["behavior"] = self.behavior

    def opening(self) -> str:
        text = super().opening()
        if self.paraphrase and "A key requirement is:" in text:
            head, c = text.split("A key requirement is:", 1)
            c = c.strip().rstrip(".")
            reworded = {"light": c, "medium": _reword_one(c, self.rng)}.get(
                self.paraphrase) or _reword_heavy(c, self.rng)
            return f"{head.strip()} I really need it to be {reworded}."
        return text

    def _disclose(self, ask_attribute: object) -> str:
        if self.browse_gated and self.scenario == "browsing":
            attribute = ask_attribute if isinstance(ask_attribute, str) else None
            if attribute is None or attribute == "other":
                return ("I'm still just browsing - ask me about one particular thing "
                        "and I'll tell you.")
            if attribute not in ALLOWED_ATTRIBUTES:
                return "I don't have an additional preference for that."
            matches = self._match(attribute, 1)          # a pointed ask -> one constraint
            if not matches:
                return f"I don't have an additional preference for {attribute}."
            self.disclosed.update(matches)
            return self._render(matches)
        return super()._disclose(ask_attribute)

    def _render(self, matches: list[str]) -> str:
        if self.paraphrase:
            return paraphrase_disclosure(matches, self.paraphrase, self.rng)
        return super()._render(matches)


def parse_spec(spec: str) -> dict:
    """'paraphrase:medium+browse-gated' -> {paraphrase:'medium', browse_gated:True}."""
    out = {"paraphrase": "", "browse_gated": False, "decoy": False}
    for part in spec.split("+"):
        part = part.strip()
        if not part or part == "official":
            continue
        if part.startswith("paraphrase"):
            level = part.split(":", 1)[1] if ":" in part else "medium"
            if level not in ("light", "medium", "heavy"):
                raise ValueError(f"paraphrase level must be light|medium|heavy, got {level!r}")
            out["paraphrase"] = level
        elif part == "browse-gated":
            out["browse_gated"] = True
        elif part == "decoy":
            out["decoy"] = True
        else:
            raise ValueError(f"unknown stressor: {part}")
    return out


# --------------------------------------------------------------------------------
# Session runner (faithful copy of evaluate()'s loop) + retrieval diagnostic
# --------------------------------------------------------------------------------

def _target_ranks(agent: Agent, sid: str, target: str) -> tuple[int | None, int | None]:
    """Where the target sits in the final retrieval pool, and after a plain
    rerank. Recomputed from the accumulated state - a read-only probe."""
    state = agent._states.get(sid)
    if state is None:
        return None, None
    pool = retrieve(agent.index, state, agent.config.retrieval)
    pool_ids = [a for a, _ in pool]
    pool_rank = pool_ids.index(target) + 1 if target in pool_ids else None
    ranked = rerank(agent.index, state, pool, agent.config.rerank)
    ranked_ids = [a for a, _ in ranked]
    ranked_rank = ranked_ids.index(target) + 1 if target in ranked_ids else None
    return pool_rank, ranked_rank


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

    gt_tokens: set[str] = set()
    for c in customer._constraints():
        gt_tokens |= set(terms(str(c)))
    state = agent._states.get(sid)
    seen = set(terms(state.full_text())) if state is not None else set()
    coverage = len(gt_tokens & seen) / len(gt_tokens) if gt_tokens else 1.0
    pool_rank, ranked_rank = _target_ranks(agent, sid, target)

    return {
        "scenario_type": sample["scenario_type"],
        "hit": hit_turn is not None,
        "first_hit_turn": hit_turn,
        "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        "token_coverage": coverage,
        "pool_rank": pool_rank,       # None = never retrieved
        "ranked_rank": ranked_rank,
    }


def score(sessions: list[dict]) -> dict:
    n = len(sessions)
    hit = sum(s["hit"] for s in sessions) / n
    mrr = statistics.fmean(s["reciprocal_rank"] for s in sessions)
    mttc = statistics.fmean(s["first_hit_turn"] or (MAX_TURNS + 1) for s in sessions)
    eff = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {
        "hit": hit, "mrr": mrr, "mttc": mttc,
        "score": 0.50 * hit + 0.30 * mrr + 0.20 * eff,
        "tok_cov": statistics.fmean(s["token_coverage"] for s in sessions),
    }


def retrieval_diag(sessions: list[dict]) -> str:
    n = len(sessions)
    never = sum(1 for s in sessions if s["pool_rank"] is None)
    got = [s["pool_rank"] for s in sessions if s["pool_rank"] is not None]
    deep = sum(1 for r in got if r > 100)
    ranked_out = sum(1 for s in sessions
                     if s["pool_rank"] is not None and not s["hit"]
                     and (s["ranked_rank"] or 999) > TOP_K)
    med = int(statistics.median(got)) if got else None
    return (f"never_retrieved {never}/{n}  pool_rank>100 {deep}/{n}  "
            f"median_pool_rank {med}  ranked_out {ranked_out}/{n}")


# --------------------------------------------------------------------------------
# Target selection - the hard-to-retrieve subset
# --------------------------------------------------------------------------------

def _phrase_df(index, phrase: str) -> int:
    """How many catalog products contain this constraint span as a phrase."""
    toks = terms(phrase)
    if not toks:
        return 10 ** 9
    rows = index._match('"' + " ".join(toks) + '"', 5000, DEFAULT_WEIGHTS)
    return len(rows)


def select_generic(samples, products, index, threshold: int = 400) -> list[dict]:
    """Keep samples whose disclosed constraint spans are ALL high-frequency in the
    catalog - BM25 cannot separate the target on any of them, so it lands deep in
    the pool and retrieval, not ranking, is on the hook."""
    out = []
    for s in samples:
        card, _ = materialize_hidden_fields(s, products)
        spans = [str(v) for v in card.get("hard_constraints", [])] + \
                [str(v) for v in card.get("soft_preferences", [])]
        if spans and all(_phrase_df(index, sp) >= threshold for sp in spans):
            out.append(s)
    return out


# --------------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------------

def build_customer(spec: dict, sample, card, behavior, categories, target, index_products):
    seed = f"{sample.get('sample_id','')}\0{spec}"
    rng = random.Random(seed)
    return StressCustomer(
        sample=sample, card=card, behavior=behavior, categories=categories,
        target=target, rng=rng, index_products=index_products, **spec,
    )


def run_all(agent, samples, catalog_ids, categories, products, spec: dict) -> list[dict]:
    StressCustomer.decoys_injected = StressCustomer.decoys_eligible = 0
    out = []
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        cust = build_customer(spec, sample, card, behavior, categories, target,
                              agent.index.products)
        out.append(run_session(agent, sample, cust, catalog_ids, target))
    return out


def _print_run(name, sessions, base):
    s = score(sessions)
    delta = "" if base is None else f"  ({s['score'] - base:+.4f})"
    print(f"{name:<26} {s['hit']:>6.3f} {s['mrr']:>7.4f} {s['mttc']:>6.2f} "
          f"{s['score']:>8.5f} {s['tok_cov']:>7.3f}{delta}")
    return s["score"]


def _print_scenarios(sessions):
    by: dict[str, list[dict]] = {}
    for x in sessions:
        by.setdefault(x["scenario_type"], []).append(x)
    for sc in sorted(by):
        s = score(by[sc])
        print(f"  {sc:<16} hit={s['hit']:.3f} mrr={s['mrr']:.4f} "
              f"score={s['score']:.4f}  [{retrieval_diag(by[sc])}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--dataset", default="data/public_set.jsonl")
    ap.add_argument("--customer", default="official",
                    help="stressor spec, e.g. 'paraphrase:medium+browse-gated'")
    ap.add_argument("--all", action="store_true", help="a curated matrix of stressors")
    ap.add_argument("--targets", default="all", choices=["all", "generic"],
                    help="'generic' = only hard-to-retrieve targets")
    ap.add_argument("--configs", default="", help="comma-separated tools/sweep.py config names")
    ap.add_argument("--misroute-matrix", action="store_true",
                    help="force each track, tabulate true x routed (needs a browse-gated spec)")
    ap.add_argument("--verify", action="store_true",
                    help="assert the un-stressed path reproduces local_evaluator")
    ap.add_argument("--limit", type=int, help="first N sessions after --targets filtering "
                    "(bounds wall time / token spend on a real --configs llm_rerank run)")
    args = ap.parse_args()

    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)

    if args.targets == "generic":
        samples = select_generic(samples, products, agent.index)
        print(f"generic-target subset: {len(samples)} sessions\n")

    if args.verify:
        from evaluator.local_evaluator import evaluate
        ref = evaluate(Agent(args.catalog), load_jsonl(args.dataset), catalog_ids, categories, products)
        mine = score(run_all(agent, load_jsonl(args.dataset), catalog_ids, categories, products,
                             parse_spec("official")))
        d = abs(mine["score"] - ref["recommended_technical_score"])
        print(f"harness official = {mine['score']:.6f}   evaluator = "
              f"{ref['recommended_technical_score']:.6f}   |delta| = {d:.2e}")
        print("PASS" if d < 1e-4 else "FAIL")
        sys.exit(0 if d < 1e-4 else 1)

    hdr = f"{'customer':<26} {'hit@10':>6} {'mrr':>7} {'mttc':>6} {'score':>8} {'tok_cov':>7}"

    if args.misroute_matrix:
        spec = parse_spec(args.customer if args.customer != "official" else "browse-gated")
        print(f"dataset={args.dataset}  spec={spec}\n")
        for forced in ("buying", "browsing"):
            route = BUYING if forced == "buying" else BROWSING
            saved_c, saved_d = agent_module.classify, agent_module.detect_turn_intent
            agent_module.classify = lambda _o, _r=route: _r
            agent_module.detect_turn_intent = lambda *a, _r=route, **k: _r
            try:
                sessions = run_all(agent, samples, catalog_ids, categories, products, spec)
            finally:
                agent_module.classify, agent_module.detect_turn_intent = saved_c, saved_d
            by: dict[str, list[dict]] = {}
            for x in sessions:
                by.setdefault(x["scenario_type"], []).append(x)
            row = "  ".join(
                f"true-{sc}={score(by[sc])['hit']:.2f}/{score(by[sc])['mrr']:.2f}"
                for sc in ("buying", "browsing") if sc in by)
            print(f"routed as {forced:<9}: {row}")
        return

    if args.configs:
        from tools.sweep import build_configs
        configs = build_configs(args.catalog)
        spec = parse_spec(args.customer)
        print(f"dataset={args.dataset}  spec={spec}\n{hdr}\n" + "-" * len(hdr))
        base = None
        last: list[dict] = []
        for name in [c.strip() for c in args.configs.split(",") if c.strip()]:
            a = Agent(args.catalog, configs[name])
            last = run_all(a, samples, catalog_ids, categories, products, spec)
            sc = _print_run(name, last, base)
            if base is None:
                base = sc
        print()
        _print_scenarios(last)
        return

    _MATRIX = ["official", "paraphrase:light", "paraphrase:medium", "paraphrase:heavy",
               "browse-gated", "paraphrase:medium+browse-gated",
               "paraphrase:heavy+browse-gated", "decoy"]
    runs = ([(n, parse_spec(n)) for n in _MATRIX] if args.all
            else [(args.customer, parse_spec(args.customer))])

    print(f"dataset={args.dataset}  sessions={len(samples)}\n{hdr}\n" + "-" * len(hdr))
    base = None
    last: list[dict] = []
    for name, spec in runs:
        last = run_all(agent, samples, catalog_ids, categories, products, spec)
        sc = _print_run(name, last, base)
        if base is None:
            base = sc
        if spec["decoy"]:
            print(f"   [decoys injected {StressCustomer.decoys_injected}/"
                  f"{StressCustomer.decoys_eligible} override sessions]")
    print(f"\n{runs[-1][0]} per scenario:")
    _print_scenarios(last)


if __name__ == "__main__":
    main()
