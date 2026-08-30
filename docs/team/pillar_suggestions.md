# Pillar Suggestions — Where to Invest Now That the Metric Is Saturated

We ace Pillar IV (public: **1.000 Hit@10 / 0.9305 TechnicalScore**, `runs/public-20260830-132349`).
This document proposes what to build for the other three pillars, ranked by what it
actually moves in judging.

---

## 1. The scoring reality

Judging is **Technical Execution 35% / Innovation 20% / Impact 20% / Feasibility 15% /
Presentation 10%** ([future_steps.md](future_steps.md) §1). The automated TechnicalScore
(`0.50·Hit@10 + 0.30·MRR + 0.20·Efficiency`) is only *an input* to the Technical
Execution criterion — at most ~35% of one of five criteria. Pillars I–III are listed in
the spec under "Innovation Directions": they are pitched to human judges, and they map
onto **Innovation + Impact + Presentation ≈ 50% of the total**. That is where our
marginal point is, not in the last 0.005 of public score.

Current standing (all 2026-08-30):

| Set | N | Hit@10 | MRR | MTTC | Score | Latency |
|---|---|---|---|---|---|---|
| public | 200 | 1.000 | 0.901 | 3.00 | **0.9305** | 31 ms |
| hard | 96 | 0.896 | 0.720 | 4.09 | **0.8020** | 68 ms |
| generated adversarial | 200 | 0.930 | 0.808 | 3.68 | **0.8537** | 66 ms |

**House rule (unchanged):** every suggestion below ships as an **opt-in layer** that
must not regress the public score, and proves its value on the hard/adversarial sets
or in the demo/report. Dev selects, holdout gates.

Legend per suggestion: *(criterion it moves · effort S/M/L · public-score risk)*.

---

## 2. Pillar I — Intent Routing & Hybrid Pipeline

**Current state:** `src/router.py` has a genuinely rich classifier (`classify()`,
`detect_turn_intent()`, cue + facet-density scoring, confidence, scenario hints) — but
its only consumer is `src/phrasing.py`, which prepends a tone string. Retrieval
(`src/retrieval.py`) is three BM25 routes (anchor / full-text / focused) over one FTS5
index, fused with weighted RRF. No dense route, no category route; the cross-encoder
reranker was built, measured, and removed (`e44ca2e`, preserved on branch
`semantic-rerank`).

**Gap:** we claim "dual-track routing" and "hybrid pipeline", but the track only
changes wording and the pipeline is single-family (lexical). A judge reading
`starter/agent.py:166` will see the route computed and then not used to route.

### Suggestions

1. **Wire the dual track for real** *(Innovation, Presentation · S · none — flag-gated)*
   Consume `Route.suggested_first_recommend_turn` and buying/browsing in
   `starter/agent.py:_shortlist()`: buying → earlier first list, tighter ramp,
   confirm-style question; browsing → wider pool, later first list. Routing retrieval
   by intent was measured flat before (holdout −0.002), so keep it off by default —
   but the *timing* lever was never the thing tested, and even score-flat it makes the
   architecture diagram honest. A/B row in `tools/sweep.py`.

2. **Dense fallback route, trigger-gated** *(Innovation, Impact · M · none)*
   The hard set has 6 `never_retrieved` sessions — pure recall failures BM25 cannot
   fix. Add a dense route (bge-small, or a stdlib TF-IDF/char-n-gram cosine if we must
   stay dependency-free — final scoring may be network-restricted,
   `docs/submission_rules.md`) that fires **only when the BM25 pool looks weak**
   (low top score, thin pool). It never touches public sessions, so public recall
   (200/200) is untouchable, and any hard-set recovery is a clean Impact story:
   "hybrid where hybrid helps, lexical where lexical wins."

3. **Category-constrained route for the buying track** *(Innovation · S · none)*
   When the router detects hard constraints, add a category-filtered BM25 route into
   the RRF fusion. Category is currently rerank-only (`_category_match`, `_tail_match`);
   moving it into retrieval is the "heterogeneous retrieval routing" the problem
   statement names, and RRF makes it additive rather than destructive.

4. **Tell the semantic-rerank story as a strength** *(Presentation, Feasibility · S · none)*
   We built a cross-encoder, measured it (dev 0.9268 → 0.9211 at ~13× latency),
   established the oracle ceiling (+0.043 public / +0.084 hard), and deleted it with
   the numbers preserved (`docs/team/rerank_signals.md` §9). In the report this is not
   a missing feature — it is evidence-driven engineering. Optionally: LLM/cross-encoder
   rerank of the top ~20 **only** when span coverage saturates (homogeneous clusters),
   the one regime where it measurably helps.

---

## 3. Pillar II — Dialog Strategy

**Current state:** `src/state.py` is the strongest module — provenance-tracked
utterances, decline → `dead_attributes`, override → down-weight (not erase, which we
measured as strictly better), productive-turn tracking. This pillar produced the
0.11 → 0.78 jump. `src/phrasing.py` already computes a real over-generality signal
(facet entropy over the top-40 pool) but uses it only for prose;
`ask_attribute` is always `"other"` (`FixedPolicy`). `InfoGainPolicy` exists but loses
on holdout.

**Gap:** no *named* state machine, and "proactive retrieval cutoff on over-generality"
— explicitly demanded by the problem statement — does not exist as behavior.

### Suggestions

1. **True over-generality cutoff** *(Innovation, Impact · M · low — flag-gated)*
   When `_grounded()` finds a facet with high gain-ratio over an undifferentiated pool,
   actually **act**: set `ask_attribute` to that facet (not `"other"`) and, when the
   pool is huge and flat, hold the list for one turn and ask first. On the public
   simulator `"other"` dominates (it always answers), so gate it and measure on
   browsing + adversarial sessions where MTTC upside lives. This is the one Pillar-II
   feature with a plausible score story *and* a demo story.

2. **Name the state machine** *(Presentation · S · zero)*
   A thin `SessionPhase` enum — `EXPLORE → NARROW → OVERRIDE_RECOVERY → CONVERGED` —
   derived from flags that already exist (`override_turn`, `productive_turns`,
   pool confidence). Zero behavior change, but it turns "a dataclass with flags" into
   the Dynamic State Machine the problem statement asks for, gives the report a
   diagram, and is the substrate Pillar III hangs off.

3. **Hybrid question policy** *(Innovation, demo realism · S · none)*
   `FixedPolicy` for the first ask (drains the simulator's freebie constraints), then
   `InfoGainPolicy` for targeted follow-ups. Known to be score-flat on this simulator
   ([future_steps.md](future_steps.md) §2) — build it as the opt-in realism layer and the private-set
   hedge if that simulator discloses less per ask.

4. **Wire `detect_turn_intent()`** *(Innovation · S · none — flag-gated)*
   Browsing→buying promotion mid-session currently computes and discards. Feed it into
   shortlist timing (promote → recommend now). Completes the "Intent Override /
   evolution" story with code, not prose.

---

## 4. Pillar III — Self-Evolution (weakest pillar, highest marginal judging value)

**Current state:** near-nothing on the live path. `user_profile` is stored but read
only by the non-default `InfoGainPolicy`; the profile rerank signal was tried four
times and failed holdout every time (`src/rerank.py:43-47`). Every turn runs the same
fixed pipeline. But the *inputs* for adaptation already exist and go unconsumed:
`productive_turns`, `last_turn_productive`, `dead_attributes`, pool confidence,
`detect_turn_intent`.

**Gap:** the entire pillar. This is where a judge will find the least, so each unit of
work here buys the most differentiation.

### Suggestions

1. **Failure detection + strategy switching** *(Innovation, Impact · M · low — flag-gated)*
   The flagship build. A per-turn orchestrator: if the last N turns were unproductive
   and the pool is stagnant → switch strategy (widen retrieval pool, force the
   elimination scan, diversify the list, change question type). "Failure detection and
   strategy switching" is a *named* Innovation Direction in the spec, and
   `IMPLEMENTATION.md` §10 already sketches it. This is "runtime workflow
   re-orchestration" in exactly the problem statement's words, driven by signals we
   already compute.

2. **Context distillation as a retrieval view** *(Innovation · M · low — flag-gated)*
   Today the query context is raw concatenated text. Add a distilled view: typed
   facets from `FacetStore` (`src/facets.py`) compiled into a structured query, as a
   fourth RRF route or a rerank signal. On the public set verbatim text wins (the
   simulator quotes catalog text); on paraphrased/adversarial sessions the typed view
   is the robustness layer — which is precisely "Personalized Context Distillation"
   with a measurable demo.

3. **Safe personalization, honestly scoped** *(Presentation · S · none)*
   Either ship the profile tag prior as a tie-break-only epsilon weight (visible in
   explanations: "you tend to buy leather — leading with leather options"), or present
   the four measured failures as a finding: only 26.5% of targets satisfy all their
   profile tags, so aggressive personalization is *wrong* on this data. Both versions
   are a strong report section; the second costs nothing.

4. **Per-turn budget as a designed property** *(Feasibility · S · none)*
   31 ms/turn, $0, offline, deterministic. Frame the orchestrator (III-1) as also
   budget-aware — skip stages that cannot contribute. Feasibility is 15% of judging
   and we win it by default; say so explicitly.

---

## 5. Pillar IV — Evaluation (already strong: protect and polish)

**Current state:** three-layer harness (frozen official evaluator; `tools/sweep.py`
dev/holdout A/B; `tools/observe.py` tracer with failure taxonomy), adversarial
generators, weight-fitting with holdout as gate, regression tests pinning the floor.

### Suggestions

1. **Fix stale numbers** *(Presentation, credibility · S · zero)*
   README Disclosure and `IMPLEMENTATION.md` §7 still say 0.8592 / 16.6 ms;
   `agent_changes.md` still lists `popularity_weight 0.02` (now 0.4); root
   `results.json` is a hard-set run that reads like the headline score. A judge who
   spots one wrong number discounts every right one.

2. **Extend the realism harness** *(Problem Insight, Feasibility · M · none)*
   Per [future_steps.md](future_steps.md) §3: sessions the official simulator cannot produce —
   paraphrased disclosure, genuine decoys on override, partial disclosure,
   "I'm not sure" customers. Every Pillar I–III opt-in above gets its proof here.

3. **Put the harness in the report** *(Technical Execution, Presentation · S · zero)*
   Failure taxonomy (`never_retrieved` vs `ranked_out`), dev-selects/holdout-gates
   discipline, adversarial buckets, negative results kept with their numbers. Our
   evaluation culture is the strongest single differentiator we own — make it a
   first-class section, including "what the public simulator can and can't measure."

---

## 6. Priority — recommended order

| # | Suggestion | Pillar | Criterion moved | Effort | Score risk |
|---|---|---|---|---|---|
| 1 | Strategy switching orchestrator | III-1 | Innovation + Impact | M | low (flag) |
| 2 | Over-generality cutoff | II-1 | Innovation + Impact | M | low (flag) |
| 3 | Wire dual-track timing | I-1 | Innovation + Presentation | S | none (flag) |
| 4 | Named state machine | II-2 | Presentation | S | zero |
| 5 | Stale-number doc refresh | IV-1 | Credibility | S | zero |
| 6 | Dense fallback route | I-2 | Innovation + Impact | M | none (gated) |
| 7 | Context-distillation view | III-2 | Innovation | M | low (flag) |
| 8 | Realism harness extension | IV-2 | Problem Insight | M | none |
| 9 | Hybrid question policy | II-3 | Realism / hedge | S | none |
| 10 | Personalization write-up | III-3 | Presentation | S | none |

Items 1–5 are the sprint: two M-sized features that create Pillars III and II behavior
that does not exist today, plus three S-sized items that make everything we already
built *visible*. Items 6–8 are the private-set insurance. Everything ships behind
flags; the 0.9305 floor is never at stake.
