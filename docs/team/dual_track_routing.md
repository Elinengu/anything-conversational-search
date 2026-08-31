# Dual-track routing — making it real, and measuring it

Branch: `dual_tracking` (cut from `main` at `dd9ba8a`). `main` is untouched.

## 1. The problem

The spec names "Buying versus Browsing routing" as Pillar I. Until this branch the
router (`src/router.py:classify`) produced a `Route` that reached exactly one
consumer — `src/phrasing.py:clarify`, where it swaps a lead-in phrase. Retrieval,
rerank, `ask_attribute` and recommendation timing were all track-blind, and the
public simulator never reads `message`, so routing was score-neutral **by
construction** (`docs/team/future_steps.md:18`).

Worse, the public simulator *cannot* reward routing even in principle. In
`evaluator/local_evaluator.py`:

- `initial_message` (`:154`) pre-discloses `hard_constraints[0]` for a buyer and
  says "still exploring" for a browser — the only scenario difference.
- `customer_reply` (`:166`) is then **scenario-agnostic** (bar the one-shot
  `boundary` decline). On `ask_attribute="other"` — what `FixedPolicy` emits every
  turn — line 180 hands over *every* still-undisclosed constraint.

So a decided buyer and a vague browser are drained identically, and
`FixedPolicy("other")` is unbeatable. There is nothing for routing to do.

## 2. What the branch adds

### 2a. A realism harness — `tools/stress_harness.py --customer browse-gated`

Keeps the buying / intent_override / boundary customers exactly as the organizer
wrote them. Changes only the **browsing** customer: with no shopping list to
recite, they disclose a constraint only when the agent asks a *pointed* question
whose attribute matches (`classify_constraint`), never on a broad "anything
else?". Runs a faithful copy of `evaluate()`'s loop; `evaluator/` and `data/` are
never edited. `--verify` asserts bit-identical scoring against the official
evaluator (delta `3.6e-07`).

(The `browse-gated` stressor was originally `tools/dual_track_harness.py` on this
branch; it is now one composable customer in the merged `tools/stress_harness.py`
— see `docs/team/stress_harness.md`, which also adds paraphrase / decoy stressors
and the retrieval-vs-ranking diagnostic.)

### 2b. Track-aware behaviour — `AgentConfig.use_router`

`use_router` (already `True` by default) is widened from "route the phrasing" to
"route the behaviour". The track — buying or browsing, re-checked every turn by
`detect_turn_intent`, promoted one-way to buying once enough is disclosed or after
an override — now drives four levers:

| lever | buying | browsing |
|---|---|---|
| clarification policy (S4) | `FixedPolicy` — a decided customer recites everything on "anything else?" | `InfoGainPolicy` — asks the highest-gain attribute *once broad questions stop paying off* (it self-adapts, so on a cooperative customer it still tracks Fixed) |
| rerank weights (S6) | `buying_rerank` config, `None` ⇒ shared | `browsing_rerank` config, `None` ⇒ shared |
| constraint semantics (S6) | `hard_filter`: a candidate that positively contradicts an authoritative stated facet is banished to the bottom of the list | soft penalty only (`facet_conflict_weight`), unchanged |
| recommendation timing (S7) | `buying_first_recommend_turn` / `buying_list_size_ramp` | `browsing_*` equivalents |

The **policy** keys off how the session *opened* and stays there (InfoGain
self-adapts, so a browser who turns decisive keeps a policy that will still dig
for the constraints they have left). The promotable track drives the other three.

`use_router=False` bypasses all of it — scored output is bit-identical to the
pre-routing agent. It is kept as the measurement baseline and the guaranteed-safe
fallback (the `respond()` exception path is flat regardless).

Retrieval pool width is deliberately **not** a lever: BM25 recall at pool 300 is
~100%, and routing retrieval was measured flat / −0.002 and dropped long ago
(`IMPLEMENTATION.md:507-516`).

## 3. Results

### On the official (fully-cooperative) simulator — a small cost, no upside

| set | `use_router=False` | `use_router=True` | delta |
|---|---|---|---|
| public 200 (score) | 0.930502 | 0.917680 | −0.0128 |
| public Hit@10 | 200 / 200 | 200 / 200 | held |
| adversarial 96 (score) | 0.801978 | 0.799380 | −0.0026 (noise) |
| sweep dev 120 | 0.9418 | 0.9268 | −0.015 |
| sweep holdout 80 | 0.9136 | 0.9041 | −0.0095 |

Adversarial buckets (`tools/hard_cases.py --run`): `homogeneous_cluster`,
`budget_only_signal`, `cross_category_collision` unchanged; `generic_override`
+0.001; `degenerate_card` −0.003; **`boilerplate_soft` 0.893 → 0.880** — the one
real bucket regression, a browsing bucket where `InfoGainPolicy` costs ~0.013 MRR
against the cooperative disclosure.

The public cost is concentrated in the **boundary** scenario (dev boundary MRR
1.00 → 0.76). A boundary customer opens exactly like a browser — "still
exploring" — so they are routed to the browsing (InfoGain) policy, which spends a
question on an attribute they then decline. The router cannot tell them apart at
turn 1; the opening text is identical.

`router_off` scores **bit-for-bit** with the pre-branch agent on every split —
the flat path is untouched.

### On the realism harness — the payoff the public sim can't see

`python3 tools/stress_harness.py --customer browse-gated --configs router_off,router_on`

| | `router_off` | `router_on` | delta |
|---|---|---|---|
| overall score | 0.7308 | 0.8775 | **+0.147** |
| browsing Hit@10 | 0.59 | 0.95 | **+0.36** |
| browsing MRR | 0.24 | 0.67 | **+0.43** |
| browsing MTTC | 7.4 | 4.2 | **−3.2 turns** |
| buying (hit / mrr / mttc) | 1.00 / 0.90 / 2.8 | 1.00 / 0.90 / 2.8 | identical |

`router_off` on the realistic browser collapses: `FixedPolicy` asks "other" every
turn, the browser volunteers nothing, the pool never narrows, the target is found
in only 59% of sessions. `router_on` routes to targeted questioning and recovers
it to 95% — while leaving every buyer's number untouched.

### The misroute cost is ~10× asymmetric

`python3 tools/stress_harness.py --customer browse-gated --misroute-matrix`

| | true buyer | true browser |
|---|---|---|
| **routed as buyer** | 1.00 / 0.90 | 0.59 / 0.24 |
| **routed as browser** | 1.00 / 0.83 | 0.95 / 0.67 |

Treating a browser as a buyer costs 0.66 MRR and 41 points of hit rate — the
"other" spam gets nothing out of them. Treating a buyer as a browser costs 0.07
MRR — InfoGain still asks broadly first and a buyer answers everything. This is
the measured basis for "browsing is the safe fallback" (`IMPLEMENTATION.md:502`),
which until now was only asserted.

## 4. What is *not* claimed

- **No public-score gain.** `use_router=True` costs ~0.013 on the public set and
  ~0.01–0.015 on dev/holdout. It is not proposed for `main` and this branch does
  not merge. The value is the harness result and the Pillar-I / Presentation
  story.
- **One adversarial bucket regresses** (`boilerplate_soft`, −0.013). Under the
  house "no bucket regresses" rule this alone would disqualify a weight change on
  `main`.
- **The hard-filter lever is off by default** (`RerankConfig.hard_filter=False`).
  Turned on (`router_on_hardfilter` in `tools/sweep.py`) it regressed holdout
  0.9136 → 0.861 and adversarial boundary hit 1.00 → 0.75 — the authoritative
  facet extraction is not clean enough to evict on. It stays a documented switch,
  not a default.
- **The boundary regression is real and unfixed.** Routing a boundary opening to
  the broad policy instead recovers it but needs a turn-2+ signal (the customer
  declining), which measured as a no-op on the public sim and a regression on the
  harness (InfoGain's wrong-bucket questions also parse as declines). Left as-is.
- The harness is a *plausibility* instrument, not the private set. It shows that a
  less cooperative browsing customer makes routing load-bearing; it does not
  predict the 800 private sessions.

## 5. Reproduction

```
git checkout stress_harness            # dual_tracking + the merged harness
python3 -m unittest discover -s tests -t .                          # 95 tests
python3 tools/sweep.py --split holdout --configs router_off         # == pre-branch floor
python3 -m evaluator.local_evaluator                                # public, use_router on
python3 tools/hard_cases.py --run                                   # per-bucket
python3 tools/stress_harness.py --verify                            # delta 3.6e-07
python3 tools/stress_harness.py --customer browse-gated --configs router_off,router_on
python3 tools/stress_harness.py --customer browse-gated --misroute-matrix
```

## 6. Files

`starter/agent.py` (AgentConfig fields, `_track` / `_policy_for` / `_rerank_config`
/ `_first_recommend_turn` / `_list_size_ramp`, `_shortlist` threading),
`src/rerank.py` (`track` kwarg, `RerankConfig.hard_filter`, banish branch),
`tools/sweep.py` (`router_off` / `router_on` / `router_on_hardfilter` rows),
`tests/test_components.py` (+10). The `browse-gated` customer and its tests live
in `tools/stress_harness.py` / `tests/test_stress_harness.py` (branch
`stress_harness`). `src/router.py` unchanged — `detect_turn_intent` was already
there, just never called.
