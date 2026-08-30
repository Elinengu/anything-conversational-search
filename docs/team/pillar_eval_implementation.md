# Pillar Evaluation — Implementation Briefs (for a fresh Claude session)

Each TASK below is self-contained: open a new session, point it at ONE task, and it
has everything it needs. Tasks can run in parallel sessions — the interface contracts
between them are stated explicitly. Human-readable rationale lives in
[pillar_eval_plan.md](pillar_eval_plan.md); you do not need it to implement.

## Ground rules (read first, applies to every task)

- Repo root: this repository. Python 3, **standard library only** (no new deps).
- Run tests: `python3 -m unittest discover tests` — all must pass before and after.
- **`evaluator/local_evaluator.py` is frozen organizer code. Never modify it.**
  Import from it freely.
- The public score is pinned: `tests/test_regression.py` must stay green.
- Datasets in `data/{catalog,public_set}.jsonl` are read-only inputs.
- Repo culture: deterministic, seeded; dead options are deleted, not parked;
  docstrings state what drift a test catches.

### Key frozen-evaluator facts you will rely on

- Session record schema (`data/public_set.jsonl`), exactly these keys:
  `sample_id, scenario_type, category_bucket, difficulty_bucket, ground_truth{parent_asin}, user_profile{...}`.
  `scenario_type ∈ {buying, browsing, intent_override, boundary}`.
- `materialize_hidden_fields(sample, products)` (`evaluator/local_evaluator.py:204`):
  if a sample carries explicit `intent_card` and/or `behavior` keys, they are used
  **verbatim**; otherwise they are derived from the target product. Extra/unknown keys
  in a sample are ignored by `evaluate()`. This is the extension point for custom
  datasets.
- `intent_card` shape: `{"target_category": str, "hard_constraints": [str], "soft_preferences": [str]}`.
- `behavior` shape (only intent_override has structure):
  `{"scenario_type": "intent_override", "override": {"turn": int, "old_value": str, "new_value": str, "message": str}}`.
  The override message is sent as the user message on `override.turn`; `new_value` is
  added to the disclosed set; hits cannot count before the override fires.
- `customer_reply(sample, ask_attribute, disclosed, boundary_used)` (`:166`): on an
  ask, scans `hard_constraints + soft_preferences` for undisclosed values where
  `attribute == "other"` or `classify_constraint(value) == attribute`, and discloses
  **at most 2** per turn. `classify_constraint` (`:137`) cascade:
  budget → material → color → size → style → use_case → default `feature`.
  There is no brand/category branch.
- `initial_message` (`:154`): buying opens with `hard_constraints[0]` (pre-disclosed);
  intent_override opens with the `override.old_value` text; browsing/boundary open
  with only a coarse category.
- Other importables: `load_jsonl(path)` `:90`, `catalog_index(path)` `:112` →
  `(ids, categories, products)`, `intent_card(product)` `:52`, `behavior_for` `:74`,
  `coarse_category` `:126`, `metric_summary(sessions)` `:188`,
  `evaluate(agent, samples, catalog_ids, categories, products)` `:216`.
- CLI: `python3 -m evaluator.local_evaluator --catalog data/catalog.jsonl --dataset <any.jsonl> --output results.json`
  — `--dataset` accepts any jsonl with the schema above.

---

## TASK 1 — bucket support in `tools/observe.py` (small)

**Goal:** dataset-defined buckets survive into traces and summaries. Today,
per-bucket metrics exist only as stdout in `tools/hard_cases.py --run`; a
`data/hard_set.jsonl` run of observe loses the `hard_bucket` field entirely.

**Modify (one file):** `tools/observe.py`
1. In `TracingAgent.reset` (~line 183, where `sample.get("difficulty_bucket")` etc.
   are copied into the session record): also copy `sample.get("bucket") or
   sample.get("hard_bucket")` into a `bucket` key on the trace record.
2. In the summary construction (~lines 817–838, where `scenario_metrics` is built):
   add `bucket_metrics` — group the evaluator's per-session outcome dicts by the
   trace record's `bucket` (skip records with no bucket) and run each group through
   `metric_summary` imported from `evaluator.local_evaluator` (`:188`), exactly as
   `scenario_metrics` does. Emit `{}` when no record has a bucket.

**Interface contract (Task 3 depends on this):** trace records gain an optional
top-level `"bucket": str|null`; `summary.json` gains `"bucket_metrics": {bucket: metric_summary_dict}`.

**Acceptance:**
- `python3 tools/observe.py --catalog data/catalog.jsonl --dataset data/hard_set.jsonl --tag hard --verify`
  passes `--verify` (traced score exactly equals untraced) and the new summary
  contains one `bucket_metrics` entry per `hard_bucket` value.
- A public-set run still works; its `bucket_metrics` is `{}`.
- `tests/test_observe.py` passes unchanged. Add a small test in that file's style:
  build a minimal record dict with/without `bucket` and assert grouping behavior.

---

## TASK 2 — `tools/generate_pillar_sets.py` + `tests/test_pillar_sets.py` (medium)

**Goal:** three seeded, validated pillar-stress datasets that run through the frozen
evaluator via the explicit `intent_card`/`behavior` extension point.

**CLI:** `python3 tools/generate_pillar_sets.py --pillar 1|2|3 [--seed 20260830] [--output data/pillarN_set.jsonl]`

**Reuse (do not reimplement):**
- From `evaluator.local_evaluator`: `intent_card` `:52`, `behavior_for` `:74`,
  `coarse_category` `:126`, `classify_constraint` `:137`, `load_jsonl` `:90`.
- From `tools.hard_cases`: `load_catalog`, `classify_product`, `primary`,
  `MATERIALS`, `COLORS`, `synth_profile` (bucket mining: `classify_product(product,
  cluster_size, matcol_count) -> [bucket]` with buckets incl. `degenerate_card`,
  `boilerplate_soft`, `cross_category_collision`, `budget_only_signal`; see its
  `build()` for the cluster/matcol Counter pass to copy).
- From `tools/generate_test_set.py`: the `validate()` strict-schema-gate pattern
  (`:142`), `existing_targets()` (`:85`), `PROTECTED_INPUTS` refusal (`:23`),
  `profile()`/`preference_tags()` for clean profiles.

**Output record contract:** the public 6-key schema **plus** `"bucket": str`, and —
only where the bucket needs it — explicit `"intent_card"` and/or `"behavior"`.
`sample_id` prefix `pillar{N}_`. Targets must not overlap `public_set.jsonl`,
`hard_set.jsonl`, `generated_test_set.jsonl`, `generated_adversarial_set.jsonl`, or
each other. Skip products whose derived card has empty `hard_constraints`.

**Pillar 1 (~100): routing & retrieval stress**
- `paraphrased_constraints` (~40, scenario buying): derive the real card, then
  rewrite constraint strings with a seeded, deterministic synonym table (e.g.
  "genuine leather"→"real leather", "stainless steel"→"rust-proof steel",
  "sterling silver"→"925 silver", budget "budget around $X"→"hoping to stay near
  $X") + light reordering. Requirements: rewritten text must NOT be a verbatim
  substring of the product's title/features/description corpus, and
  `classify_constraint(rewritten)` must return the same attribute as the original
  (so the simulator still routes asks correctly). Ship the card via explicit
  `intent_card`.
- `recall_hard` (~40, buying): targets mined from `cross_category_collision` and
  `budget_only_signal` buckets. No explicit card (evaluator derives).
- `routing_probe` (~20): 5 per scenario across all four scenarios, plain derived
  cards — clean openings for routing-accuracy measurement.

**Pillar 2 (~100): dialog stress**
- `decoy_override` (~30, intent_override): explicit `behavior` where
  `override.old_value` is a constraint drawn from a *different* product in the same
  coarse category (the decoy), `new_value` = the target card's `hard_constraints[0]`,
  `turn` ∈ {3,4}, `message` = `f"Actually, ignore my earlier preference. What I need is: {new_value}."`.
  (The public simulator's old_value is target-derived; a genuine decoy is the test
  down-weighting-vs-erasure never had.)
- `early_override` / `late_override` (~10 each, intent_override): as
  `behavior_for` would build, but `turn` forced to 2 / 6 via explicit `behavior`.
- `sparse_disclosure` (~30, buying or browsing): explicit `intent_card` with 5–6
  constraints total (extend the derived card with additional cleaned feature/detail
  strings from the product) so one `"other"` ask (max 2 disclosures/turn) cannot
  drain it — question strategy starts to matter.
- `decline_heavy` (~20): 10 boundary-scenario sessions plus 10 sessions whose cards
  contain only values that `classify_constraint` maps to `feature` (so targeted asks
  for material/color/etc. return "no additional preference").

**Pillar 3 (~80): orchestration stress**
- `stagnation` (~40, browsing): targets from `degenerate_card` + `boilerplate_soft`
  mining — cards with almost nothing to disclose, producing unproductive streaks.
- `misleading_profile` (~20): normal cards, `synth_profile(..., misleading=True)`.
- `verbose_context` (~20): explicit `intent_card` whose constraint strings are the
  *longest* cleaned feature/detail strings on the product (boilerplate-laden) —
  the context-distillation case.

**`tests/test_pillar_sets.py`** (unittest, follow `tests/test_stoplist.py` /
`test_observe.py` style; module docstring says what drift each test catches):
1. Schema gate per generated set: exact key set (+ `bucket`, optional
   `intent_card`/`behavior`), unique sample_ids, targets ⊆ catalog, zero overlap with
   the four existing sets.
2. **Round-trip test (the important one):** build one sample with a custom
   `intent_card` and one with a custom `behavior`, run them through the real
   `materialize_hidden_fields` / `initial_message` / `customer_reply` and assert the
   simulator honours them (custom constraint text is what gets disclosed; decoy
   override message fires at the custom turn). This pins the extension point against
   evaluator drift.
3. Paraphrase invariants: rewritten text not a substring of the product corpus;
   `classify_constraint` attribute preserved; generation is deterministic for a
   fixed seed.
4. Generator refuses to overwrite `PROTECTED_INPUTS`.

**Acceptance:** all three sets generate and validate; each runs end-to-end via
`python3 -m evaluator.local_evaluator --catalog data/catalog.jsonl --dataset data/pillarN_set.jsonl`;
tests green.

---

## TASK 3 — `tools/pillar_metrics.py` + `tests/test_pillar_metrics.py` (medium)

**Goal:** turn observe traces into per-pillar metric tables and before/after
comparisons. Pure-function core (testable without catalog or agent), thin CLI.

**Input — `runs/<tag>/trace.jsonl`** (one JSON object per session), produced by
`python3 tools/observe.py --catalog data/catalog.jsonl --dataset <set> --tag <tag>`.
Per-session top-level keys:

```
sample_id, session_id, scenario_type, difficulty_bucket, category_bucket,
bucket (Task 1; may be absent — fall back to parsing sample_id or hard_bucket),
user_profile, target, target_title, target_price, target_categories,
coarse_category, intent_card, behavior, turns[], outcome, diagnosis
```

`outcome` = `{sample_id, scenario_type, hit, first_hit_turn, best_rank, reciprocal_rank}`.
`diagnosis` = `{label ∈ {hit, never_retrieved, ranked_out, withheld_only,
override_locked, exhausted}, explanation, best_pool_rank, best_ranked_rank,
earliest_top10_turn, turns_left_on_table, disclosures, dead_ends}`.

Per-turn record in `turns[]`:

```jsonc
{"turn": 1,
 "in": {"message": str, "kind": "opening|disclosed|override|boundary_decline|no_preference|stalled",
        "revealed": [str], "attribute": str|null},
 "route": {"name","confidence","buying_score","browsing_score","scenario_hint",
           "cues":[...], "facets":{...}} | null,
 "state": {"turn_count","override_turn","asked":[...],"dead_attributes":[...],
           "productive_turns","last_turn_productive","spans":[...],"focused_text": str},
 "retrieval": {"pool_size", "target_pool_rank": int|null, "ms"},
 "rerank": {"target_rank": int|null, "top":[{"rank","parent_asin","score","title","is_target"}], "ms"},
 "out": {"message","ask_attribute","shown_count","shown":[...],
         "target_shown_rank": int|null, "withheld": bool},
 "latency_ms": float, "error": null}
```

**CLI:**
- `python3 tools/pillar_metrics.py runs/<tag>` → writes `runs/<tag>/pillar_report.json`
  and `runs/<tag>/pillar_report.md` (three markdown tables, one per pillar).
- `python3 tools/pillar_metrics.py --compare runs/<baseline> runs/<candidate>` →
  side-by-side per-bucket table: score / hit / MTTC / and each pillar metric, with
  deltas. This is the before/after evidence table for report citations.

**Metrics (exact definitions):**

*Pillar I — routing & retrieval*
- Routing confusion matrix over turn-1 records: predicted = `route.scenario_hint`,
  actual = `scenario_type`; report accuracy + per-scenario precision/recall, and
  mean `route.confidence` for correct vs incorrect predictions.
- Never-retrieved rate: sessions where `retrieval.target_pool_rank` is null on every
  turn, per bucket.
- Rank lift: per turn where both defined, `target_pool_rank − rerank.target_rank`;
  report mean/median per bucket (what reranking buys over raw retrieval order).

*Pillar II — dialog*
- Override detection: for sessions with `behavior.override`, latency =
  `state.override_turn − behavior.override.turn` at end of session (null → missed);
  false-positive rate = non-override sessions where `state.override_turn` is set.
- Recovery turns: `outcome.first_hit_turn − behavior.override.turn` on override hits.
- Constraint capture rate: fraction of `in.revealed` strings (turns ≥ 2) whose
  content tokens (drop tokens ≤ 2 chars and pure digits) all appear in the *next*
  turn's `state.spans` joined text.
- Question efficiency: disclosures per ask = total revealed values / asks issued;
  `diagnosis.dead_ends` mean; productive-turn ratio = final `state.productive_turns`
  / turns taken; mean `turns_left_on_table`.

*Pillar III — orchestration*
- Longest unproductive streak per session (consecutive turns ≥ 2 with
  `in.kind ∈ {no_preference, stalled}` or `last_turn_productive == false`).
- Post-stagnation behavior delta: after a streak ≥ 2, did the agent change anything
  observable next turn (`out.ask_attribute` differs from previous ask, or
  `retrieval.pool_size` changes, or shown-list overlap < 100%)? Report the fraction
  of stagnated sessions where behavior changed (baseline expectation: ~0 — the
  deterministic pipeline; that number is the impact story for strategy switching).
- Personalization safety: score delta (`metric_summary`) between `misleading_profile`
  bucket and the same run's non-misleading sessions.
- Context economy: per-turn token counts of `state.focused_text` vs the concatenated
  span text; latency p50/p95 from `latency_ms`.

**`tests/test_pillar_metrics.py`:** hand-built minimal session/turn dicts via a
`_session(...)`/`_turn(...)` factory (pattern: `tests/test_observe.py:55-62`, no
catalog, sub-second); one test per metric family (confusion matrix arithmetic,
override latency incl. missed/false-positive, capture rate tokenization, streak
detection, compare-mode delta math); if you introduce any label dict, add a
taxonomy-completeness test (every produced label has an entry).

**Acceptance:** report generation works on an existing run dir (e.g. any
`runs/public-*` — Pillar II/III sections must tolerate absent `bucket` and absent
`behavior.override`); numbers cross-check against that run's `summary.json`
aggregates; tests green.

---

## Definition of done (all tasks)

- `python3 -m unittest discover tests` fully green (including `test_regression.py`).
- `evaluator/local_evaluator.py` untouched (`git diff --stat` shows no change there).
- `tools/observe.py --verify` still exact-matches the untraced score.
- All generated datasets pass their validators; no target overlaps any existing set.
- No new third-party dependencies.
