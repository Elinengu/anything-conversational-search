# Pillar Evaluation Plan — Proving Impact, Not Claiming It

Companion to [pillar_suggestions.md](pillar_suggestions.md). That document says *what*
to build for Pillars I–III; this one says how we will **measure** it, so every feature
lands with a before/after table instead of a paragraph of intent.

---

## Why

The public metric is saturated (public 0.9305, Hit@10 = 1.000) and judging is
Technical Execution 35% / Innovation 20% / Impact 20% / Feasibility 15% /
Presentation 10%. The pillar features we plan (dual-track routing, over-generality
cutoff, strategy switching, context distillation) are all flag-gated and score-neutral
on the public set *by design* — so the public evaluator cannot show their value. We
need datasets that contain the situations those features exist for, and metrics that
see inside the session, not just the final score.

Current standing (2026-08-30):

| Set | N | Hit@10 | MRR | MTTC | Score |
|---|---|---|---|---|---|
| public | 200 | 1.000 | 0.901 | 3.00 | **0.9305** |
| hard | 96 | 0.896 | 0.720 | 4.09 | **0.8020** |
| generated adversarial | 200 | 0.930 | 0.808 | 3.68 | **0.8537** |

## The trick that makes this cheap

The frozen official evaluator honours a session record's own `intent_card` and
`behavior` fields when they are present, instead of deriving them — so we can ship
custom sessions (paraphrased constraints, decoy overrides, custom override turns)
and run them through the **untouched** organizer code. And our tracer
(`tools/observe.py`) already wraps that evaluator and logs per turn: the router's
classification, whether the target entered the retrieval pool and at what rank, its
rerank position, the full dialog state, and latency. We are three small components
away from a pillar-level microscope.

## What we build

### 1. Three pillar-stress datasets

**`data/pillar1_set.jsonl` — routing & retrieval (~100 sessions)**

| bucket | stresses | proves suggestion |
|---|---|---|
| `paraphrased_constraints` | constraint text is rewritten, not verbatim catalog text — lexical recall loses its crutch | I-2 dense fallback route |
| `recall_hard` | cross-category collisions + budget-only cards (mined like hard_cases) | I-2, I-3 category route |
| `routing_probe` | normal openings across all 4 scenarios | I-1 dual-track wiring (routing accuracy) |

**`data/pillar2_set.jsonl` — dialog strategy (~100 sessions)**

| bucket | stresses | proves suggestion |
|---|---|---|
| `decoy_override` | the abandoned preference is from a *different* product — down-weighting vs erasure finally has a real test | II state handling |
| `early_override` / `late_override` | override on turn 2 / turn 6 (public only uses 3–4) | II robustness |
| `sparse_disclosure` | 5–6-constraint cards; one "other" ask can't drain them | II-1 cutoff, II-3 hybrid policy |
| `decline_heavy` | asks that miss, boundary declines | II dead-attribute handling |

**`data/pillar3_set.jsonl` — orchestration (~80 sessions)**

| bucket | stresses | proves suggestion |
|---|---|---|
| `stagnation` | degenerate/boilerplate cards → unproductive turn streaks | III-1 strategy switching |
| `misleading_profile` | profile tags the target cannot satisfy | III-3 personalization safety |
| `verbose_context` | long boilerplate-laden constraints | III-2 context distillation |

### 2. A small observe.py extension

Carry the dataset's `bucket` field into traces and add a `bucket_metrics` breakdown
to `summary.json` (today per-bucket numbers exist only as stdout in
`hard_cases.py --run`). The tracer's `--verify` guarantee — traced score identical to
untraced — stays mandatory.

### 3. A pillar analyzer (`tools/pillar_metrics.py`)

Reads trace files, emits a JSON report + three report-ready markdown tables:

- **Pillar I** — routing confusion matrix (predicted scenario vs actual), routing
  confidence calibration, never-retrieved rate per bucket, rank lift from reranking.
- **Pillar II** — override detection latency and false positives, recovery turns
  after override, constraint capture rate, disclosures per ask, dead-end turns.
- **Pillar III** — longest unproductive streak, whether agent behavior changes after
  stagnation (baseline answer: it doesn't — that's the point), misleading-vs-clean
  profile score delta, context size per turn, latency percentiles.

Plus a `--compare baseline candidate` mode: the before/after impact table each
flag-gated feature cites in the report.

## How it gets used

1. **Now:** baseline runs on all three sets. This alone is report material — it
   quantifies today's gaps with numbers ("on decoy overrides, recovery takes X turns";
   "after 3 unproductive turns the agent's behavior never changes").
2. **Per feature:** implement a suggestion from `pillar_suggestions.md` behind its
   flag, re-run, cite the delta. Innovation and Impact become measured claims.
3. **Report:** the "what the public simulator can and can't measure, and how we tested
   beyond it" section writes itself from these tables.

## Order & effort

1. Observe extension (S) — unblocks everything.
2. Dataset generator + validation tests (M).
3. Analyzer + tests (M).
4. Baseline runs + numbers into docs (S).

House rules unchanged: the official evaluator is never modified; all features stay
opt-in; the public floor (0.9305) is pinned by `tests/test_regression.py`; dev
selects, holdout gates.

Implementation briefs for each component (self-contained, one per Claude session) are
in [pillar_eval_implementation.md](pillar_eval_implementation.md).
