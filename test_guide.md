# Testing & evaluation guide

Every kind of test and evaluation in this repo, and the exact command to run it
yourself. Read `README.md` and `IMPLEMENTATION.md` §3 ("How the score is
calculated") for background; this file is the operations manual.

All commands are run from the repo root with `python3`. There is no `make`, no
`pytest`, no task runner.

---

## Quick reference — "I want to know X → run Y"

| I want to know… | Run |
|---|---|
| The official leaderboard number (public set) | `python3 -m evaluator.local_evaluator` |
| The official number on the adversarial set | `python3 -m evaluator.local_evaluator --dataset data/hard_set.jsonl` |
| Did my code change break anything | `python3 -m unittest discover -s tests -t .` |
| Is config A better than config B | `python3 tools/sweep.py --split dev --configs a,b` then `--split holdout` |
| Did a change survive on data I didn't tune on | `python3 tools/sweep.py --split holdout` |
| Which adversarial buckets regressed | `python3 tools/hard_cases.py --run` |
| *Why* a specific session missed | `python3 tools/observe.py --tag mine` then open `runs/latest/viewer.html` |
| How robust is the agent to paraphrase / gated browsers | `python3 tools/stress_harness.py --all` |
| Is my change measured correctly | the checklist at the bottom of this file |

The **only** number that counts for the competition is the one from
`evaluator/local_evaluator.py`. Everything else is a decision aid.

---

## 1. One-time setup

### Catalog download

`data/catalog.jsonl` is a 50,000-row, ~60 MB file. It is **git-ignored** and does
not come with a clone. Per `README.md`:

```bash
# Download catalog.jsonl.gz from the repository's GitHub Release, then:
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
# Verify against the published SHA256SUMS file.
```

Expected: `wc -l data/catalog.jsonl` → `50000`.

### Dependencies

None. The agent, the evaluator, the sweep harness and the stress harness all run
on the Python standard library alone — no `pip install`, no network at scoring
time. (A `bge-small` embedding signal was tried on branch `dense_rerank`; it did
not help and this branch does not carry it — see `docs/team/dense_rerank.md`.)

### Confirm you're set up

```bash
python3 -c "from src.index import load_index; load_index()"
```

This parses the catalog and builds the in-memory FTS5 index (~4 s). If it returns
without error, everything downstream will run.

---

## 2. The unit test suite

```bash
python3 -m unittest discover -s tests -t .
```

Runtime ~35 s, almost all of it in `test_regression.py` (it runs the real
evaluator over the 80-session holdout split). "OK" at the end means every test
passed; any `FAIL`/`ERROR` names the test and shows the assertion.

Run one module / class / test:

```bash
python3 -m unittest tests.test_components
python3 -m unittest tests.test_components.RerankTests
python3 -m unittest tests.test_components.RerankTests.test_shipped_weights_are_pinned
```

| File | What it guards |
|---|---|
| `test_regression.py` | End-to-end scoring floor. Runs the **real evaluator on the holdout split**. `MINIMUM_SCORE = 0.78`, `MINIMUM_HIT_RATE = 0.88`, every scenario ≥ 0.70, token usage == 0. This is the net that catches "local win, global loss". |
| `test_contract.py` | The response envelope stays legal (dict, `message` str, `ask_attribute` in the `docs/agent_api_contract.json` enum, ≤ 10 unique in-catalog recs, non-negative int usage) on hostile input, missing `reset()`, cross-session isolation, no recs before evidence, `top_k`. A raised exception is a silent missed session. |
| `test_components.py` | Unit tests for the scoring-bearing stages: text/span normalisation, `DialogState`, facet extraction, policies (gain-ratio not dominated by cardinality), the reranker (verbatim span beats retrieval score, facet-conflict demotion, popularity tie-break; **`test_shipped_weights_are_pinned`** locks the 8 `RerankConfig` weights), dual-track routing, list-size ramp, `src/phrasing.py`. |
| `test_stress_harness.py` | The stressor grammar and customer policies in `tools/stress_harness.py` (spec parsing, browse-gating, paraphrase levels, decoy). No catalog load. |
| `test_observe.py` | `tools/observe.py` still recognises the evaluator's reply templates, and the failure-mode `diagnose()` logic (including that pre-override top-10 placements are not counted as wasted turns). |
| `test_stoplist.py` | The learned `src/stoplist.py` never strips attribute words (cotton, black, steel…), always strips structural metadata (department, dimensions, month names, years), and the hand-written care-phrase remainder survives regeneration. |

---

## 3. The official score (the leaderboard number)

```bash
python3 -m evaluator.local_evaluator                                 # public set, 200 sessions
python3 -m evaluator.local_evaluator --dataset data/hard_set.jsonl   # adversarial set, 96 sessions
```

Flags: `--catalog` (default `data/catalog.jsonl`), `--dataset` (default
`data/public_set.jsonl`), `--output` (default `results.json`).

- It **prints** the full result minus the per-session list, and **writes** the
  complete result (including `sessions`) to `results.json`.
- The headline number is `recommended_technical_score`:

  ```
  TechnicalScore = 0.50 * HitRate@10  +  0.30 * MRR  +  0.20 * Efficiency
  Efficiency = clamp((11 - MTTC) / 10, 0, 1)      MTTC: a miss counts as turn 11
  ```

- `scenario_metrics` breaks HitRate/MRR/MTTC down by `buying` / `browsing` /
  `intent_override` / `boundary`. A collapse hidden in the average shows up here —
  `intent_override` is the weakest scenario and the usual suspect for broken state
  handling.

Runtime ~28 s for the public set. **This evaluator is frozen** — it, `data/`, and
the five root `docs/` files are organizer-owned and must never be edited. The
score it prints is the real one; nothing else in this repo produces an official
number.

---

## 4. Config A/B on the dev / holdout split

```bash
python3 tools/sweep.py --split dev                       # all configs, 120 sessions
python3 tools/sweep.py --split holdout --configs floor,rerank
python3 tools/sweep.py --split all   --configs natural_off,natural_on --output sweep.json
```

Flags: `--split {dev,holdout,all}`, `--configs` (comma-separated subset; errors on
an unknown name), `--dataset`, `--catalog`, `--output` (optional full JSON).

`split_samples()` divides the 200 public sessions deterministically (sorted by
`sample_id`, stratified by scenario) with `DEV_FRACTION = 0.6` → **120 dev / 80
holdout**. All configs run in one process sharing one catalog index, then it
prints a table plus per-scenario hit/MRR components.

**Dev is for choosing; holdout is only a veto.** Differences below ~`0.02` on the
80-session holdout are noise. Never pick a weight or threshold by its holdout
argmax — that fits the 200 public sessions through the back door, and the private
800 decide the real score.

### Configs defined in `build_configs()` right now

| Name | What it is |
|---|---|
| `floor` | The committed baseline: terms-only retrieval, no reranker, `FixedPolicy`, hold recs until turn 3. |
| `rerank` | `floor` + post-override focused route + verbatim-span reranking. |
| `infogain`, `infogain_specific` | `InfoGainPolicy` instead of `FixedPolicy` (second bans broad questions). |
| `plain`, `elim1`/`2`/`3`, `elim_hold1`/`2` | Recommendation-timing sweep: first-recommend turn, hold-until-stalled. |
| `conflict00/04/08`, `pair00/08/15`, `ramp_flat/3/4/5/55`, `pop002/010/030/040/050` | Weight brackets for negative-facet-evidence, pair-span, first-slate size ramp, and popularity. Each brackets its shipped default (`0.4` / `0.8` / `(4,10)` / `0.4`). |
| `weights_argmax` | The coordinate-ascent dev argmax — higher on dev/holdout/public, **regresses the hard set**; kept as a row, not shipped. |
| `natural_off` / `natural_on` | Pool-aware clarification wording (`src/phrasing.py`). Must score **bit-for-bit identical** — the simulator never reads `message`. The row exists to prove that. |
| `router_off` / `router_on` / `router_on_hardfilter` | Dual-track routing. `router_off` is the flat single-track pipeline (bit-for-bit like the pre-routing agent); `router_on` lets the buying/browsing track drive policy/rerank/timing; `_hardfilter` adds the buying-track candidate banish. |

Add a row to `build_configs()` to test your own change.

---

## 5. Adversarial buckets

```bash
python3 tools/hard_cases.py --run          # regenerate data/hard_set.jsonl, then score it
python3 tools/hard_cases.py --per-bucket 20 --seed 7
```

Flags: `--run` (score the shipped agent after writing), `--per-bucket` (default
16), `--seed` (default 2026), `--output` (default `data/hard_set.jsonl`).

It picks catalog targets that sit in regions where the disclosed constraints are
**not** discriminative, across 6 buckets:

`homogeneous_cluster`, `budget_only_signal`, `boilerplate_soft`,
`degenerate_card`, `generic_override`, `cross_category_collision`.

`--run` prints a per-bucket table (n / hit@10 / MRR / MTTC / score) plus an `ALL`
row and the public-set reference line. You can also score the same file with the
frozen evaluator:

```bash
python3 -m evaluator.local_evaluator --dataset data/hard_set.jsonl
```

**The rule: no bucket regresses.** A change that lifts the overall score while
dropping one bucket is not accepted — that is the signature of overfitting to the
public distribution.

---

## 6. Per-session tracing

```bash
python3 tools/observe.py --tag mybaseline                       # full public set
python3 tools/observe.py --only public_0008 --tag one
python3 tools/observe.py --scenario browsing --tag browse
python3 tools/observe.py --dataset data/hard_set.jsonl --tag hard
python3 tools/observe.py --limit 20 --no-markdown               # quick triage
python3 tools/observe.py --verify                               # ~2x runtime, asserts identical score
```

Flags: `--dataset`, `--catalog`, `--tag` (names the folder), `--scenario`,
`--only` (comma-separated `sample_id`s), `--limit N`, `--top` (recs recorded per
turn, default 10), `--no-markdown` (keep only `trace.jsonl`), `--verify` (re-run
untraced and assert the score matches to the last digit).

Each run writes `runs/<tag>-<timestamp>/` (also symlinked as `runs/latest/`):

| File | Contents |
|---|---|
| `index.md` | One row per session, worst first. |
| `sessions/<id>.md` | Full annotated transcript: customer disclosures, route, state, pool rank → rerank rank, what was shown. |
| `trace.jsonl` | One JSON record per turn, machine-readable. |
| `summary.json` | Aggregate metrics + failure-mode counts + turns-left-on-the-table. |
| `viewer.html` | Self-contained offline browser; filter by scenario / outcome / failure mode. Open it directly. |

`runs/` is git-ignored — regenerate, don't commit.

### Diagnosis labels (the payoff)

| Label | Meaning | Stage to fix |
|---|---|---|
| `hit` | found and scored | — |
| `never_retrieved` | target never entered the candidate pool | S1 / S5 (retrieval / recall) |
| `ranked_out` | in the pool the whole time, never reached the shown top 10 | S6 (ranking) |
| `withheld_only` | reached the top 10, but the list was held back every such turn | S7 (timing) |
| `override_locked` | shown before the override fired, so the hit couldn't count | S3 (state) |
| `exhausted` | 10 turns elapsed with the target outside the top 10 | — |

A `never_retrieved` / `ranked_out` miss is an S1/S5 vs S6 problem and needs a
different repair from a `withheld_only` one (S7). Compare the *mix* of labels
between two runs, not just the score.

---

## 7. The stress harness (robustness, NOT the leaderboard)

`tools/stress_harness.py` drives the **unmodified** agent through a faithful copy
of the evaluator loop with customers the official simulator cannot produce:
paraphrasing, non-cooperative browsers, genuine decoys. Pure standard library.

**Always run `--verify` first:**

```bash
python3 tools/stress_harness.py --verify
```

It asserts the un-stressed path reproduces `python3 -m evaluator.local_evaluator`
(`|delta|` ~0, prints `PASS`). If this fails, the tree has moved — stop and
investigate.

```bash
python3 tools/stress_harness.py --all                                        # the stressor matrix
python3 tools/stress_harness.py --customer paraphrase:heavy+browse-gated
python3 tools/stress_harness.py --customer browse-gated --configs router_off,router_on
python3 tools/stress_harness.py --customer browse-gated --misroute-matrix
python3 tools/stress_harness.py --all --targets generic                      # hard-to-retrieve targets only
```

### Stressor spec grammar (`--customer`)

- `official` — the base cooperative customer (no stressor).
- `paraphrase:light` — same constraints, verbatim tokens kept, carrier sentence reworded.
- `paraphrase:medium` — the constraint itself reworded (`"color: blue"` → `"in blue"`).
- `paraphrase:heavy` — medium + broad synonym substitution (`leather` → `cowhide`) + clause shuffle + filler; erodes FTS5 recall, not just the span signal.
- `browse-gated` — the browsing customer discloses only when asked a *pointed* question whose bucket matches; never on the broad "anything else?".
- `decoy` — `intent_override` sessions where the pre-override preference is a real facet value the target lacks.
- Compose with `+`: `paraphrase:heavy+browse-gated` is the closest single spec to the feared private simulator.

Other flags: `--targets {all,generic}` (`generic` keeps only targets whose
constraint spans are all high-frequency in the catalog, so retrieval — not
ranking — is on the hook), `--configs` (comma-separated `tools/sweep.py` config
names), `--misroute-matrix` (force each track, tabulate true × routed; needs a
browse-gated spec), `--dataset`, `--catalog`.

### Reading the output

Main table columns: `hit@10`, `mrr`, `mttc`, `score`, `tok_cov` (fraction of the
ground-truth constraint tokens the agent actually saw — drops as paraphrase bites),
and `Δ` vs the first row.

Per-scenario retrieval-vs-ranking diagnostic (in `[...]`):

| Column | Meaning |
|---|---|
| `never_retrieved N/M` | target never entered the ~300-candidate pool — a **retrieval** failure |
| `pool_rank>100 N/M` | in the pool but buried deep |
| `median_pool_rank` | median position of the target in the pool |
| `ranked_out N/M` | in the pool, not a hit, final rank > 10 — a **ranking** failure |

These numbers are a **robustness probe and a private-set hypothesis**. They are
never the official score — do not quote them as one.

---

## 8. Measuring a change correctly — checklist

Distilled from `.claude/skills/record-change/SKILL.md`. Follow it whenever you
touch `src/`, `starter/`, or `tools/` in a way that could move the score.

1. **Baseline first.** Capture before you edit — not after, not from memory, not
   from a number quoted earlier in the conversation. Run the evaluator,
   `tools/observe.py` (public + hard), and `tools/sweep.py --split dev` /
   `--split holdout`.
2. **Check the tree isn't moving under you.** `git status --porcelain` and file
   mtimes *before and after* the run. Several people work in this repo; a
   teammate's commit landing mid-measurement voids the comparison — re-run. If
   `sweep.py` and `local_evaluator.py` disagree on the same config, that's the
   alarm.
3. **Measure both states in one process** where you can (loop over
   `before`/`after`, re-evaluating with everything else held constant). One run,
   not two.
4. **Holdout is a gate, never a selector.** It must not regress. Sub-`0.02`
   holdout moves are noise. Never tune a weight or threshold to its dev/holdout
   argmax.
5. **No hard-set bucket regresses.** Run `tools/hard_cases.py --run` on both
   states and compare every bucket, not just `ALL`.
6. **Sit mid-plateau, not at the argmax.** If a range of values all beat the
   baseline and the curve between them is flat, pick the middle. The public 200
   decide nothing; the private 800 decide everything.
7. **Compare the failure-mode mix, not only the score.** Run `tools/observe.py`
   on both states. A change that holds the score but moves sessions from
   `never_retrieved` to `ranked_out` improved retrieval and hurt ranking — say so.
8. **A measured no-change is a real result.** Report `0.000000` plainly and state
   what the change buys instead (robustness, auditability, dead code removed). Do
   not hunt for a split where it happens to look positive.

When done, log it with the `record-change` skill (updates `IMPLEMENTATION.md` and
`docs/team/agent_changes.md`) and re-run the unit suite.
