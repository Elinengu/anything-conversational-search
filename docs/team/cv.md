# Learning the rerank weights, method 3: a variance-controlled accept rule

*Branch `kwongweng_fit_cv`. Exploratory study — not a shipping proposal. Fit on
the 120-session dev split only; holdout (80), public-200 and hard-96 are
one-shot gates.* Companion: `tools/fit_weights_cv.py`, `tools/fit_common.py`
(shared, identical across the three method branches), `tools/stress_harness.py`
(vendored from `dense_rerank`). Sweep rows `fit_cv_kfold_plain` /
`fit_cv_kfold_default` / `fit_cv_boot_plain` / `fit_cv_boot_default`.

## 1. Problem

Change 12 (`docs/team/rerank_signals.md` §10, step 2) ran plain greedy
coordinate ascent on the official technical score over the seven fitted rerank
weights. The dev argmax was `popularity 0.8, retrieval 0.1, facet 0.5,
tail 1.2, conflict 0`. The one-shot holdout gate *confirmed the direction*
(+0.019 on data the fit never saw) — but the adversarial hard set **regressed
by 0.016** (MRR 0.725 → 0.675). Only the single-weight change
`popularity_weight 0.02 → 0.4` shipped.

The failure was not the search; it was the **accept decision**. Plain
coordinate ascent accepts a move whenever the dev scalar rises past
`EPSILON = 1e-6`, regardless of how concentrated that gain is across sessions.
A move that helps 90 dev sessions by a hair and hurts 30 by a lot is accepted
exactly like a move that helps all 120. This method adds one thing: a variance
filter on the accept decision.

## 2. Method

`tools/fit_weights_cv.py` is `tools/fit_weights.py` with **one substantive
change** — the acceptance rule. Everything else is byte-identical: the absolute
`GRID = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0)`, the `FITTED`
cycle order, `span_weight` fixed at 1.0, the current value always retained in
the candidate set, `EPSILON = 1e-6`, the weight-tuple cache, the initial vector
= shipped `RerankConfig()` defaults. `--max-cycles` default is 3.

**Why it costs zero extra evaluations.** `evaluate()` returns, alongside the
scalar, `result["sessions"]` — a per-session record
`{sample_id, scenario_type, hit, first_hit_turn, best_rank, reciprocal_rank}`.
`scalar_from_sessions()` (`tools/fit_common.py`) reconstructs the exact official
scalar `0.5·hit@10 + 0.3·mrr + 0.2·eff`, `eff = clip((11−mttc)/10, 0, 1)`, from
any subset of those records. So **one** `evaluate()` on all 120 dev sessions
gives every fold score and every bootstrap-resample score for free. Every
candidate vector still gets a real `evaluate()` — the session transcript is
weight-dependent (a session ends at the first hit, the confidence gate reads
scores), so there is no weight-dependent-transcript mismatch; the only thing
reused is the arithmetic over the returned per-session outcomes.

**k-fold rule (default).** Partition the 120 dev sessions into 5 stratified
folds: group by `scenario_type`, sort each group by `sample_id`, assign the
j-th member of a group to fold `j % 5`. For a candidate move `current → trial`,
compute the 5 fold scores of each. Accept iff **both**:

1. `mean(trial folds) > mean(current folds) + EPSILON`, and
2. `trial fold[i] ≥ current fold[i] − 1e-9` for **at least 4 of the 5 folds**.

Clause 2 is the variance filter: a move that lifts the mean by concentrating
its gain in one or two folds while regressing others is rejected.

**bootstrap rule (`--bootstrap 20`).** `rng = numpy.random.default_rng(0)`;
draw 20 index arrays, each 120 draws with replacement from `range(120)`. For
each resample `b`, `diff_b = score(trial sessions[idx_b]) −
score(current sessions[idx_b])`. Accept iff `numpy.percentile(diffs, 5) > 0` —
the 5th-percentile lower bound of the paired difference is positive.

**Honesty note.** This is *not* classical cross-validation. There is no
per-fold model training — the "model" is a discrete choice of weight value. It
is a fold-agreement / paired-subsample **acceptance rule**. Its claim is
robustness of the accept *decision* (the thing change 12 lacked), not an
unbiased estimate of generalisation. The fold scores are in-sample; they are
used only to check *agreement*, not to estimate held-out performance.

## 3. The fit

### 3.1 k-fold rule

Both variants converge in one improving cycle (cycle 2 finds nothing). The
trajectory, `plain` variant:

| move | plain-CA verdict | k-fold verdict | why |
|---|---|---|---|
| `popularity 0.4 → 0.3` | accept (dev +0.0005) | **reject** | folds 0, 4 regress |
| `popularity 0.4 → 0.5 → 0.8` | accept | **accept** | all 5 folds rise |
| `retrieval 1.0 → 0` | accept (dev +0.0005) | **reject** | folds 0, 4 regress |
| `retrieval 1.0 → 0.02` | accept | **accept** | all 5 folds rise |
| `retrieval 0.02 → 0.1 / 0.2 / 0.5` | accept (dev +0.0002 to +0.0003) | **reject** | fold 3 regresses each time |
| `facet 0.3 → 0.05 → 0.1` | accept | **accept** | all folds rise |
| `tail 0.8 → 1.2`, `pair 0.8 → 2.0`, `conflict 0.4 → 0.1` | accept | **accept** | all folds rise |

**k-fold argmax (plain):** `popularity 0.8, retrieval 0.02, facet 0.1,
category 0.4, tail 1.2, pair 2.0, conflict 0.1` — dev 0.951917, folds
`[0.949, 0.964, 0.939, 0.951, 0.958]`.
**k-fold argmax (default):** `popularity 0.8, retrieval 0.02, facet 0.1,
category 0.4, tail 0.8, pair 0.8, conflict 0.4` — dev 0.952250, folds
`[0.951, 0.965, 0.940, 0.949, 0.957]`.

**What the k-fold rule caught, and what it didn't.** It rejected the *final*
retrieval move (`0.02 → up`, and `1.0 → 0`) on fold-3 disagreement — but it
**accepted** the big `retrieval 1.0 → 0.02` cut and the `popularity 0.4 → 0.8`
climb, because on those two moves all five dev folds agree. Change 12's two
headline moves survive the k-fold filter almost intact.

### 3.2 bootstrap rule

The 5th-percentile test is much stricter. `plain` variant:

| move | plain-CA verdict | bootstrap verdict | p5 of paired diff |
|---|---|---|---|
| `popularity 0.4 → 0.5` | accept | **accept** | +0.0005 |
| `popularity 0.5 → 0.8` | accept (dev +0.0025) | **reject** | −0.0010 |
| `popularity 0.5 → 1.2 / 2.0` | accept | **reject** | −0.003 / −0.008 |
| `retrieval 1.0 → 0.02` | accept (dev +0.0048) | **reject** | −0.0041 |
| `retrieval 1.0 → 0.05` | accept | **reject** | −0.0024 |
| `retrieval 1.0 → 0.1` | accept | **accept** | +0.0014 |
| `facet_conflict 0.4 → 0 … 0.3` | accept (dev up to +0.0012) | **reject** (all) | ≤ 0 |
| `facet_conflict 0.4 → 0.8` | (would not test) | **accept** | +0.0002 |

**bootstrap argmax (plain):** `popularity 0.5, retrieval 0.1, facet 0.3,
category 0.4, tail 0.8, pair 0.8, conflict 0.8` — dev 0.951875.
**bootstrap argmax (default):** `popularity 0.5, retrieval 0.1, facet 0.3,
category 0.4, tail 0.8, pair 0.8, conflict 0.4` — dev 0.951815 (identical to
shipped except `popularity 0.4 → 0.5` and `retrieval 1.0 → 0.1`).

**The bootstrap rule rejects *both* of change 12's signature moves** —
`popularity → 0.8` and `retrieval → 0.02` — because in the worst 5% of session
resamples each move is a net loss. It lands one grid step from the shipped
config. The `plain` run additionally pushes `facet_conflict` *up* to 0.8 (the
opposite direction to every other method), which the plateau (§4) shows is a
weak local effect.

## 4. Rounding and plateau

All four argmax vectors already lie on the `snap()` grid `{0, .02, .05, .1, .2,
.3, .4, .5, .8, 1, 1.2, 1.5, 2}`, so "rounded" = "raw".

### k-fold plain, dev at ±50% per weight (base 0.951917)

| weight (value) | ×0.5 | ×1.5 | reading |
|---|---:|---:|---|
| popularity (0.8) | 0.948312 | 0.950500 | both slightly worse — a mild local max |
| retrieval (0.02) | 0.951917 | 0.951917 | flat (near 0) |
| facet (0.1) | 0.950500 | 0.950042 | both worse |
| category (0.4) | 0.951917 | 0.951250 | ~flat |
| tail (1.2) | 0.951417 | 0.951750 | ~flat |
| pair (2.0) | 0.951750 | 0.951417 | ~flat |
| conflict (0.1) | 0.951750 | 0.949917 | mild |

### bootstrap plain, dev at ±50% per weight (base 0.951875)

| weight (value) | ×0.5 | ×1.5 | reading |
|---|---:|---:|---|
| popularity (0.5) | 0.947750 | 0.949236 | both worse — 0.5 is a local max |
| retrieval (0.1) | 0.951083 | 0.950167 | both slightly worse |
| facet (0.3) | 0.951708 | 0.949653 | mild |
| category (0.4) | 0.951542 | 0.951542 | flat |
| tail (0.8) | 0.950792 | 0.952042 | mild up |
| pair (0.8) | 0.951875 | 0.952042 | flat |
| conflict (0.8) | 0.950815 | 0.951250 | weak — 0.8 barely beats its neighbours |

**Unlike Method 2 (BO), the CV vectors sit on real plateaus** — no knife-edge,
no weight screaming to move by a grid step. The variance filter did produce
*stable* dev optima. They just are not *good enough* on the gate.

## 5. Gate table

Holdout and hard are one-shot gates. Dev is the fitting set.

| vector | dev (120) | holdout (80) | public (200) / hit | hard (96) / hit |
|---|---:|---:|---:|---:|
| baseline (shipped defaults) | 0.942757 | 0.914119 | 0.931302 / **200/200** | **0.802811** / .896 |
| change 12 dev argmax (ref) | 0.951958 | 0.928958 | 0.942758 / 200/200 | 0.782400 / .885 |
| **k-fold plain** | 0.951917 | **0.932333** | 0.944083 / **200/200** | 0.786223 / **.885** |
| **k-fold default** | 0.952250 | 0.928396 | 0.942708 / **200/200** | 0.783190 / **.885** |
| **bootstrap plain** | 0.951875 | 0.924583 | 0.940958 / **200/200** | 0.781101 / **.885** |
| **bootstrap default** | 0.951815 | 0.926333 | 0.941623 / **200/200** | 0.783680 / **.885** |

**Every vector regresses the hard set 0.8028 → 0.781–0.786 (−0.017 to −0.022)
and loses a converted hit (0.896 → 0.885)** — the change-12 trade, again.
Public stays 200/200; dev / holdout / public all gain.

The striking row is **bootstrap default**: its weights are `popularity 0.5,
retrieval 0.1` and otherwise the shipped defaults — the single move
`retrieval 1.0 → 0.1` that the bootstrap rule *did* let through — and that
alone regresses the hard set by −0.019. Cutting `retrieval_weight` *at all* is
the hard-set killer, because the hard set's thin, unreviewed targets sit in
homogeneous clusters where span coverage saturates and popularity is neutral,
so BM25 retrieval order is the only signal left to rank them by.

Per-scenario hit/MRR, **bootstrap default** (baseline in parens):

| | pub boundary | pub browsing | pub buying | pub override | hard browsing | hard buying | hard override |
|---|---|---|---|---|---|---|---|
| baseline | 1.00/0.860 | 1.00/0.910 | 1.00/0.899 | 1.00/0.899 | .906/0.732 | .896/0.727 | .875/0.673 |
| bootstrap default | 1.00/1.00 | 1.00/0.926 | 1.00/0.918 | 1.00/0.888 | **.875**/0.704 | .906/0.712 | **.844**/0.657 |

**Stress harness** (`tools/stress_harness.py`; robustness probe, **not** the
official score). Baseline row is `pop040` (= shipped defaults). `--verify`
passes (|Δ| ≈ 2e-7).

| customer | baseline (`pop040`) | k-fold plain | k-fold default | boot plain | boot default |
|---|---:|---:|---:|---:|---:|
| `official` (= evaluator) | 0.93130 / 1.000 | 0.94408 / 1.000 | 0.94271 / 1.000 | 0.94096 / 1.000 | 0.94162 / 1.000 |
| `paraphrase:heavy+browse-gated` | 0.68607 / **.815** | 0.74096 / **.840** | 0.73418 / .830 | 0.72023 / .825 | 0.73005 / **.835** |

Same shape as Methods 1 and 2: on `official` the fitted vectors gain +0.010 to
+0.013; on the paraphrase cell they gain +0.034 to +0.055 and (k-fold plain)
recover a lost hit — again by ranking the *retrieved* targets better, not by
touching the ~29/80 browsing targets that heavy paraphrase never retrieves. A
robustness-for-hard-set trade that neither the rules nor the evaluator reward.

## 6. Verdict

**Nothing here ships.** The hard-set gate fails for all four vectors (−0.017 to
−0.022, one lost hit).

But the method **worked as designed** — it just revealed that the problem is
deeper than a noisy accept decision:

- The **k-fold** rule rejected the *marginal* retrieval moves (fold-3
  disagreement) but accepted the big `retrieval 1.0 → 0.02` and
  `popularity → 0.8` climbs, because on those the dev folds genuinely agree.
- The **bootstrap** rule — the strict one — rejected **both** of change 12's
  headline moves and landed one grid step from the shipped config
  (`popularity 0.5, retrieval 0.1`). This is the closest any of the four
  estimators (change 12, pairwise, BO, CV) came to *not* overfitting.
- **Yet even that near-shipped vector regresses the hard set.** The one move it
  accepted, `retrieval 1.0 → 0.1`, is by itself the regression.

The conclusion is not "use a better accept rule." It is: **the dev distribution
does not contain the information needed to protect the hard set.** No filter
computed from 120 dev sessions can see that cutting `retrieval_weight` hurts
thin-target sessions, because the dev split has almost none of them. Fixing this
requires changing the *fitting objective* to include that distribution —
e.g. fitting against synthetic sessions generated across the whole catalog and
through the stress-harness customer models, not the 120 cooperative dev
sessions (see `docs/team/` weight-learning notes / Method 4).

`popularity_weight = 0.4` (change 12's shipped value) stays the defensible
choice: it is the largest popularity boost that does **not** also require
cutting retrieval, and cutting retrieval is what every fit that beats dev
does and what breaks the hard set every time.

## 7. Reproduction

```bash
git checkout kwongweng_fit_cv
python3 tools/stress_harness.py --verify
python3 -m unittest tests.test_stress_harness

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  nice -n 10 python3 tools/fit_weights_cv.py --variant both
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  nice -n 10 python3 tools/fit_weights_cv.py --variant both --bootstrap 20

CFG=fit_cv_kfold_plain,fit_cv_kfold_default,fit_cv_boot_plain,fit_cv_boot_default
python3 tools/sweep.py --split dev     --configs $CFG
python3 tools/sweep.py --split holdout --configs $CFG
python3 tools/sweep.py --split all     --configs $CFG
python3 tools/sweep.py --dataset data/hard_set.jsonl --split all --configs $CFG
python3 tools/stress_harness.py --customer official --configs pop040,$CFG
python3 tools/stress_harness.py --customer paraphrase:heavy+browse-gated --configs pop040,$CFG
```

All numbers are from logged runs on branch `kwongweng_fit_cv` (based on
`origin/main` `72b021f`). `python3 -m unittest discover -s tests -t .` → all
pass; `tests/test_components.py` unmodified; `git diff --stat origin/main`
touches only `tools/fit_weights_cv.py`, `tools/fit_common.py`,
`tools/stress_harness.py`, `tests/test_stress_harness.py`, `tools/sweep.py`,
`docs/team/cv.md`.
