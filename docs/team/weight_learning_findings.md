# Learning the S6 rerank weights — four methods, consolidated findings

*This file is identical on branches `kwongweng_fit_pairwise`, `kwongweng_fit_bo`,
`kwongweng_fit_cv`, `kwongweng_fit_lambdamart`. Each branch also carries its own
detailed writeup (`docs/team/pairwise.md`, `bo.md`, `cv.md`, `lambdamart.md`).
All four branches are exploratory — **nothing is proposed for merge.** Based on
`origin/main` `72b021f`.*

---

## TL;DR

The reranker (`src/rerank.py`) scores candidates as a **linear weighted sum** of
8 features. Seven weights are free (`span_weight` is the fixed unit). "Change 12"
(`docs/team/rerank_signals.md` §10) fit them once by coordinate ascent and found
a dev argmax that helped dev/holdout/public but **regressed the adversarial hard
set −0.016**; only the single move `popularity_weight 0.02 → 0.4` shipped.

We ran four more estimators for the same seven weights:

| # | branch | method |
|---|---|---|
| 1 | `kwongweng_fit_pairwise` | pairwise logistic regression (linear RankNet) |
| 2 | `kwongweng_fit_bo` | Bayesian optimization (Gaussian-process surrogate + Expected Improvement) |
| 3 | `kwongweng_fit_cv` | coordinate ascent with a variance-controlled accept rule (k-fold / bootstrap) |
| 4 | `kwongweng_fit_lambdamart` | gradient-boosted-tree ranker on ~50k synthetic catalog-wide sessions *(in progress)* |

**Result (methods 1–3): not one vector beats the shipped `popularity_weight =
0.4` on both the official evaluator and the adversarial hard set.** Every fit
that raises dev/holdout/public does it by cutting `retrieval_weight`, and every
such cut regresses the hard set by −0.015 to −0.022 and drops a converted hit
(0.896 → 0.885). `popularity_weight = 0.4` is the Pareto knee — the largest
popularity boost that does **not** also require cutting retrieval.

Why: the hard set's targets are thin / unreviewed products inside homogeneous
clusters where popularity is neutral and span coverage saturates, so BM25
**retrieval order is the only signal that can separate them**. The 120-session
dev split contains almost none of that distribution, so no fit computed from dev
can protect it. Method 4 tests whether a non-linear model trained on a
catalog-wide synthetic dataset (including stress-customer sessions) breaks that
ceiling.

---

## 1. The problem in detail

### 1.1 The model

`src/rerank.py:rerank()` takes the retrieved candidate pool and re-scores each
member:

```
total = 1.0·span_coverage                    (span_weight, the fixed unit)
      + pair_weight       · pair_coverage
      + retrieval_weight  · (retrieval_score / top_score)
      + popularity_weight · _popularity(product)
      + facet_weight      · _facet_agreement(customer_facets, product_facets)
      + category_weight   · _category_match(state, product)
      + tail_weight       · _tail_match(state, product)
      − facet_conflict_weight · _facet_conflicts(authoritative_facets, product_facets, text)
```

then sorts by `(−total, asin)`. The eight helpers are pure functions of the
`DialogState` and the product dict. `RerankConfig` (a dataclass) holds the eight
weights; shipped defaults: `span 1.0, retrieval 1.0, popularity 0.4, facet 0.3,
category 0.4, tail 0.8, pair 0.8, facet_conflict 0.4`.

The seven fitted weights, in the order every fitter uses them
(`FITTED` in `tools/fit_weights.py`):
`popularity, retrieval, facet, category, tail, pair, facet_conflict`.

### 1.2 The objective

`evaluator/local_evaluator.py:evaluate()` (frozen) returns
`recommended_technical_score = 0.50·Hit@10 + 0.30·MRR + 0.20·efficiency`, where
`efficiency = clip((11 − MTTC)/10, 0, 1)` and MTTC averages the first-hit turn
(a miss counts as 11). It also returns per-session records and per-scenario
metrics. The metric is **non-smooth** — a session ends the moment the target
first appears in a shown slate, so the score is a step function of the ranking,
and it is **weight-dependent** — the confidence gate reads rerank scores, so
changing the weights changes which slate is shown when, and therefore the
transcript.

### 1.3 The splits and the house method

`tools/sweep.py:split_samples()` splits `data/public_set.jsonl` (200 sessions,
stratified by scenario) into **dev = 120** and **holdout = 80**.
`data/hard_set.jsonl` (96 adversarial sessions, `tools/hard_cases.py`) is scored
by the same `evaluate()`. The private grader uses **800** sessions.

House rules (`docs/team/signal_descriptions.md`):
1. Reference weights (`span`, `retrieval`) = 1.0, not tuned.
2. Fit on **dev only**; holdout / hard are **one-shot gates**, never selectors.
3. What ships is a **rounded, plateau-checked** point (re-evaluate each weight at
   ±50%) — never the raw dev argmax. A *knife-edge* (best at one value,
   monotonically worse either side) is the signature of overfitting.
4. Penalty terms take the smallest weight on their plateau.
5. Public Hit@10 must stay **200/200**.

### 1.4 What change 12 already established

Plain coordinate ascent on the dev score converged (168 evals) to
`popularity 0.8, retrieval 0.1, facet 0.5, tail 1.2, conflict 0`:

| vector | dev | holdout | public / hit | hard / hit |
|---|---:|---:|---:|---:|
| baseline `pop 0.02` (pre-change-12) | 0.9268 | 0.9096 | 0.9199 / 200 | 0.7981 / .885 |
| change-12 raw dev argmax | 0.9520 | 0.9290 | 0.9428 / 200 | **0.7824** / .885 |
| **shipped: `pop 0.02 → 0.4` only** | 0.9418 | 0.9136 | 0.9305 / 200 | 0.8020 / .896 |

The holdout *confirmed the direction* of the argmax (+0.019 on unseen data — not
narrow dev overfit) but it **regressed the hard set −0.016** and lost a
converted hit. Only the smallest robust move shipped. The full argmax is
preserved as the `weights_argmax` sweep row.

The near-miss anatomy (`rerank_signals.md` §10) explained why the argmax helps
public: among the 33 public near-miss sessions, every lexical signal is *exactly
tied* between the target and the rank-1 impostor; the normalised retrieval score
breaks the tie toward the **impostor** 33/33 (BM25 length-normalisation favours
thin listings), while popularity points at the **target** 31/33 (the target is a
real purchase → a reviewed, documented product). So public wants `retrieval` out
and `popularity` up. The hard set is the opposite regime.

---

## 2. Shared harness (all four branches)

- **`tools/fit_common.py`** — identical file on all four branches. Provides
  `make_config(weights, variant)` (`variant="plain"` →
  `AgentConfig(use_router=False, policy=FixedPolicy())`, `variant="default"` →
  `AgentConfig()`), `load_all()`, `scalar_from_sessions()` (recomputes the
  official scalar from any subset of per-session records — used by Method 3 for
  free fold scores), `Scorer` (cached dev scorer), `gate()` (dev / holdout /
  public-200 / hard-96), `snap()` (round to the fit grid), `plateau()` (±50% per
  weight), `stress()` (shells the two stress-harness cells).
- **`tools/stress_harness.py`** + `tests/test_stress_harness.py` — vendored from
  the `dense_rerank` branch into each fit branch (they are not on `origin/main`).
  A robustness probe that drives the **unmodified** agent through non-cooperative
  customer models the official simulator cannot produce. `--verify` asserts the
  un-stressed path reproduces `evaluate()` (|Δ| ≈ 2e-7). Two cells are reported:
  `official` (= the evaluator, a consistency check) and
  `paraphrase:heavy+browse-gated` (the closest proxy to a hostile private grader
  — heavy synonym substitution so verbatim span coverage collapses, plus a
  browsing customer who only discloses when asked a pointed question).
- Every fitter runs **thread-pinned** (`OMP_NUM_THREADS=1` etc.) and `nice`.

---

## 3. Method 1 — pairwise logistic regression (linear RankNet)

**Branch `kwongweng_fit_pairwise`. Full writeup: `docs/team/pairwise.md`.**

### What it is

The classical *learning-to-rank* objective, with a linear scorer. Instead of
"maximise the metric," learn weights so that for every session
`score(target) > score(each non-target)` in that session's own pool. Pairwise
logistic regression on the feature *difference* `φ(target) − φ(impostor)` is
exactly a linear RankNet; the fitted coefficient vector, rescaled so the
span-coverage coefficient is 1.0, is a candidate weight vector.

### How it works (`tools/fit_weights_pairwise.py`)

1. **Instrumented snapshot pass.** Re-implements only `evaluate()`'s outer
   per-sample loop (importing every helper verbatim). Monkey-patches
   `src.rerank.rerank` with a spy that records, on every turn, the **pre-rerank
   candidate pool** `[(asin, retrieval_score), …]` plus `state.opening`,
   `full_text()`, `focused_text()`, `query_spans()`, `query_pair_spans()`. After
   each turn that shows a slate it stores a `Snapshot`. `--verify` asserts the
   loop reproduces `evaluate()`'s per-session records exactly (scalar Δ 0).
2. **Offline features.** For each snapshot, recompute the 8 features (via the
   reranker's own pure helpers + an exact replica of the two inline coverage
   loops) for the target + the top-20 non-target pool members.
3. **Fit.** Per snapshot, per negative: `d = φ(target) − φ(neg)`. Stack
   `X = [d; −d]`, `y = [1…; 0…]`, fit
   `LogisticRegression(fit_intercept=False, C=C)`. Convert the 8 coefficients:
   `w = c / c[span]`; `popularity = max(0, w[pop])` … `facet_conflict = max(0,
   −w[conflict])` (the conflict term is subtracted). `C ∈ {0.1, 1, 10}` selected
   once by best **dev** score.
4. **Iterate.** The transcript is weight-dependent, so refit on snapshots taken
   at `w_{i-1}` until `‖Δw‖/‖w‖ < 0.05` (≤ 3 iterations). Separately for
   `plain` and `default`.

### The "no per-candidate labels" objection

The data gives **one gold product per session and no graded relevance for
anything else**. So the negatives are *synthesised* (gold vs each top-20 pool
member). Mitigations: only shown-slate turns (pool already constraint-filtered);
use only the coefficient *direction*, rescaled; the plateau + hard-set gate do
the real filtering.

### What it found

| vector | popularity | retrieval | facet | category | tail | pair | conflict |
|---|---:|---:|---:|---:|---:|---:|---:|
| **plain — rounded** | **1.5** | **0.0** | 0.3 | 0.3 | 0.5 | 0.2 | 0.05 |
| **default — rounded** | **1.2** | **0.0** | 0.2 | 0.3 | 0.5 | 0.2 | 0.05 |

The raw fit wants `retrieval_weight` **strongly negative** every C and every
iteration (−0.55 to −0.84 as a coefficient) — the near-miss anatomy made visible
to the fit directly: among the impostors the target competes with, the
normalised retrieval score is *anti-correlated* with relevance, so the pairwise
objective would subtract it. Clamped to 0. `facet_conflict` is sign-unstable
near zero, clamps to ~0.04.

**Offline↔online disagreement:** the pairwise (offline) loss converges
monotonically, but the real dev score **falls every iteration** (plain 0.9483 →
0.9479 → 0.9462). The plateau line-search shows the rounded vector is a dev
*dip* on `popularity_weight` (better at both ±50%) — disqualifying by house rule.

### Gate (rounded vectors)

| vector | dev | holdout | public / hit | hard / hit | stress official | stress paraphrase / hit |
|---|---:|---:|---:|---:|---:|---:|
| baseline (default) | 0.9428 | 0.9141 | 0.9313 / **200** | **0.8028** / .896 | 0.9313 / 1.0 | 0.6861 / .815 |
| plain — rounded | 0.9460 | 0.9279 | 0.9388 / **200** | 0.7881 / **.885** | 0.9388 / 1.0 | 0.7441 / .840 |
| default — rounded | 0.9477 | 0.9315 | 0.9412 / **200** | 0.7877 / **.885** | 0.9412 / 1.0 | 0.7392 / .840 |

**Verdict:** nothing ships. Regresses the hard set −0.015, one lost hit — the
change-12 trade. The pairwise objective's answer is not materially different
from the direct-metric fit — it reaches the same conclusion, more extremely
(`popularity` 1.2–1.5 vs 0.8, `retrieval` a hard 0 vs 0.1) plus extra cuts to
`pair` / `tail` that are the synthesised-negative risk surfacing.

### Reproduce (branch `kwongweng_fit_pairwise`)

```bash
git checkout kwongweng_fit_pairwise
python3 tools/stress_harness.py --verify
python3 tools/fit_weights_pairwise.py --verify        # loop fidelity, both variants
# the fit — dev only, ~25 min, prints the full trajectory + plateau + writes JSON
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  nice -n 10 python3 tools/fit_weights_pairwise.py --variant both --out /tmp/pw.json
# the rounded weights are already the fit_pairwise_* rows in tools/sweep.py;
# the raw w_final is in /tmp/pw.json -> variants.{plain,default}.w_final
python3 tools/sweep.py --split dev     --configs fit_pairwise_plain,fit_pairwise_default
python3 tools/sweep.py --split holdout --configs fit_pairwise_plain,fit_pairwise_default
python3 tools/sweep.py --split all     --configs fit_pairwise_plain,fit_pairwise_default   # public 200 + hit
python3 tools/sweep.py --dataset data/hard_set.jsonl --split all \
  --configs fit_pairwise_plain,fit_pairwise_default
python3 tools/stress_harness.py --customer official \
  --configs pop040,fit_pairwise_plain,fit_pairwise_default
python3 tools/stress_harness.py --customer paraphrase:heavy+browse-gated \
  --configs pop040,fit_pairwise_plain,fit_pairwise_default
```

---

## 4. Method 2 — Bayesian optimization

**Branch `kwongweng_fit_bo`. Full writeup: `docs/team/bo.md`.**

### What it is

Treat the dev score as an expensive black-box `f(w)` over the 7-D weight box.
Build a probabilistic model of `f` from the runs done so far (a **Gaussian
Process**), and use it to pick the next `w` to try — the one that maximises
**Expected Improvement** (a balance of "the model predicts high here" and "the
model is uncertain here"). Coordinate ascent moves one weight at a time and only
climbs locally; BO searches the whole box.

### How it works (`tools/fit_weights_bo.py`)

1. **Box:** `popularity, retrieval, facet, category, tail, pair ∈ [0, 2]`;
   `facet_conflict ∈ [0, 1.5]`.
2. **Init:** 10 Latin-hypercube points (`scipy.stats.qmc`) + the baseline vector
   → 11 real `evaluate()` runs.
3. **Surrogate:** `GaussianProcessRegressor` with
   `ConstantKernel · Matérn(ν=2.5) + WhiteKernel` (`sklearn.gaussian_process`),
   refit each iteration. **The WhiteKernel (noise term) is essential** — `f` is a
   step function, so without a modelled noise term the GP interpolates every
   spike and its uncertainty collapses.
4. **Acquisition:** Expected Improvement (`ξ = 0.01`), maximised by 5000 random
   samples + 5 L-BFGS-B restarts.
5. **Budget:** 11 init + 40 EI iterations, per variant, per objective
   (`dev` and a 3-fold-CV variant). ~51 unique dev evaluations per variant —
   **~100 total for both variants** vs 168 for one coordinate-ascent run.

Every BO evaluation is a real `evaluate()`, so there is **no** weight-dependent
transcript mismatch and no iterate loop (BO's advantage over Method 1). Its
weakness is the opposite: on ~50 evaluations over a noisy 7-D step function, the
GP is a weak guide away from its data and EI chases apparent optima into the box
corners.

### What it found

| vector | popularity | retrieval | facet | category | tail | pair | conflict |
|---|---:|---:|---:|---:|---:|---:|---:|
| **plain — rounded** | **2.0** | 0.4 | 0.3 | 0.02 | 0.3 | 0.8 | 0.4 |
| **default — rounded** | **2.0** | 0.3 | 0.8 | 0.0 | 1.2 | 2.0 | 0.0 |

Both slam `popularity` to the box ceiling and `category` to ~0. The `--cv`
objective moved the argmax by less than rounding.

**Plateau:** neither vector is on a plateau. `plain` `popularity = 2.0` is a
knife-edge — dev *improves* when halved to 1.0 — and `conflict` wants to be
lower. `default` wants `popularity` and `retrieval` *higher* and `facet` lower.
The two variants disagree on the sign of the `popularity` gradient.

### Gate (rounded vectors)

| vector | dev | holdout | public / hit | hard / hit | stress official | stress paraphrase / hit |
|---|---:|---:|---:|---:|---:|---:|
| baseline (default) | 0.9428 | 0.9141 | 0.9313 / **200** | **0.8028** / .896 | 0.9313 / 1.0 | 0.6861 / .815 |
| plain — rounded | 0.9502 | 0.9250 | 0.9401 / **200** | 0.7836 / **.885** | 0.9401 / 1.0 | 0.7310 / .840 |
| default — rounded | 0.9491 | 0.9280 | 0.9407 / **200** | 0.7855 / **.885** | 0.9407 / 1.0 | 0.7383 / .840 |

**Verdict:** nothing ships. A global model-guided search finds the **same**
overfit corner as coordinate ascent and pairwise logistic (`popularity ↑,
retrieval ↓`) — faster, and more extreme (`popularity` pinned to the box edge).
Regresses the hard set −0.018 with a lost hit. Raising the budget would not help:
the holdout confirms these are not narrow dev overfit — they genuinely maximise
the public-shaped objective, and that objective trades the hard set away.

### Reproduce (branch `kwongweng_fit_bo`)

```bash
git checkout kwongweng_fit_bo
python3 tools/stress_harness.py --verify
# the fit — dev only, ~50 min, seed 0, deterministic
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  nice -n 10 python3 tools/fit_weights_bo.py --variant both --iters 40 --seed 0 \
  --out /tmp/bo.json
# rounded weights = the fit_bo_* rows in tools/sweep.py;
# raw argmax = /tmp/bo.json -> variants.{plain,default}.runs.dev.raw_best.weights
python3 tools/sweep.py --split dev     --configs fit_bo_plain,fit_bo_default
python3 tools/sweep.py --split holdout --configs fit_bo_plain,fit_bo_default
python3 tools/sweep.py --split all     --configs fit_bo_plain,fit_bo_default
python3 tools/sweep.py --dataset data/hard_set.jsonl --split all \
  --configs fit_bo_plain,fit_bo_default
python3 tools/stress_harness.py --customer official \
  --configs pop040,fit_bo_plain,fit_bo_default
python3 tools/stress_harness.py --customer paraphrase:heavy+browse-gated \
  --configs pop040,fit_bo_plain,fit_bo_default
```

---

## 5. Method 3 — coordinate ascent with a variance-controlled accept rule

**Branch `kwongweng_fit_cv`. Full writeup: `docs/team/cv.md`.**

### What it is

Change 12's failure was not the *search* — it was the *accept decision*. Plain
coordinate ascent accepts a move whenever the dev scalar rises past `1e-6`,
regardless of how concentrated that gain is across sessions. This method is
`tools/fit_weights.py` with **one change**: a variance filter on the accept
decision. Both filters cost **zero extra evaluations** — one `evaluate()` on all
120 dev sessions returns per-session records, and `scalar_from_sessions()`
recomputes the official scalar over any subset for free.

- **k-fold rule (default):** partition dev into 5 stratified folds. Accept a move
  only if the mean out-of-fold score improves **and** ≥ 4 of the 5 folds do not
  regress.
- **bootstrap rule (`--bootstrap 20`):** resample the 120 sessions 20× with
  replacement. Accept only if the **5th percentile** of the 20 paired
  `(trial − current)` differences is > 0.

*Honesty note:* this is not classical CV — there is no per-fold model training;
the "model" is a discrete weight choice. It is a **fold-agreement / paired-
subsample acceptance rule**; its claim is robustness of the accept *decision*,
not an unbiased generalisation estimate.

### What it found

| vector | popularity | retrieval | facet | category | tail | pair | conflict |
|---|---:|---:|---:|---:|---:|---:|---:|
| **k-fold plain** | 0.8 | **0.02** | 0.1 | 0.4 | 1.2 | 2.0 | 0.1 |
| **k-fold default** | 0.8 | **0.02** | 0.1 | 0.4 | 0.8 | 0.8 | 0.4 |
| **bootstrap plain** | **0.5** | **0.1** | 0.3 | 0.4 | 0.8 | 0.8 | 0.8 |
| **bootstrap default** | **0.5** | **0.1** | 0.3 | 0.4 | 0.8 | 0.8 | 0.4 |

- The **k-fold** rule rejected the *marginal* retrieval moves (`retrieval 0.02 →
  {0, 0.1, 0.2, 0.5}` — fold 3 regressed each time) but **accepted** the big
  `retrieval 1.0 → 0.02` cut and the `popularity 0.4 → 0.8` climb, because on
  those two moves all five dev folds agree. Change 12's headline moves survive
  the k-fold filter almost intact.
- The **bootstrap** rule — the strict one — **rejected both** of change-12's
  headline moves (`popularity → 0.8`: p5 = −0.001; `retrieval → 0.02`: p5 =
  −0.004) and accepted only `popularity 0.4 → 0.5` and `retrieval 1.0 → 0.1`,
  landing **one grid step from the shipped config**. This is the closest any of
  the five estimators (change 12 + the four here) came to *not* overfitting.

**Plateau:** unlike Method 2, the CV vectors sit on real plateaus — no
knife-edge. The variance filter produced *stable* dev optima. They just are not
good enough on the gate.

### Gate

| vector | dev | holdout | public / hit | hard / hit | stress official | stress paraphrase / hit |
|---|---:|---:|---:|---:|---:|---:|
| baseline (default) | 0.9428 | 0.9141 | 0.9313 / **200** | **0.8028** / .896 | 0.9313 / 1.0 | 0.6861 / .815 |
| k-fold plain | 0.9519 | **0.9323** | 0.9441 / **200** | 0.7862 / **.885** | 0.9441 / 1.0 | 0.7410 / .840 |
| k-fold default | 0.9523 | 0.9284 | 0.9427 / **200** | 0.7832 / **.885** | 0.9427 / 1.0 | 0.7342 / .830 |
| bootstrap plain | 0.9519 | 0.9246 | 0.9410 / **200** | 0.7811 / **.885** | 0.9410 / 1.0 | 0.7202 / .825 |
| bootstrap default | 0.9518 | 0.9263 | 0.9416 / **200** | 0.7837 / **.885** | 0.9416 / 1.0 | 0.7301 / .835 |

The striking row is **bootstrap default**: its weights are `popularity 0.5,
retrieval 0.1` and otherwise the shipped defaults — the single move
`retrieval 1.0 → 0.1` that the bootstrap rule *did* let through — and **that
alone regresses the hard set −0.019.**

**Verdict:** nothing ships. The method worked as designed — the strict rule
rejected the extreme moves — but it revealed that the problem is deeper than a
noisy accept decision: **cutting `retrieval_weight` at all breaks the hard set,
and no filter computed from 120 dev sessions can see that, because the dev split
has almost none of the thin-target distribution.** The fix is a different
*fitting objective*, which is Method 4.

### Reproduce (branch `kwongweng_fit_cv`)

```bash
git checkout kwongweng_fit_cv
python3 tools/stress_harness.py --verify
# the two fits — dev only, deterministic
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  nice -n 10 python3 tools/fit_weights_cv.py --variant both
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  nice -n 10 python3 tools/fit_weights_cv.py --variant both --bootstrap 20
# the argmax vectors print at the end of each run and are the fit_cv_* sweep rows
CFG=fit_cv_kfold_plain,fit_cv_kfold_default,fit_cv_boot_plain,fit_cv_boot_default
python3 tools/sweep.py --split dev     --configs $CFG
python3 tools/sweep.py --split holdout --configs $CFG
python3 tools/sweep.py --split all     --configs $CFG
python3 tools/sweep.py --dataset data/hard_set.jsonl --split all --configs $CFG
python3 tools/stress_harness.py --customer official --configs pop040,$CFG
python3 tools/stress_harness.py --customer paraphrase:heavy+browse-gated --configs pop040,$CFG
```

---

## 6. Method 4 — GBDT ranker on synthetic catalog-wide sessions *(in progress)*

**Branch `kwongweng_fit_lambdamart`. Full writeup: `docs/team/lambdamart.md`
(written when the run completes).**

### Hypothesis

Methods 1–3 all fit a **linear** model on the **120 dev sessions** and all
overfit the same way. Method 4 changes both:

1. **Non-linear model.** A linear model has one global coefficient per feature
   and must compromise between the tie-break regime (`popularity ↑, retrieval ↓`)
   and the thin-target regime (`retrieval` matters). A gradient-boosted tree
   ensemble can express *"if span-coverage gap ≈ 0 and rating count high →
   weight popularity; if rating count < 5 → weight retrieval rank"* — serve both
   at once. LambdaMART (GBDT + a listwise ranking loss) is the industry-standard
   tool for ranking over tabular features.
2. **Big, distribution-diverse training set.** Synthesise a session for (almost)
   every one of the ~49,700 catalog products that is **not** a public/hard test
   target, **including ~20–25% run through the stress-harness customer models**
   (`paraphrase:heavy`, `browse-gated`, `decoy`). This puts the thin-target /
   paraphrased distribution — the one the 120 dev sessions lack — into training.

### Honest expectation

The oracle-reranker ceiling (`rerank_signals.md` §9) is only **+0.043 public /
+0.084 hard** — that bounds any reranker. ~100 of 142 public near-misses already
rank the target #1. ~25 of the rest are homogeneous clusters where target and
impostor features are *identical* — a tree cannot split on features that do not
differ. And a large part of the hard-set / stress loss is retrieval **recall**
(under `paraphrase:heavy+browse-gated`, 29/80 browsing targets never enter the
pool) — no reranker touches that. So the realistic outcome is: **helps the
tie-break regime and the stress `official` cell, probably still does not fix the
hard set.** A clean negative result is itself valuable; a positive one ships (or
its distilled interaction rule ships as a conditional term needing no new
dependency).

### Integration (default-off)

`src/rerank.py` gains `RerankConfig.model_path` / `model_weight` (default `""` /
`0.0` — byte-identical to today). When set, the GBDT score is added as a **9th
additive term** with its own bracket-tuned weight, leaving the existing linear
sum fully ablatable.

### Reproduce — see `docs/team/lambdamart.md` on branch `kwongweng_fit_lambdamart`
once the run completes.

---

## 7. Consolidated conclusion

| config | weights (`pop / retr / facet / cat / tail / pair / conflict`) | dev | holdout | public / hit | hard / hit |
|---|---|---:|---:|---:|---:|
| **baseline (shipped)** | 0.4 / 1.0 / 0.3 / 0.4 / 0.8 / 0.8 / 0.4 | 0.9428 | 0.9141 | 0.9313 / **200** | **0.8028** / **.896** |
| change-12 argmax (ref) | 0.8 / 0.1 / 0.5 / 0.4 / 1.2 / 0.8 / 0.0 | 0.9520 | 0.9290 | 0.9428 / 200 | 0.7824 / .885 |
| M1 pairwise (default) | 1.2 / 0.0 / 0.2 / 0.3 / 0.5 / 0.2 / 0.05 | 0.9477 | 0.9315 | 0.9412 / 200 | 0.7877 / .885 |
| M2 BO (default) | 2.0 / 0.3 / 0.8 / 0.0 / 1.2 / 2.0 / 0.0 | 0.9491 | 0.9280 | 0.9407 / 200 | 0.7855 / .885 |
| M3 CV k-fold (default) | 0.8 / 0.02 / 0.1 / 0.4 / 0.8 / 0.8 / 0.4 | 0.9523 | 0.9284 | 0.9427 / 200 | 0.7832 / .885 |
| M3 CV bootstrap (default) | 0.5 / 0.1 / 0.3 / 0.4 / 0.8 / 0.8 / 0.4 | 0.9518 | 0.9263 | 0.9416 / 200 | 0.7837 / .885 |
| M4 LambdaMART | *pending* | | | | |

**Three findings that hold across all four estimators:**

1. **Every fit that raises dev/holdout/public cuts `retrieval_weight`, and every
   such cut regresses the hard set by −0.015 to −0.022 with a lost converted
   hit.** Public Hit@10 stays 200/200 throughout — the disqualifier is the hard
   set, every time.

2. **The regression is caused by `retrieval_weight` alone.** M3's bootstrap
   `default` vector is the shipped config with just `popularity 0.4→0.5` and
   `retrieval 1.0→0.1`, and it still regresses hard −0.019. The hard set's thin,
   unreviewed targets in homogeneous clusters have no popularity signal and
   saturated span coverage, so BM25 retrieval order is the only thing that ranks
   them — and every fit that helps public reduces its weight.

3. **`popularity_weight = 0.4` is the Pareto knee.** It is the largest popularity
   boost that does **not** require also cutting retrieval (§10 step-1 bracket:
   `pop` 0.1/0.3/0.4/0.5 are all ≥ baseline on every split). Every estimator that
   is *free* to cut retrieval finds a higher dev/holdout/public score and pays
   for it on the hard set.

**On the stress harness:** every fitted vector *improves* both stress cells —
`official` by +0.008 to +0.013, `paraphrase:heavy+browse-gated` by +0.034 to
+0.055 with (sometimes) a recovered hit. That improvement is real but comes
entirely from ranking the *retrieved* targets better; the paraphrase cell's
dominant failure mode is 29/80 browsing targets never being retrieved at all,
which no weight vector changes. The stress gain and the hard-set loss are the
same trade seen from two sides: popularity-dominant ranking helps when span
coverage is gone (paraphrase) and hurts when the target is genuinely unpopular
(hard set).

**Root cause and the way forward.** The 120-session dev split does not contain
the information needed to protect the hard set — it has almost none of the
thin-target / homogeneous-cluster distribution. No estimator, no accept rule, and
no amount of budget fixes that; the *objective* has to change. Method 4 tests the
one change that could: fit against a synthetic dataset drawn from the whole
catalog and through the stress customers, so the thing that breaks the hard set
is punished during training. If Method 4 also fails to clear the hard-set gate,
that is strong evidence the ceiling is real and `popularity_weight = 0.4` is the
correct shipped value — full stop.

---

## 8. Reproduce everything from scratch

```bash
# one-time: the four branches are on origin
git fetch origin

for b in kwongweng_fit_pairwise kwongweng_fit_bo kwongweng_fit_cv; do
  git worktree add ../acs-$b $b        # or: git checkout $b
done

# each branch: verify the vendored harness, run the fit, run the gates
# (commands are in §3 / §4 / §5 above and in each branch's docs/team/<method>.md
#  "Reproduction" section, all deterministic)

# the shipped baseline for comparison, from origin/main:
git checkout origin/main
python3 -m evaluator.local_evaluator                                   # public 200
python3 -m evaluator.local_evaluator --dataset data/hard_set.jsonl     # hard 96
python3 tools/sweep.py --split dev      --configs pop040
python3 tools/sweep.py --split holdout  --configs pop040
```

Every number in this file and in the per-method writeups is from a logged run on
the named branch (based on `origin/main` `72b021f`) against
`data/catalog.jsonl` / `data/public_set.jsonl` / `data/hard_set.jsonl`.
`python3 -m unittest discover -s tests -t .` → all pass on every branch;
`tests/test_components.py` is unmodified everywhere; `git diff --stat origin/main`
on each branch touches only `tools/`, `tests/test_stress_harness.py`,
`docs/team/`, and (Method 4 only) an additive default-off hook in `src/rerank.py`.
