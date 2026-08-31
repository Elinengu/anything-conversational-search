# S6 rerank weights — Method 2: Bayesian optimization

Branch `kwongweng_fit_bo`. Exploratory. **Nothing here is proposed for shipping.**
This is a third independent estimator for the same seven rerank weights that
change 12 fit by coordinate ascent (`rerank_signals.md` §10) and Method 1 fit by
pairwise logistic regression (`docs/team/pairwise.md`).

Companion tooling: `tools/fit_weights_bo.py` (the optimizer), `tools/fit_common.py`
(shared eval helpers, identical across the three method branches),
`tools/stress_harness.py` (vendored from `dense_rerank`). Sweep rows
`fit_bo_plain` / `fit_bo_default`.

---

## 1. Problem

Same as `pairwise.md` §1: `src/rerank.py` scores each candidate as a linear sum
of eight features; `span_weight` is the fixed unit, the other seven are free.
Change 12's coordinate-ascent dev argmax (`popularity 0.8, retrieval 0.1, facet
0.5, tail 1.2, conflict 0`) beat baseline on dev/holdout/public but **regressed
the adversarial hard set −0.016**; only `popularity_weight 0.02 → 0.4` shipped.

Coordinate ascent is a *local* search — it moves one weight at a time and can
only climb. This method asks whether a *global*, model-guided search of the whole
7-D box finds anywhere better. Like change 12 and Method 1 it treats
`popularity_weight` as a tunable feature weight, setting aside the house rule that
priors are set "by reasoning, not sweeping" (`signal_descriptions.md` §5).

---

## 2. Method

Bayesian optimization treats the dev technical score as an **expensive black-box
function** `f(w)` of the 7-D weight vector and spends a fixed evaluation budget
finding its maximum, using a probabilistic model of `f` to decide where to look
next.

### 2.1 The loop (`tools/fit_weights_bo.py`)

1. **Search box** (real units): `popularity, retrieval, facet, category, tail,
   pair ∈ [0, 2]`; `facet_conflict ∈ [0, 1.5]`. Standardised to `[0, 1]^7`
   internally.
2. **Initial design:** 10 points from a Latin-hypercube (`scipy.stats.qmc`,
   seed 0) plus the shipped `RerankConfig()` vector — 11 real `evaluate()` runs.
   Including the baseline guarantees the incumbent is never worse than shipped.
3. **Surrogate:** a Gaussian Process
   `ConstantKernel · Matérn(ν=2.5, per-dim length scale) + WhiteKernel`
   (`sklearn.gaussian_process`), refit on all observations each iteration.
   **The WhiteKernel (noise term) is essential** — the technical score is a step
   function of the ranking (the session ends at the first hit), so `f` is
   genuinely discontinuous. Without a noise term the GP tries to interpolate
   every spike and its uncertainty estimates collapse.
4. **Acquisition:** Expected Improvement (`ξ = 0.01`), maximised by 5000 random
   samples + 5 L-BFGS-B restarts. EI picks the point that best balances "the GP
   predicts a high score here" against "the GP is uncertain here".
5. **Budget:** 11 init + **40 EI iterations per variant**. The holdout / hard
   gate — not more iterations — is what catches overfit.
6. **Objective variants:** `dev` (plain dev technical score) and `--cv` (mean of
   a 3-fold stratified dev CV, recomputed per point from the one `sessions`
   list — free). Run for both, and for `variant ∈ {plain, default}`.

### 2.2 Why this is cleaner than Method 1

Every BO evaluation is a **real `evaluate()` call**, so there is no
offline↔online mismatch and no weight-dependent-transcript iterate loop —
the thing that made Method 1's fit walk *away* from the real metric. BO's
weakness is the opposite: with only ~50 evaluations over a noisy 7-D
step-function, the GP is a weak guide away from the data it has seen, and EI
will chase apparent optima into the corners of the box.

---

## 3. The fit

`--variant both --iters 40`, seed 0. 51 unique dev evaluations per variant
(1701 s plain, 1123 s default) — **~100 evaluations total for both variants**,
versus 168 for one coordinate-ascent run. Faster per unit of search.

### 3.1 Raw dev argmax

| variant / objective | popularity | retrieval | facet | category | tail | pair | conflict | dev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **plain / dev** | **2.00** | 0.39 | 0.29 | 0.01 | 0.35 | 0.81 | 0.36 | 0.950083 |
| **default / dev** | 1.77 | 0.32 | 0.74 | 0.00 | 1.17 | **2.00** | 0.00 | 0.950792 |
| shipped defaults | 0.4 | 1.0 | 0.3 | 0.4 | 0.8 | 0.8 | 0.4 | 0.9428 |
| change 12 dev argmax | 0.8 | 0.1 | 0.5 | 0.4 | 1.2 | 0.8 | 0.0 | 0.9520 |

The `--cv` objective moved the argmax by less than rounding on both variants and
is not tabulated separately.

**Both variants slam a weight to the box edge** — `popularity → 2.0` (plain),
`pair → 2.0` (default) — and both drive `category → ~0`. This is the same
overfit corner change 12 and Method 1 reached (`popularity ↑, retrieval ↓`),
found here by a global search that was free to move all seven weights at once.
The two variants disagree on *which* weight to max out, which is itself a sign
the objective surface near the top is flat and noisy rather than genuinely
peaked.

### 3.2 Rounded vectors

Snapped to the grid `{0, .02, .05, .1, .2, .3, .4, .5, .8, 1, 1.2, 1.5, 2}`:

| vector | popularity | retrieval | facet | category | tail | pair | conflict |
|---|---:|---:|---:|---:|---:|---:|---:|
| **plain — rounded** | **2.0** | 0.4 | 0.3 | 0.02 | 0.3 | 0.8 | 0.4 |
| **default — rounded** | **2.0** | 0.3 | 0.8 | 0.0 | 1.2 | 2.0 | 0.0 |

---

## 4. Plateau — dev score at ±50% per weight

### plain rounded (base dev 0.950167)

| weight (value) | ×0.5 | ×1.5 | reading |
|---|---:|---:|---|
| popularity (2.0) | **0.951625** | 0.949792 | **better when halved to 1.0** — the box edge is past the real optimum |
| retrieval (0.4) | 0.946732 | 0.945875 | both worse — 0.4 is fine |
| facet (0.3) | 0.948917 | 0.948292 | both worse |
| category (0.02) | 0.950190 | 0.950292 | ~flat |
| tail (0.3) | 0.945833 | 0.950458 | wants higher (toward shipped 0.8) |
| pair (0.8) | 0.950167 | 0.950167 | flat |
| conflict (0.4) | **0.952083** | 0.950417 | **better when lowered** |

### default rounded (base dev 0.949125)

| weight (value) | ×0.5 | ×1.5 | reading |
|---|---:|---:|---|
| popularity (2.0) | 0.948667 | **0.951083** | **wants to go past the box ceiling** |
| retrieval (0.3) | 0.949167 | 0.950792 | wants higher |
| facet (0.8) | **0.951958** | 0.949375 | **wants lower** (toward shipped 0.3) |
| category (0.0) | 0.949125 | 0.949125 | flat |
| tail (1.2) | 0.949000 | 0.950208 | ~flat, mild up |
| pair (2.0) | 0.949125 | 0.948958 | flat |
| conflict (0.0) | 0.949125 | 0.949125 | flat |

**Neither vector sits on a plateau.** On `plain`, `popularity` is a knife-edge —
halving it *improves* dev — and `conflict` wants to be lower. On `default`,
`popularity` and `retrieval` want to go *higher* and `facet` wants to go *lower*.
The two variants even disagree on the sign of the `popularity` gradient. Change
12's shipping bar was that `popularity` 0.1/0.3/0.4/0.5 are all ≥ baseline on
every split (a real plateau); BO's argmax is a noisy ridge point, not a plateau.

---

## 5. Gate table

Holdout and hard are one-shot gates. Dev is the fitting set.

| vector | dev (120) | holdout (80) | public (200) / hit | hard (96) / hit |
|---|---:|---:|---:|---:|
| baseline (shipped defaults) | 0.942757 | 0.914119 | 0.931302 / **200/200** | **0.802811** / .896 |
| change 12 dev argmax (ref) | 0.951958 | 0.928958 | 0.942758 / 200/200 | 0.782400 / .885 |
| BO **plain — raw argmax** | 0.950083 | 0.925271 | 0.940158 / **200/200** | 0.784570 / **.885** |
| BO **plain — rounded** | 0.950167 | 0.925021 | 0.940108 / **200/200** | 0.783606 / **.885** |
| BO **default — raw argmax** | 0.950792 | 0.930208 | 0.942558 / **200/200** | 0.782471 / **.885** |
| BO **default — rounded** | 0.949125 | 0.928021 | 0.940683 / **200/200** | 0.785473 / **.885** |

Read the hard column: **every BO vector regresses the adversarial set 0.8028 →
~0.784 (−0.018) and loses a converted hit (0.896 → 0.885)** — the same trade,
on the same thin/unreviewed target distribution, that stopped change 12's argmax
(−0.016) and Method 1's pairwise vectors (−0.015). Public stays 200/200; dev /
holdout / public all gain ~+0.008 / +0.012 / +0.009 over baseline.

Per-scenario hit/MRR, **rounded** vectors (baseline in parens):

| | pub boundary | pub browsing | pub buying | pub override | hard browsing | hard buying | hard override |
|---|---|---|---|---|---|---|---|
| baseline | 1.00/0.860 | 1.00/0.910 | 1.00/0.899 | 1.00/0.899 | .906/0.732 | .896/0.727 | .875/0.673 |
| BO plain — rounded | 1.00/1.00 | 1.00/0.930 | 1.00/0.919 | 1.00/0.878 | **.875**/0.686 | .906/0.706 | **.844**/0.660 |
| BO default — rounded | 1.00/1.00 | 1.00/0.928 | 1.00/0.921 | 1.00/0.884 | **.875**/0.690 | .917/0.717 | **.812**/0.638 |

Same shape as Method 1: public boundary MRR jumps (0.86 → 1.00 — the boundary
customer discloses nothing, so a dominant `popularity` term carries ranking),
and hard `browsing` / `override` lose hits.

**Stress harness** (`tools/stress_harness.py`; robustness probe, **not** the
official score). Baseline row is `pop040` (= shipped defaults). `--verify`
passes (|Δ| ≈ 2e-7).

| customer | baseline (`pop040`) | `fit_bo_plain` | `fit_bo_default` |
|---|---:|---:|---:|
| `official` (= evaluator) | 0.93130 / hit 1.000 | 0.94011 / 1.000 | 0.94068 / 1.000 |
| `paraphrase:heavy+browse-gated` | 0.68607 / hit **.815** | 0.73100 / **.840** | 0.73827 / **.840** |

Like Method 1, the paraphrase row cuts the *other* way: under heavy paraphrase +
browse-gating (`tok_cov` 0.83 → 0.44, verbatim span coverage collapses) the
reranker falls back on the priors, and the popularity-dominant BO vectors
**recover +0.045 to +0.052 of score and a lost hit** (.815 → .840) over
baseline. The recovery is entirely in the sessions where the target *is*
retrieved — `ranked_out` drops 4/80 → 0/80 on browsing — while the 29/80
browsing targets that never enter the pool are untouched (a retrieval-recall
wall, not a ranking one). So BO, like every other method here, trades
**adversarial-target robustness** (the hard set) for **paraphrase robustness**.
Neither the house rules nor the official evaluator reward the second trade;
recorded as a real property of the vectors.

---

## 6. Verdict

**Nothing here ships.**

1. **The hard-set gate fails** for every BO vector (−0.018, one lost hit) — the
   change-12 trade.
2. **No vector sits on a plateau** (§4): `plain` popularity is a knife-edge,
   `default` wants three weights to move further, and the two variants disagree
   on the popularity gradient's sign.

**Does a global search find anywhere better than local coordinate ascent?** No —
it finds the *same* overfit corner (`popularity ↑, retrieval ↓`), faster (~100
evals vs 168), and if anything *more* extreme (`popularity` pinned at the box
edge). On a ~50-evaluation budget over a noisy 7-D step function, the GP surrogate
is a weak guide and Expected Improvement chases dev noise into the box corners.
Raising the budget would not help: the holdout confirms these vectors are not dev
overfit in the narrow sense (holdout gains too) — they genuinely maximise the
public-shaped objective, and that objective genuinely trades the hard set away.

**Does any BO vector beat `pop 0.4` on both evaluators without regressing the
hard set or public hit?** No. `popularity_weight = 0.4` remains the defensible
shipped value — the Pareto knee between the tie-break regime (which every fit
climbs) and the thin-target regime (which every fit sacrifices).

---

## 7. Reproduction

```bash
git checkout kwongweng_fit_bo

python3 tools/stress_harness.py --verify
python3 -m unittest tests.test_stress_harness

# the fit (dev only; ~50 min shared box)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  nice -n 10 python3 tools/fit_weights_bo.py --variant both --iters 40 --seed 0 \
  --out /tmp/bo_fit.json

# gates (rounded vectors = the fit_bo_* sweep rows)
python3 tools/sweep.py --split dev      --configs fit_bo_plain,fit_bo_default
python3 tools/sweep.py --split holdout  --configs fit_bo_plain,fit_bo_default
python3 tools/sweep.py --split all      --configs fit_bo_plain,fit_bo_default
python3 tools/sweep.py --dataset data/hard_set.jsonl --split all \
  --configs fit_bo_plain,fit_bo_default
python3 tools/stress_harness.py --customer official \
  --configs pop040,fit_bo_plain,fit_bo_default
python3 tools/stress_harness.py --customer paraphrase:heavy+browse-gated \
  --configs pop040,fit_bo_plain,fit_bo_default
```

All numbers are from logged runs on branch `kwongweng_fit_bo` (based on
`origin/main` `72b021f`). `python3 -m unittest discover -s tests -t .` → all
pass; `tests/test_components.py` unmodified; `git diff --stat origin/main`
touches only `tools/fit_weights_bo.py`, `tools/fit_common.py`,
`tools/stress_harness.py`, `tests/test_stress_harness.py`, `tools/sweep.py`,
`docs/team/bo.md`.
