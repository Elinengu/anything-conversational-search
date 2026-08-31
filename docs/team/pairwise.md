# S6 rerank weights — Method 1: pairwise logistic regression (linear RankNet)

Branch `kwongweng_fit_pairwise`. Exploratory. **Nothing here is proposed for
shipping** — this is a second, independent estimator for the same seven rerank
weights that change 12 fit by coordinate ascent, run to see whether a different
objective lands anywhere different.

Companion tooling: `tools/fit_weights_pairwise.py` (the fitter),
`tools/fit_common.py` (shared eval helpers, identical across the three
weight-learning method branches), `tools/stress_harness.py` (vendored from
`dense_rerank` for the robustness gate). Sweep rows `fit_pairwise_plain` /
`fit_pairwise_default` in `tools/sweep.py`.

---

## 1. Problem

`src/rerank.py` scores each candidate as a linear sum of eight features:

```
total = 1.00*coverage + pair_weight*pair_coverage + retrieval_weight*(retr/top)
      + popularity_weight*popularity + facet_weight*facet + category_weight*category
      + tail_weight*tail - facet_conflict_weight*conflict
```

`span_weight` is the definitional unit (fixed 1.0); the other seven weights are
free. Change 12 (`rerank_signals.md` §10) fit them by **coordinate ascent
directly on the non-smooth technical score**, dev split only. Its dev argmax
(`popularity 0.8, retrieval 0.1, facet 0.5, tail 1.2, conflict 0`) was confirmed
on the sealed holdout (+0.019) but **regressed the adversarial hard set 0.016**
(`hard_cases.py` draws thin, unreviewed targets where a strong popularity prior
overshoots). Only the one-weight change `popularity_weight 0.02 → 0.4` shipped.

This method fits the **same linear model** with the classical learning-to-rank
objective instead: for each session, the gold product should out-score every
non-gold candidate in that session's own retrieved pool. Pairwise logistic
regression on the feature *differences* `phi(gold) - phi(impostor)` is exactly a
linear RankNet; its coefficient vector, rescaled so the span-coverage coefficient
is 1.0, is a candidate weight vector. Where coordinate ascent optimises the metric
we actually report, this optimises the *ranking* the metric is a proxy for — so
the interesting question is whether the two objectives disagree.

Like change 12, this treats `popularity_weight` as a tunable feature weight,
setting aside the house rule that priors are set "by reasoning, not sweeping"
(`signal_descriptions.md` §5). That rule is why `popularity_weight` is 0.4 and not
its fitted value; this doc is measuring what the fit *wants*, not overriding the
rule.

---

## 2. Method

### 2.1 Instrumented snapshot pass

`evaluator/local_evaluator.py:evaluate()` is frozen. `snapshot_pass()` in
`tools/fit_weights_pairwise.py` re-implements **only its outer per-sample loop**,
importing every helper verbatim (`materialize_hidden_fields`, `initial_message`,
`customer_reply`, `coarse_category`, `normalize_recommendations`, `MAX_TURNS`,
`TOP_K`). It drives the real `Agent` and monkey-patches `src.rerank.rerank` /
`starter.agent.rerank` with a spy that records, on every turn, the **pre-rerank
candidate pool** `[(asin, retrieval_score), …]` plus `state.opening`,
`full_text()`, `focused_text()`, `query_spans()`, `query_pair_spans()`. After each
turn that emits a non-empty slate, it stores a `Snapshot(opening, full, focused,
spans, pairs, pool, target)` — the exact inputs `rerank()` saw, frozen before the
next turn mutates the state.

**`--verify` gate** (must pass before any fit): `snapshot_pass` also rebuilds the
per-session records `evaluate()` builds (`hit`, `first_hit_turn`, `best_rank`,
`reciprocal_rank`) and asserts they match `evaluate()`'s output **exactly** and
that `scalar_from_sessions()` of the two agrees to < 1e-9. Result:

```
[verify plain]   session-record mismatches=0  scalar delta 0.00e+00   -> PASS
[verify default] session-record mismatches=0  scalar delta 0.00e+00   -> PASS
```

(The residual vs `evaluate()`'s published `recommended_technical_score` is
5.6e-8, purely its `round(…, 6)` on the intermediate metrics — the session
records themselves are bit-identical.)

### 2.2 Offline features

For each snapshot: `head = pool[:300]`, `top_norm = max(retrieval_score in head)`.
`phi(asin)` recomputes the eight features exactly as `rerank()` does —
`coverage`/`pair_coverage` by the same padded word-bounded substring test and the
same `1 + 0.12·wordcount` bonus, `retrieval_score/top_norm`, `_popularity`,
`_facet_agreement(extract_query_facets(full), extract(product))`,
`_category_match` / `_tail_match` against a `SimpleNamespace(opening=…)`,
`_facet_conflicts(extract_query_facets(focused), extract(product), text)`. The
feature helpers are pure, so this is the reranker's own arithmetic with no agent
in the loop. **Approximation:** features are computed for the target plus only the
**top-20** non-target asins of the depth-300 head (by retrieval order), not all
299 — a truncation that keeps the pairs to the impostors the target actually
competes with, at the cost of ignoring deep-pool negatives.

Feature index order: `0 span_cov, 1 pair_cov, 2 retr, 3 pop, 4 facet, 5 cat,
6 tail, 7 conflict`. Index 0 is the reference.

### 2.3 The "no per-candidate labels" objection, head-on

`ideas.md` idea 4 rejected logistic regression for exactly this: *"the
'known-correct' sessions give no per-candidate labels."* True — the data gives
**one gold product per session and no graded relevance for anything else**. So
the negatives here are **synthesised**: gold vs each of the top-20 non-gold asins
in the pool.

**The risk** is a synthesised negative that is actually a near-perfect substitute
for the gold — same category, same disclosed facets, a product the customer would
have accepted. Training the model to rank the gold strictly above it teaches a
distinction that isn't real, and the coefficients absorb whatever spurious feature
difference happens to separate that pair.

**The mitigations:**

1. **Shown-slate turns only.** A snapshot is taken only on a turn that emitted a
   slate — i.e. after the customer has disclosed constraints and retrieval has
   filtered the pool to those constraints. Every top-20 impostor already matches
   the disclosed spans/facets; the negatives are hard by construction, but they
   are also all genuine non-answers (the session did not end on them).
2. **Rescale, don't trust magnitudes.** Only the *direction* of the coefficient
   vector is used, rescaled to `span_cov = 1`. Weights the raw fit drives
   negative are clamped to 0 (and reported).
3. **The plateau + hard-set gate do the real filtering.** The offline fit is
   never shipped on its own recognisance: its rounded vector is line-searched on
   the actual evaluator (±50% per weight) and one-shot gated on holdout + the
   adversarial hard set. A vector that won only because of a bad synthesised pair
   shows up as a knife-edge on the plateau or a hard-set regression.

### 2.4 The fit

For each snapshot and each of ≤20 negatives, `d = phi(target) - phi(neg)`. Stack
`X = [d; -d]`, `y = [1…; 0…]`, fit
`LogisticRegression(fit_intercept=False, C=C, max_iter=5000,
class_weight="balanced")`. The coefficient vector `c` (length 8) is converted:
`scale = c[0]`; `w = c / scale`; then
`popularity = max(0, w[3])`, `retrieval = max(0, w[2])`, `facet = max(0, w[4])`,
`category = max(0, w[5])`, `tail = max(0, w[6])`, `pair = max(0, w[1])`,
`facet_conflict = max(0, -w[7])` (the conflict term is subtracted, so a healthy
fit has `w[7] < 0`).

`C` is selected once, on iteration 0's pairs, as the value whose resulting weight
vector scores best on **dev** (dev is the fitting set — this is allowed), then
reused for the weight-dependent re-fits.

### 2.5 Iterate

The transcript is weight-dependent (the session ends at the first hit; the
confidence gate reads scores), so `w_0` is fit on snapshots taken at the baseline
weights, then `snapshot_pass(w_{i-1}) → refit → w_i` until
`‖w_i - w_{i-1}‖ / ‖w_{i-1}‖ < 0.05` or 3 iterations. Run separately for
`variant="plain"` (`use_router=False`, `FixedPolicy`) and `variant="default"`
(shipped pipeline).

---

## 3. The fit

Both variants: `C = 0.1` selected (the **most** regularised of `{0.1, 1.0,
10.0}` — smaller `C` is stronger L2 — gave the best dev score at iteration 0:
plain 0.9483 vs 0.9438 / 0.9467; default 0.9497 vs 0.9458 / 0.9489), then reused
for the weight-dependent re-fits. Every fit converged inside 3 iterations.

### 3.1 `variant = plain` (`use_router=False`, `FixedPolicy`)

| iter | transcript from | popularity | retrieval | facet | category | tail | pair | conflict | dev score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | 1.419 | 0.000 | 0.178 | 0.337 | 0.640 | 0.190 | 0.000 | 0.948333 |
| 1 | w_0 | 1.334 | 0.000 | 0.258 | 0.290 | 0.650 | 0.196 | 0.089 | 0.947875 |
| 2 | w_1 | 1.366 | 0.000 | 0.253 | 0.289 | 0.646 | 0.197 | 0.036 | 0.946167 |

`‖w_1 - w_0‖/‖w_0‖ = 0.096`, `‖w_2 - w_1‖/‖w_1‖ = 0.041 < 0.05` → **converged at
iteration 2.** `w_final` = `pop 1.366, retrieval 0.000, facet 0.253, category
0.289, tail 0.646, pair 0.197, conflict 0.036`.

**Weights the raw fit drove negative (all clamped to 0):**

- **`retrieval_weight`** — every C, every iteration wants it *strongly* negative
  (`w[2]` = −0.55 at C=0.1, −0.74 at C=1.0, −0.81 at C=10.0). This is the §10
  near-miss anatomy made visible to the fit directly: among the impostors the
  target competes with, the normalised retrieval score points at the *impostor*
  (BM25 length normalisation favours the thin listing), so the pairwise objective
  concludes the retrieval feature is anti-correlated with relevance in this pool
  and would subtract it. `retrieval_weight` can't go negative in `RerankConfig`
  interpretation, so it clamps to 0 — the fit's way of saying "ignore retrieval".
  Change 12's coordinate-ascent argmax reached the milder `retrieval 0.1`.
- **`facet_conflict`** — `w[7]` sits within ±0.05 of zero and is sign-unstable:
  at iteration 0 the fit wants it *positive* (`w[7]` = +0.049), i.e. conflict
  should *raise* the score — incoherent for a term that is subtracted; by
  iterations 1–2 it is a healthy but tiny negative (−0.09, −0.04). Rescaled and
  clamped, `facet_conflict_weight` lands at 0.00–0.04. The pairwise signal does
  not support a conflict penalty; change 12's argmax also zeroed it.
- **`facet_weight`** — wanted slightly negative at the higher `C` values
  (−0.04 at C=1.0, −0.17 at C=10.0 for `plain`); at the chosen C=0.1 it is a
  small positive (0.18–0.26).

**The offline↔online disagreement, visible already in the trajectory:** the
pairwise (offline) loss converges monotonically, but the **real-evaluator dev
score falls every iteration** — 0.94833 → 0.94788 → 0.94617. The iterate loop is
walking toward weights the offline objective likes better and the actual metric
likes slightly *less*. Iteration 0's vector (fit on the baseline transcript, before
any weight-dependent drift) is the best of the three on the real metric. The
plateau line-search (§4) shows where else the two diverge.

### 3.2 `variant = default` (shipped pipeline)

| iter | transcript from | popularity | retrieval | facet | category | tail | pair | conflict | dev score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | 1.387 | 0.000 | 0.183 | 0.318 | 0.663 | 0.218 | 0.000 | 0.949667 |
| 1 | w_0 | 1.291 | 0.000 | 0.239 | 0.280 | 0.620 | 0.179 | 0.097 | 0.949375 |
| 2 | w_1 | 1.296 | 0.000 | 0.242 | 0.275 | 0.606 | 0.173 | 0.040 | 0.947667 |

`‖w_1 - w_0‖/‖w_0‖ = 0.102`, `‖w_2 - w_1‖/‖w_1‖ = 0.040 < 0.05` → **converged at
iteration 2.** `w_final` = `pop 1.296, retrieval 0.000, facet 0.242, category
0.275, tail 0.606, pair 0.173, conflict 0.040`.

Same picture as `plain`: `C = 0.1` selected (C=1.0 → dev 0.945750, C=10.0 →
0.948875); `retrieval_weight` wanted −0.53 to −0.84 across all C, clamped to 0;
`facet_conflict` near zero and sign-unstable, clamps to 0.04; the real-evaluator
dev score again **falls each iteration** (0.94967 → 0.94938 → 0.94767) while the
offline loss converges. The two variants land within rounding of each other —
the router / turn-2 gate does not change what the pairwise objective wants.

---

## 4. Rounding + plateau

Snapped to the grid `{0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.2, 1.5,
2.0}` (nearest):

| vector | popularity | retrieval | facet | category | tail | pair | conflict |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain `w_final` (raw) | 1.366 | 0.000 | 0.253 | 0.289 | 0.646 | 0.197 | 0.036 |
| **plain — rounded** | **1.5** | **0.0** | **0.3** | **0.3** | **0.5** | **0.2** | **0.05** |
| default `w_final` (raw) | 1.296 | 0.000 | 0.242 | 0.275 | 0.606 | 0.173 | 0.040 |
| **default — rounded** | **1.2** | **0.0** | **0.2** | **0.3** | **0.5** | **0.2** | **0.05** |
| — shipped defaults | 0.4 | 1.0 | 0.3 | 0.4 | 0.8 | 0.8 | 0.4 |
| — change 12 dev argmax | 0.8 | 0.1 | 0.5 | 0.4 | 1.2 | 0.8 | 0.0 |

### Plateau — plain rounded vector, dev score at ±50% per weight

Base (rounded vector) dev = **0.946000**. Baseline (shipped) dev = 0.941757.

| weight (rounded value) | ×0.5 | ×1.5 | reading |
|---|---:|---:|---|
| popularity (1.5) | 0.948250 | 0.948292 | **both sides better than base** — the rounded value sits in a dev *dip*, not on a plateau |
| retrieval (0.0) | 0.946000 | 0.946000 | flat (0·f = 0) |
| facet (0.3) | 0.947708 | 0.947042 | both sides better |
| category (0.3) | 0.946167 | 0.945833 | ~flat |
| tail (0.5) | 0.944833 | 0.946000 | ×1.5 (→0.75, near the shipped 0.8) better; ×0.5 worse |
| pair (0.2) | 0.945833 | 0.946167 | ~flat |
| conflict (0.05) | 0.946208 | 0.947708 | ×1.5 better |

**This is the offline↔online disagreement stated numerically.** The rounded
pairwise vector scores 0.946 on the real dev metric, but the line-search says the
metric wants popularity *away* from 1.5 in either direction, facet lower, conflict
higher, tail higher — i.e. back toward the shipped values. The pairwise objective
and the technical score do not share a maximum. Iteration 0's vector (dev 0.9483,
the best point this method ever produced) was already discarded by the iterate
loop and by rounding.

**Knife-edge flag:** `popularity_weight` is a knife-edge / dip at the rounded
value — worse than both neighbours. By the house rule (`signal_descriptions.md`
"how weights are chosen" §3) that alone disqualifies the rounded vector as a
shipping candidate.

### Plateau — default rounded vector, dev score at ±50% per weight

Base (rounded vector) dev = **0.947667**. Baseline (shipped) dev = 0.942757.

| weight (rounded value) | ×0.5 | ×1.5 | reading |
|---|---:|---:|---|
| popularity (1.2) | 0.949083 | 0.949542 | **both sides better** — same dip as `plain` |
| retrieval (0.0) | 0.947667 | 0.947667 | flat |
| facet (0.2) | 0.947958 | 0.949375 | ×1.5 (→0.3, the shipped value) better |
| category (0.3) | 0.947500 | 0.947667 | ~flat |
| tail (0.5) | 0.946042 | 0.947333 | ×1.5 (→0.75, near shipped 0.8) better |
| pair (0.2) | 0.947667 | 0.947833 | ~flat |
| conflict (0.05) | 0.947667 | 0.949375 | ×1.5 better |

Identical shape to `plain`: the dev metric is **non-convex at the pairwise
vector** — `popularity` scores better at both 0.5× and 1.5×, and `facet` /
`tail` / `conflict` all improve when nudged **back toward the shipped values**.
The offline fit and the online metric disagree not just on magnitude but on
whether this point is even a local optimum: it is not.

---

## 5. Gate table

Holdout and hard are **one-shot gates**, run once on the final vectors, never
selectors. Dev is the fitting set.

| vector | dev (120) | holdout (80) | public (200) / hit | hard (96) / hit |
|---|---:|---:|---:|---:|
| baseline (shipped defaults) | 0.942757 | 0.914119 | 0.931302 / **200/200** | **0.802811** / .896 |
| pairwise **plain — raw argmax** | 0.946167 | 0.928396 | 0.939058 / **200/200** | 0.788155 / **.885** |
| pairwise **plain — rounded** | 0.946000 | 0.927896 | 0.938758 / **200/200** | 0.788099 / **.885** |
| pairwise **default — raw argmax** | 0.947667 | 0.929396 | 0.940358 / **200/200** | 0.790863 / **.885** |
| pairwise **default — rounded** | 0.947667 | 0.931458 | 0.941183 / **200/200** | 0.787664 / **.885** |

Read the hard column: every pairwise vector **regresses the adversarial set
0.8028 → 0.788 (−0.015) and loses a converted hit (0.896 → 0.885)** — the same
−0.016 regression, on the same distribution of thin/unreviewed targets, that
change 12's coordinate-ascent argmax produced and that kept it from shipping.
Public stays 200/200 throughout; dev/holdout/public all gain (+0.003 dev, +0.014
holdout, +0.0075 public over baseline).

Per-scenario hit/MRR, **rounded** vectors (baseline in parens):

| | public boundary | public browsing | public buying | public override | hard browsing | hard buying | hard override |
|---|---|---|---|---|---|---|---|
| baseline | 1.00/0.860 | 1.00/0.910 | 1.00/0.899 | 1.00/0.899 | .906/0.732 | .896/0.727 | .875/0.673 |
| plain — rounded | 1.00/1.00 | 1.00/0.932 | 1.00/0.920 | 1.00/0.890 | **.875**/0.700 | .917/0.723 | **.812**/0.659 |
| default — rounded | 1.00/1.00 | 1.00/0.932 | 1.00/0.929 | 1.00/0.890 | **.875**/0.700 | .917/0.724 | **.812**/0.621 |

Public: every scenario ≥ baseline (boundary MRR 0.86 → 1.00 is the big move —
the boundary customer discloses nothing, so ranking falls to the priors and a
dominant `popularity` term helps there). Hard: `browsing` and `intent_override`
**lose hits** (.906 → .875, .875 → .812) — the thin-target regression.

**Stress harness** (unmodified agent, `tools/stress_harness.py`; robustness
probe over customers the official simulator cannot produce — **not** the official
score). Baseline row is `pop040` (= shipped defaults). `--verify` passes (the
un-stressed path reproduces the evaluator, |Δ| = 2e-7).

| customer | baseline | `fit_pairwise_plain` | `fit_pairwise_default` |
|---|---:|---:|---:|
| `official` (= evaluator) | 0.93130 / hit 1.000 | 0.93876 / 1.000 | 0.94118 / 1.000 |
| `paraphrase:heavy+browse-gated` | 0.68607 / hit **.815** | **0.74410** / **.840** | 0.73915 / .840 |

The second row is the interesting one, and it cuts the *other* way: under heavy
paraphrase + browse-gating — the closest proxy for a hostile private simulator —
verbatim span coverage collapses (`tok_cov` 0.83 → 0.44), the reranker falls back
on the priors, and the pairwise vector's dominant `popularity` / `category` terms
**recover 0.058 of score and a lost hit** (.815 → .840) over baseline. So the
pairwise fit trades **adversarial-target robustness** (the hard set, where its
targets are thin and popularity misleads) for **paraphrase robustness** (where
span coverage is gone and popularity is all that is left). Neither the house
rules nor this evaluator reward the second trade, but it is a real property of
the vector and worth recording.

---

## 6. Verdict

**Nothing here ships.** Two reasons, both pre-declared house rules:

1. **The hard-set gate fails.** Every pairwise vector — raw and rounded, both
   variants — regresses the adversarial set **0.8028 → 0.788 (−0.015)** and loses
   one converted hit (**0.896 → 0.885**). Same regression, same distribution of
   deliberately thin / unreviewed targets, that stopped change 12's argmax
   (−0.016). Public Hit@10 stays **200/200** for the rounded vectors, so that
   criterion alone does not disqualify them — the hard-set one does.
2. **The rounded vector is a knife-edge on `popularity_weight`** (§4): dev is
   worse at the rounded value than at both ±50% neighbours, on both variants.
   Change 12's bar was that `popularity` 0.1/0.3/0.4/0.5 are *all* ≥ baseline on
   all four splits; the pairwise vector does not sit on a plateau at all.

**Is the pairwise answer materially different from change 12's direct-metric
fit?** **No.** Both objectives — coordinate ascent on the metric, and pairwise
logistic on the ranking — reach the same conclusion the §10 near-miss anatomy
reached by hand: in the tie-break regime that holds the public headroom, the
normalised retrieval score points at the impostor and popularity points at the
target, so `retrieval` should go and `popularity` should dominate. The pairwise
fit is a *more extreme* version of the argmax — `popularity` 1.2–1.5 vs 0.8,
`retrieval` a hard 0 vs 0.1, `facet_conflict` ≈ 0 (same) — plus cuts to `pair`
(0.8 → 0.2) and `tail` (0.8 → 0.5) that coordinate ascent left alone. Those extra
cuts are the synthesised-negative risk surfacing: with `retrieval` gone and
`popularity` dominant, the model leans on whatever else separated target from
impostor in the *pairs it was shown*, and every shown pair is a constraint-matched
slate impostor where popularity genuinely is the tell — nothing tells it that
`pair` / `tail` matter on the sessions that never reach a slate.

**Does any pairwise vector beat `pop 0.4` on both evaluators without regressing
the hard set or public hit?** No. Every pairwise vector regresses the hard set
by ~0.015 with a lost hit — the exact trade change 12 declared out of bounds.
`popularity_weight = 0.4` remains the defensible shipped value; this method
re-derives *why* (the tie-break regime is real) without producing anything that
clears the bar.

---

## 7. Reproduction

```bash
git checkout kwongweng_fit_pairwise

# vendored gate harness
python3 tools/stress_harness.py --verify
python3 -m unittest tests.test_stress_harness

# loop fidelity
OMP_NUM_THREADS=1 python3 tools/fit_weights_pairwise.py --verify

# the fit (dev only; ~25 min shared box, prints the full trajectory + plateau)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  nice -n 10 python3 tools/fit_weights_pairwise.py --variant both \
  --out /tmp/pairwise_fit.json

# gates (rounded vectors are the fit_pairwise_* sweep rows; raw argmax = pass the
# w_final weights from /tmp/pairwise_fit.json into RerankConfig directly)
python3 tools/sweep.py --split dev      --configs fit_pairwise_plain,fit_pairwise_default
python3 tools/sweep.py --split holdout  --configs fit_pairwise_plain,fit_pairwise_default
python3 tools/sweep.py --split all      --configs fit_pairwise_plain,fit_pairwise_default   # public 200, hit check
python3 tools/sweep.py --dataset data/hard_set.jsonl --split all \
  --configs fit_pairwise_plain,fit_pairwise_default
python3 tools/stress_harness.py --customer official \
  --configs pop040,fit_pairwise_plain,fit_pairwise_default
python3 tools/stress_harness.py --customer paraphrase:heavy+browse-gated \
  --configs pop040,fit_pairwise_plain,fit_pairwise_default
```

All numbers in this doc are from logged runs on branch `kwongweng_fit_pairwise`
(based on `origin/main` `72b021f`) against `data/catalog.jsonl` /
`data/public_set.jsonl` / `data/hard_set.jsonl`. The fit's raw JSON (`w_final`,
per-iteration coefficients, plateau grid) is what `--out` writes; the tables
above are transcribed from it and from the sweep / gate / stress logs.

Verification: `python3 -m unittest discover -s tests -t .` → 94/94;
`tools/stress_harness.py --verify` → PASS (|Δ| 2.1e-7);
`tools/fit_weights_pairwise.py --verify` → PASS both variants (0 session-record
mismatches, scalar Δ 0).
