# S6 rerank weights — Method 4: non-linear (LambdaMART) ranker on whole-catalog synthetic sessions

Branch `kwongweng_fit_lambdamart`. Exploratory. **Nothing here is proposed for
shipping** unless the gate table below clears every bar the four prior methods
failed. This is the fourth independent attempt to beat the shipped linear
reranker's weight vector, and the first that adds (a) non-linear model capacity
and (b) a training set ~400× the 120-session dev split, drawn from the whole
50,000-product catalog including the stress-harness's non-cooperative customers.

Companion tooling: `tools/lm_targets.py` (seed-target list), `tools/lm_generate.py`
(synthetic session specs), `tools/lm_snapshot.py` (rollout + per-turn ranking-query
snapshot), `tools/fit_weights_lambdamart.py` (the fit + the same-data linear
control), `tools/lm_explain.py` (importances + interaction extraction),
`tools/fit_common.py` (shared gate helpers, identical across the method branches),
`tools/stress_harness.py` (vendored from `dense_rerank`). Sweep rows
`fit_lm_additive` / `fit_lm_replace`.

---

## 1. Problem

`src/rerank.py` scores each candidate as a linear weighted sum of eight features:

```
total = 1.00*span_coverage + pair_weight*pair_coverage + retrieval_weight*(retr/top)
      + popularity_weight*popularity + facet_weight*facet + category_weight*category
      + tail_weight*tail - facet_conflict_weight*conflict
```

`span_weight` is the definitional unit (fixed 1.0); the other seven are free.
Four methods have tried to learn them, all fit on the 120-session dev split:
coordinate ascent (change 12), pairwise logistic / linear RankNet (Method 1),
Bayesian optimization (Method 2), CV/bootstrap coordinate ascent (Method 3).
**Every one that raises dev/holdout/public regresses the adversarial hard set by
−0.015 to −0.022**, because they all cut `retrieval_weight`, and the hard set's
targets are thin/unreviewed products in homogeneous clusters where popularity is
neutral and span coverage saturates — BM25 retrieval order is the only signal
that can rank them. The dev split contains almost none of that distribution.

A linear model has ONE global coefficient per feature. It must compromise between
the **tie-break regime** (public near-misses: wants `popularity↑, retrieval↓`)
and the **thin-target regime** (hard set: wants `retrieval` to matter). This
method tests two hypotheses:

1. **Non-linear capacity.** A tree ensemble can express *"if span-coverage gap ≈ 0
   and rating_number high → weight popularity; if rating_number < 5 → weight
   retrieval rank"* — serving both regimes at once.
2. **Big, diverse training data.** A synthetic session for (almost) every catalog
   product as target, including sessions run through the stress-harness's
   non-cooperative customer models, so the training distribution contains the
   thin-target / paraphrased cases the 120 dev sessions lack.

**Honest expected outcome, stated up front:** the oracle-reranker ceiling
(`rerank_signals.md` §9) is only **+0.043 public / +0.084 hard** over baseline —
that bounds ANY reranker. ~100 of 142 public near-misses already rank the target
#1. ~25 of the rest are homogeneous clusters where target and impostor features
are *identical* (a tree cannot split on features that do not differ). A large
chunk of the hard-set / stress loss is retrieval **recall** — the target never
enters the pool — which no reranker touches. So the realistic result is: helps
the tie-break regime and the stress `official` cell, probably does not fix the
hard set. **A clean negative result is the expected and publishable outcome.**

---

## 2. Method

### 2.1 Dependency path

`lightgbm` and `xgboost` were not installed. `pip install lightgbm` **succeeded**
(a 3.5 MB pure-Python wheel over the system numpy/scipy), so the primary path is
`lightgbm.LGBMRanker(objective="lambdarank")`. The `HistGradientBoostingClassifier`
fallback on the pairwise-difference stack was not needed. `pyarrow` and `shap`
also installed (used for Parquet output and the interaction analysis).

### 2.2 LambdaMART / LGBMRanker primer

LambdaMART is a gradient-boosted regression-tree ensemble trained to optimise a
ranking metric (here NDCG@10). Instead of a pointwise loss, each boosting round
fits trees to *lambda gradients* — per-document forces derived from every
mis-ordered pair within a query, weighted by how much swapping that pair would
change NDCG. Queries are groups; here a group is one `(session_id, turn)` slate,
its documents are the depth-300 retrieval head, and the single relevant document
(`label=1`) is the session's target (everything else `label=0`, `label_gain=[0,1]`).
The trees split on the 14 features below; the model output is an additive score
per candidate, used exactly like the linear `total`.

### 2.3 Label synthesis and its risk

The data gives ONE gold product per session and NO graded per-candidate labels.
Negatives are every non-target in the pool. The risk — identical to Method 1's —
is that a higher-capacity model overfits the *simulator's generative quirks* more
thoroughly, not less. Excluding the 294 public/hard targets from the seed list
(`tools/lm_targets.py --assert-no-leak`, verified 0 leak) stops answer-key
memorisation; it does **not** stop learning cooperative-simulator ranking cues
such as "the target is always the reviewed product" or "cut retrieval_weight".

### 2.4 Distribution-mismatch mitigation (mandatory)

~22 % of training sessions (`tools/lm_generate.py`, `STRESS_FRACTION`) are run
through the `dense_rerank` stress-harness `StressCustomer`, disjoint from the
cooperative set, cycling four specs with the scenario pinned so the stressor
bites: `paraphrase:heavy` (any scenario), `browse-gated` (browsing),
`decoy` (intent_override), `paraphrase:heavy+browse-gated` (browsing). In these
sessions verbatim span coverage collapses and BM25 retrieval order is the signal
that survives, so "cut retrieval_weight" is punished during training rather than
rewarded.

### 2.5 Instrumented snapshot pass

`evaluator/local_evaluator.py:evaluate()` is frozen. `tools/lm_snapshot.py`
re-implements only the per-session loop (the same faithful copy the stress
harness uses — `--verify` reproduces `evaluate()`'s scalar to **delta 0**, both
agent configs, 0 session-record mismatches). It drives the real `Agent` and
monkey-patches `src.rerank.rerank` / `starter.agent.rerank` with a spy that
records, on every slate turn, the pre-rerank pool `[(asin, retrieval_score), …]`
plus `state.opening / full_text() / focused_text() / query_spans() /
query_pair_spans()`.

A snapshot is taken only when (a) the turn emitted a slate, (b) the reranker
actually ran (non-empty `query_spans`; `rerank()` returns the pool untouched
otherwise), and (c) for intent_override, the override has already landed. The
first and last such turn per session are kept (the under-informed and
fully-informed regimes).

### 2.6 Offline features

For each snapshot, candidates are the depth-300 head; the top 150 by retrieval
rank plus the target are featurized (negatives past rank ~150 carry no span/pair
coverage — trivial negatives that only inflate the set). Per candidate:

| # | feature | note |
|---|---|---|
| 0 | `f_span_cov` | `Σ 1 + 0.12·wordcount` over matched query spans — rerank's own loop |
| 1 | `f_pair_cov` | matched association-preserving pair spans |
| 2 | `f_retr_norm` | `retrieval_score / max_in_head` |
| 3 | `f_popularity` | `_popularity(product)` |
| 4 | `f_facet_agree` | `_facet_agreement(extract_query_facets(full), extract(product))` |
| 5 | `f_category` | `_category_match` |
| 6 | `f_tail` | `_tail_match` |
| 7 | `f_facet_conflict` | `_facet_conflicts(extract_query_facets(focused), …)` |
| 8 | `f_rating_number` | raw review count |
| 9 | `f_average_rating` | raw mean rating |
| 10 | `f_text_len` | product blob length |
| 11 | `f_span_gap_to_max` | `max(f_span_cov over full 300 head) − f_span_cov` |
| 12 | `f_retr_rank` | 1-indexed retrieval position |
| 13 | `f_pool_size` | head size |

Features 0–7 are recomputed **bit-identically** to `rerank()`'s own arithmetic:
`tools/lm_snapshot.py --verify`'s feature-fidelity check scores the featurized
subset with the shipped weights and compares to `rerank()`'s captured per-candidate
totals — **worst |Δtotal| = 0.0 over 40 pools, 0 order mismatches.** Features 8–13
are the extra raw / positional signals a tree can split on but the linear reranker
cannot express.

Skipped-and-counted: queries whose target is outside the depth-300 head (a
retrieval recall miss no reranker can fix) — the per-batch rate is the recall
ceiling, reported in §3.

---

## 3. Data

<!-- FILLED FROM queries.parquet.meta.json + lm_generate output AFTER Stage B -->

| | sessions | | queries | rows |
|---|---:|---|---:|---:|
| cooperative | TBD | | | |
| stress paraphrase:heavy | TBD | | | |
| stress browse-gated | TBD | | | |
| stress decoy | TBD | | | |
| stress paraphrase:heavy+browse-gated | TBD | | | |
| **total** (× 2 agent configs) | TBD | | TBD | TBD |

### Target-not-in-pool (retrieval recall ceiling) — per batch / spec

<!-- FILLED FROM meta.json recall_miss_by_batch_spec_scenario -->

| batch / spec | queries | target not in depth-300 head | rate |
|---|---:|---:|---:|
| coop / official | TBD | TBD | TBD |
| stress / paraphrase:heavy | TBD | TBD | TBD |
| stress / browse-gated | TBD | TBD | TBD |
| stress / decoy | TBD | TBD | TBD |
| stress / paraphrase:heavy+browse-gated | TBD | TBD | TBD |
| **overall** | TBD | TBD | TBD |

---

## 4. Training

<!-- FILLED FROM fit_summary.json AFTER Stage C -->

Session-grouped 85/15 train/val split, stratified by `(scenario_type, batch)`;
a session's queries never cross the split. Hyper-parameters:
`objective="lambdarank"`, `metric="ndcg"`, `eval_at=[10]`,
`lambdarank_truncation_level=10`, `label_gain=[0,1]`, `learning_rate=0.05`,
`num_leaves ∈ {15, 31}`, `min_child_samples≈50`, `n_estimators=1000`, early
stopping 50 rounds on val NDCG@10.

| | val NDCG@10 start | val NDCG@10 best | best_iter |
|---|---:|---:|---:|
| LambdaMART (num_leaves 31) | TBD | TBD | TBD |

**Same-data linear control** — Method 1's exact pairwise `LogisticRegression` on
`phi(target) − phi(neg)` over the 8 shipped features, fit on this 400×-larger set:

| | popularity | retrieval | facet | category | tail | pair | conflict |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear control (this data) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Method 1 (dev only) | 1.30 | 0.00 | 0.24 | 0.28 | 0.61 | 0.17 | 0.04 |
| shipped defaults | 0.4 | 1.0 | 0.3 | 0.4 | 0.8 | 0.8 | 0.4 |

---

## 5. Gate table

Holdout (80), public (200) + hit, hard (96) + hit, and the two stress cells are
**one-shot gates**, never selectors. Dev (120) is allowed as a fitting-set check.

<!-- FILLED FROM tools/fit_common.gate + .stress AFTER Stage C -->

| config | dev | holdout | public / hit | hard / hit | stress `official` / hit | stress `ph+bg` / hit |
|---|---:|---:|---:|---:|---:|---:|
| baseline (shipped) | 0.942757 | 0.914119 | 0.931302 / **200/200** | 0.802811 / .896 | 0.93130 / 1.000 | 0.68607 / .815 |
| change-12 argmax (ref) | TBD | TBD | TBD | TBD | TBD | TBD |
| linear control (this data) | TBD | TBD | TBD | TBD | TBD | TBD |
| **LambdaMART additive** | TBD | TBD | TBD | TBD | TBD | TBD |
| **LambdaMART replace** | TBD | TBD | TBD | TBD | TBD | TBD |

`model_weight` for the additive row bracket-tuned on dev only (§ reproduction).
`model_weight = 0.0` reproduces baseline `evaluate()` **exactly** (dev 0.942757,
verified).

---

## 6. Model interpretation

<!-- FILLED FROM tools/lm_explain.py AFTER Stage C -->

Gain importances:

| feature | gain |
|---|---:|
| TBD | |

**The interaction rule the model found** (the deliverable even on a negative
result): TBD — e.g. "popularity matters only when `f_span_gap_to_max ≈ 0` and
`f_rating_number > N`; below that, `f_retr_rank` dominates."

---

## 7. Verdict

<!-- FILLED AFTER Stage C -->

TBD. The questions to answer:

- Does non-linear + big data beat `popularity_weight = 0.4` on BOTH evaluators
  without regressing the hard set or public hit?
- If there is a gain, is it **capacity** (LambdaMART beats the linear control on
  the same data) or **data** (both beat their small-data versions)?
- What is the target-not-in-pool rate (the recall ceiling that bounds this)?

---

## 8. Reproduction

```bash
git checkout kwongweng_fit_lambdamart
pip install lightgbm pyarrow        # shap optional, for §6

SCRATCH=/tmp/lm    # anywhere outside the repo

python3 tools/stress_harness.py --verify
python3 tools/lm_snapshot.py --verify        # loop + feature fidelity

# Stage B — synthesise + snapshot (~2-4 h; detached)
python3 tools/lm_targets.py --assert-no-leak --out $SCRATCH/seed_targets.txt
python3 tools/lm_generate.py --all --out $SCRATCH/sessions.jsonl
OMP_NUM_THREADS=1 python3 tools/lm_snapshot.py --sessions $SCRATCH/sessions.jsonl \
    --variant both --out $SCRATCH/queries.parquet

# Stage C — train + gate
python3 tools/fit_weights_lambdamart.py --queries $SCRATCH/queries.parquet \
    --out-dir $SCRATCH/lm_model
export LM_MODEL_DIR=$SCRATCH/lm_model
python3 tools/sweep.py --split dev     --configs pop040,fit_lm_additive,fit_lm_replace
python3 tools/sweep.py --split holdout --configs pop040,fit_lm_additive,fit_lm_replace
python3 tools/sweep.py --split all     --configs pop040,fit_lm_additive,fit_lm_replace
python3 tools/sweep.py --dataset data/hard_set.jsonl --split all \
    --configs pop040,fit_lm_additive,fit_lm_replace
python3 tools/stress_harness.py --customer official \
    --configs pop040,fit_lm_additive,fit_lm_replace
python3 tools/stress_harness.py --customer paraphrase:heavy+browse-gated \
    --configs pop040,fit_lm_additive,fit_lm_replace
python3 tools/lm_explain.py --queries $SCRATCH/queries.parquet --model-dir $SCRATCH/lm_model
```

All numbers in this doc are from logged runs on branch `kwongweng_fit_lambdamart`
(based on `origin/main` `72b021f`).
