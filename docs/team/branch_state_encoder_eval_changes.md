# Branch `state-encoder-eval` — embedding work, so far: no clear signal

Branch `state-encoder-eval` (from `dynamic-state-slot`). Purpose: re-test the bi-encoder
and cross-encoder work from branches `dense_rerank` / `semantic-rerank-experiment` against
the live state machine and the paraphrase/browse-gated stress harness, neither of which
existed when those branches were originally measured.

**Headline: infrastructure is done and working. The embedding itself has one real
measurement, and it does not show a clear positive signal — mixed by scenario, net
negative overall, on a small sample.**

---

## Status at a glance

### ✅ Done

| | status |
|---|---|
| Model fetch working (GCS fallback, `huggingface.co` is blocked here) | done — commit `b6ff10c` |
| Embedding artifact built — all 50,000 catalog products encoded | done — checksummed, verified `available=True` |
| Per-turn latency measured directly (not estimated) | done — ~9 ms/turn, not a concern |
| **One** embedding-signal measurement (21-session generic-tail subset, S6 rerank term only) | done — **net −0.016, does not replicate the historical +0.021** |

### ⬜ Not done

| | status |
|---|---|
| Full 200-session version of the same comparison | **not run** |
| Step 3.2 — state-gated conditional `dense_weight` | **not designed, not coded** |
| Step 3.3 — cleaner query text (`dense_query="slots"`) | **not run** |
| Step 3.4 — S5 dense *retrieval* route (`use_dense`, recall not ranking) | **not run** |
| Cross-encoder (Part 4 of the plan) | **blocked — no reachable model source found**, not attempted |

**In short: the plumbing works end to end; the actual question — does the embedding
help — has one small, inconclusive, net-negative data point and four unrun next steps.**
Nothing below Section 2 has been decided; it's all still open.

---

## 1. What is *not* embedding work (context, not the ask)

Most of this branch's history is unrelated groundwork that had to happen first:

- Merged `origin/stress_harness` and `origin/dense_rerank` onto `dynamic-state-slot`
  (commits `75718bd`, `5443e8f`). Both source branches carried a dual-track `AgentConfig`
  surface (`buying_rerank`/`browsing_rerank`/`route_policies`) from the pre-state-machine
  `dual_tracking` branch; dropped in favour of the state machine's own `_route_for`/
  `intent_track`, since the two conflicted directly.
- The merge itself introduced a regression: browsing-track sessions lost their
  clarification policy and defaulted to `FixedPolicy`'s broad "other" question, which the
  harness's `browse-gated` customer never answers. Diagnosed and fixed in two attempts
  (commits `fa033de`, `417d1f4`) — landed at public **0.923487**, holdout **0.9149**,
  `heavy+browse-gated` **0.761**, browsing `never_retrieved` **18/80**. This is a policy
  (S4) fix, not an embedding change, and is the current baseline everything below is
  measured against.

Full detail on both: `docs/team/agent_changes.md` and the plan file this session worked
from (not committed - a local Claude Code plan artifact).

---

## 2. Embedding infrastructure — done, verified working

**Model fetch (commit `b6ff10c`).** `huggingface.co` is blocked in this environment (403
at the proxy). `tools/build_embeddings.py` previously only supported
`huggingface_hub.hf_hub_download`. Added a GCS fallback: Qdrant's public fastembed mirror
(`storage.googleapis.com/qdrant-fastembed/fast-bge-small-en-v1.5.tar.gz`, verified
reachable) ships exactly the files `_Encoder` needs - `model_optimized.onnx` +
`tokenizer.json` - with input/output names identical to the HF ONNX export
(`input_ids`/`attention_mask`/`token_type_ids` -> `last_hidden_state`, confirmed by
inspecting the model directly with `onnxruntime`). `GCS_SOURCES` maps known models to a
tarball URL and is used automatically; `--use-hf` forces the original path, `--source-url`
points at a different tarball. Downloads are cached under `<out>/_dl_cache` so building
both recipes (`v1`, `v1cat`) doesn't re-fetch the ~75 MB archive.

**Artifact built.** All 50,000 catalog products encoded (`BAAI/bge-small-en-v1.5`,
384-dim, CLS pooling, recipe `v1cat`) at 27 rows/s, ~31 minutes total.
`data/embeddings/bge-small-en-v1.5.v1cat.npz` (39.1 MB) +
`data/embeddings/bge-small-en-v1.5/` (model.onnx + tokenizer.json + meta.json). Both are
git-ignored per the existing convention (distributed as a release asset, same as
`catalog.jsonl.gz`) - **not committed**, must be rebuilt or fetched separately to
reproduce anything below.

Verified: `EmbeddingIndex.available is True`, vectors shape `(50000, 384)`, 50,000 asins.
Checksums logged per the plan's anti-silent-no-op guard (a missing artifact makes every
dense config silently score identical to BM25-only, which is exactly how an earlier
cross-encoder attempt went unmeasured):

```
npz sha1:       a62e1518f14b069b25c723d48cc9bbab642dc174
model.onnx sha1: 2217438696a8b98b5578ee5ff1d6117c6167595c
```

**Latency, measured directly on this artifact** (not estimated):

| operation | latency | fires when |
|---|---|---|
| `encode_query()` - one ONNX pass/turn | mean 5.0 ms, p95 6.1 ms, max 8.9 ms | `dense_weight>0` or `use_dense` |
| `search()` - matmul over the full 50k catalog | ~3.3 ms | S5 dense retrieval route on |
| `similarities()` - dot products over a 300-candidate pool | ~0.1 ms | S6 dense rerank term on |
| model load | ~6.7 s | once per `Agent()` construction (confirmed: `evaluate()` builds one Agent and reuses it across all 200 sessions - `evaluator/local_evaluator.py:306`), not per turn |

Total added cost with both stages on: **~9 ms/turn**. For contrast, the cross-encoder
(`ms-marco-MiniLM-L6-v2`, already measured and rejected pre-session) cost mean turn
30.7 ms -> 389.8 ms, p95 1347.8 ms (`docs/team/rerank_signals.md:646`) - a ~13x hit,
because it scores every candidate individually rather than encoding the query once.
**Latency is not a concern for the bi-encoder.**

---

## 3. Embedding signal — one measurement, mixed and net negative

**What was run:** `tools/stress_harness.py --customer paraphrase:heavy+browse-gated
--targets generic --configs router_on,dense_rr_10` - the 21-session "generic tail"
subset (all disclosed constraints are catalog-common words, so exact-match reranking has
the least to work with), comparing `router_on` (dense off) against `dense_rr_10`
(`RerankConfig(dense_weight=1.0)`, the S6 rerank-signal path only). This subset and
config are the closest reproduction of `docs/team/dense_rerank.md`'s original +0.021
finding, run against the current state-machine agent for the first time.

| | router_on | dense_rr_10 | Δ |
|---|---|---|---|
| **overall (21 sessions)** | 0.62456 | 0.60841 | **−0.0161** |
| buying (9) | 0.6912 | 0.7015 | +0.0103 |
| intent_override (5) | 0.7871 | 0.8267 | **+0.0396** |
| browsing (7) | 0.4229 | 0.3329 | **−0.0900** |

**Does not replicate the historical +0.021.** Net negative overall, and the pattern is
scenario-split rather than uniform: buying and intent_override both improve, browsing
collapses (hit@10 0.571 -> 0.429 - one session flipped out of seven).

**Reading it.** The historical +0.021 was measured on the pre-state-machine baseline,
where browsing sessions got almost no real disclosure (`FixedPolicy`'s broad "other"
question every turn - see §1). The embedding likely helped there simply by having
*something* to go on. Section 1's policy fix means browsing sessions now carry much
richer, accurate lexical content by the time reranking runs; the dense term may now be
diluting an already-strong lexical signal on that track specifically, rather than filling
a gap - the same "dilutes good exact matches" failure mode `dense_rerank.md` already
diagnosed for the full set, now appearing on one track instead of uniformly.

**Not a confident read.** N=21, 7 browsing sessions - one flip moves that scenario's hit
rate by ~14 points. This is a real, directionally-informative result, not a settled one.

---

## 4. What has not been checked

- **Full 200-session comparison** at this config (`--customer
  paraphrase:heavy+browse-gated --configs router_on,dense_rr_10`, no `--targets generic`
  filter) - needed to see whether the buying/override-up, browsing-down split from §3
  holds at a sample size large enough to trust.
- **Plan Step 3.2** - a state-gated conditional `dense_weight` (fire only when
  `state.over_general` / high `pool_entropy` / low `leader_margin` say lexical matching
  has stopped discriminating). Not designed or coded yet - the §3 result suggests it may
  need to be track-aware (avoid firing on browsing), not just pool-shape-aware, which the
  plan had not anticipated.
- **Plan Step 3.3** - `dense_query="slots"` (encode `state.authoritative_text()` instead
  of the raw conversation). Not run.
- **Plan Step 3.4** - the S5 dense *retrieval* route (`use_dense`, recovering
  `never_retrieved` targets rather than reordering the pool). Not run. Historical result
  (pre-state-machine) recovered 0 of 10 missing targets; untested against the current
  18/80 baseline.
- **Cross-encoder (Part 4 of the plan).** Blocked - no reranker ONNX model found on any
  reachable host (`bge-reranker-base` is not on the Qdrant GCS bucket that served the
  bi-encoder; every mirror probed - `hf-mirror.com`, `cdn-lfs.huggingface.co`,
  `modelscope` - is blocked same as `huggingface.co` itself). `github.com` and
  `raw.githubusercontent.com` are reachable, so a release-asset route is the open option,
  untried.

---

## Reproduce

```bash
# Rebuild the artifact (git-ignored, ~31 min on this environment's CPU)
python3 tools/build_embeddings.py --recipe v1cat

# Confirm it loaded correctly (guard against the silent-no-op failure mode)
python3 -c "from src.embed import load_embedding_index; i = load_embedding_index(); \
             print(i.available, i.vectors.shape)"

# The one measurement in §3
python3 tools/stress_harness.py --customer paraphrase:heavy+browse-gated \
    --targets generic --configs router_on,dense_rr_10
```

Everything in §2/§3 is off by default (`dense_weight=0.0`, `use_dense=False`); the
official evaluator and the full test suite are unaffected regardless of artifact
presence - `python3 -m evaluator.local_evaluator` still reads **0.923487**,
`python3 -m unittest discover -s tests -t .` still passes 117/117.
