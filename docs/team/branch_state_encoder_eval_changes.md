# Branch `state-encoder-eval` — embedding work: one signal that clears noise, rerank does not

Branch `state-encoder-eval` (from `dynamic-state-slot`). Purpose: re-test the bi-encoder
and cross-encoder work from branches `dense_rerank` / `semantic-rerank-experiment` against
the live state machine and the paraphrase/browse-gated stress harness, neither of which
existed when those branches were originally measured.

**Headline: the S5 dense retrieval route (Step 3.4) is a genuine trade-off, confirmed
across three full-scale measurements - a small, consistent cost on the cooperative
simulator and holdout split (−0.004 to −0.007, both driven by the same browsing-MRR
dilution the pre-state-machine bi-encoder attempts documented), bought back several times
over under the realistic worst case this branch exists to defend against (+0.0263
overall, +0.0561 browsing, `heavy+browse-gated`). It currently fires unconditionally and
should not ship that way - the next step is gating it the way Step 3.2 already gates the
S6 rerank term. All four S6-rerank variants (ungated, two gated, a cleaner query-text
version) measured net-zero or worse at full scale. The cross-encoder is still blocked.**

---

## Status at a glance

**Infrastructure** (model fetch, the built artifact, per-turn latency) is done and
verified — detail in §2, not repeated here. This section is about the actual embedding
*signal*: there are two pipeline stages it can plug into, and two ways it can fire at
each stage.

| stage | mechanism | gating | status |
|---|---|---|---|
| **S6 rerank** — reorders the 300 candidates retrieval already found | `RerankConfig.dense_weight` — a cosine term added to the existing score | **none** — fixed weight (`1.0`), fires every turn unconditionally | ✅ measured, 21 sessions only — net **−0.016** (§3); not yet run at full scale |
| S6 rerank | same `dense_weight` term | **state-gated** on `state.over_general` (pool has stopped discriminating) | ✅ measured, 21 sessions — **identical to ungated**; the gate never closed on this subset (§3b) |
| S6 rerank | same `dense_weight` term | **state-gated** — withheld on `intent_track=="browsing"` | ✅ measured, 21 sessions — small overall gain, but structurally limited: browsing only lasts 2-3 turns before promoting to buying, so the gate has almost no turns left to act on (§3b, traced directly) |
| S6 rerank | `dense_query="slots"` — feed it `state.authoritative_text()` instead of the raw conversation | none (unconditional) | ✅ **measured at both 21 and 200 sessions** — 21: **+0.044**; 200: **+0.0023 (noise)**. The small-sample result did not hold (§3c) |
| **S5 retrieval** — searches the full 50,000-product catalog by meaning, builds the candidate pool | `RetrievalConfig.use_dense` — a 5th RRF-fused route alongside the BM25 ones | none — fires unconditionally | ✅ **measured, 200 sessions, three ways** (§3d): stressed **+0.0263** (clears noise), official **−0.0042**, holdout **−0.0065** (both within noise, same browsing-MRR-dilution mechanism). A confirmed trade-off, not shipped unconditionally |
| Cross-encoder rerank (S6, different model) | scores `(query, candidate)` pairs jointly | top-20 only, fired on state ambiguity signals (Plan Part 4) | ⬜ **blocked** — no reachable model source found, not attempted |

**In short: all four S6-rerank variants are net-zero-or-worse at full scale. The S5
retrieval route is the exception — a real, noise-clearing gain, measured directly at 200
sessions (no small-sample step this time, learning from §3c). It is not primarily a
recall fix as originally framed (`never_retrieved` barely moved); the gain shows up as
better ranking/conversion among targets the fused pool already reaches. This is the
strongest embedding result of the investigation and the one worth pursuing further.**

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
(`RerankConfig(dense_weight=1.0)`). **Stage: S6 rerank only** - reorders the pool S5
retrieval already built; the S5 dense *retrieval* route (`use_dense`) was not exercised
in this run. **Gating: none** - `dense_weight` is a fixed constant, applied on every
turn regardless of session state; no stagnation check, no pool-shape check, nothing
state-conditional. This subset and config are the closest reproduction of
`docs/team/dense_rerank.md`'s original +0.021 finding, run against the current
state-machine agent for the first time.

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

## 3b. Step 3.2 — state gating, both variants measured (21 sessions)

`RerankConfig.dense_gate_over_general` and `RerankConfig.dense_gate_exclude_browsing`,
implemented in `src/rerank.py` (`_dense_gate_open`) and threaded through
`starter/agent.py`'s `rerank()` calls via an explicit `track=track_name` (previously
unpassed - see the commit for a real precedence bug this caught before it shipped:
`state.intent_track` defaults to `"browsing"`, so treating it as an OR with `track` would
have silently vetoed an explicit `track="buying"`).

**`dense_rr_gate` (pool-shape gate alone):** scored **identical to `dense_rr_10`**, to
four decimal places. The "generic tail" subset is specifically curated so BM25 can't
discriminate - meaning `state.over_general` is essentially always true here by
construction. Not a fair test of this gate; it needs the full (mostly non-generic)
200-session set to actually open and close.

**`dense_rr_nobrowse` (withhold on browsing track):** overall **0.61437** vs `dense_rr_10`
**0.60841** (+0.006, still short of `router_on`'s 0.62456) - but the *browsing* scenario
bucket was **completely unaffected**, bit-for-bit identical to the ungated run, while the
*buying* bucket moved instead. Traced directly (a monkey-patched `_dense_gate_open`
recording every call across all 21 sessions, 152 gate checks total): every session,
regardless of dataset label, opens `intent_track=="browsing"` for exactly 2-3 turns, then
promotes to `"buying"` and stays there for the rest (often 8-12 more turns) - `track` and
`state.intent_track` were identical in all 152 checks, so there is no track/state-field
mismatch. The gate genuinely works; it just has almost no runway, because
`intent_track=="browsing"` is short-lived by design under this branch's own browsing
policy (the earlier Step 2b/2c fix extracts real disclosure fast, which is *why* it
promotes quickly). 4 of the 9 `buying`-labeled sessions *also* open ambiguously as
`intent_track=="browsing"` before promoting, which is where the buying-bucket movement
actually came from. `dense_rr_gate_nobrowse` (both gates together) == `dense_rr_nobrowse`
exactly, consistent with the pool-shape gate being a no-op on this subset.

**Reading it:** a gate keyed on live `intent_track` cannot meaningfully separate "browsing
turns" from "buying turns" here, because almost all of a session's turns are already
buying-track by the time ranking decisions matter. This is a structural limit on the
whole "exclude browsing" approach, not a bug to fix.

## 3c. Step 3.3 — `dense_query="slots"`, measured at 21 AND 200 sessions

Same `dense_weight=1.0`, unconditional; only the encoded query text changes from
`full_text()` (the raw, boilerplate-laden conversation) to `authoritative_text()` (the
state machine's compact list of active constraint values only).

| | router_on | dense_rr_slots | Δ |
|---|---|---|---|
| **21 sessions (generic tail)** | 0.62456 | 0.66876 | **+0.0442** |
| **200 sessions (full set)** | 0.76086 | 0.76312 | **+0.0023** |

The 21-session result looked like the first clear win of the whole investigation -
beating baseline on all three scenarios, including reversing the browsing collapse from
§3 entirely (hit rate matched baseline, MRR improved). **It did not hold at full scale.**
+0.0023 on 200 sessions is an order of magnitude under the ~0.02 noise floor this project
already treats as indistinguishable from zero. Per-scenario at 200: boundary +0.028,
browsing +0.0083 (small, same direction as the 21-session result, and browsing
`never_retrieved` improved 18/80 -> 17/80), buying −0.0016 (flat), **intent_override
−0.0122 (reversed from the 21-session subset's standout +0.040)**. The intent_override
reversal is the clearest tell that the small-sample result was a sample-selection
artifact, not a real effect.

**Reading it:** the mechanism (stripping boilerplate should give the embedding a cleaner
signal) is plausible and the browsing-direction consistency across both sample sizes is
mildly encouraging, but at the scale that can actually be trusted, this is a wash, not a
finding.

---

## 3d. Step 3.4 — S5 dense retrieval route, measured directly at 200 sessions

`RetrievalConfig.use_dense` (`dense_route_all` in `tools/sweep.py`, both tracks,
`weight_dense=0.6`, unconditional) - a fifth RRF-fused route alongside the three BM25
ones, searching the full 50,000-product catalog by meaning rather than reordering a
pool BM25 already built. Run directly at full 200-session scale
(`--customer paraphrase:heavy+browse-gated --configs router_on,dense_route_all`, no
small-sample step first - §3c's collapse from +0.044 to +0.0023 argued against trusting a
21-session read on anything further in this investigation).

| | router_on | dense_route_all | Δ |
|---|---|---|---|
| **overall (200)** | 0.76086 | **0.78718** | **+0.0263** |
| boundary (10) | 0.8783 | 0.8465 | −0.032 |
| **browsing (80)** | 0.6338 | **0.6899** | **+0.0561** |
| buying (80) | 0.8296 | 0.8408 | +0.0112 |
| intent_override (30) | 0.8773 | 0.8839 | +0.0066 |

**This clears the ~0.02 noise floor - the first result in the whole investigation that
does, at a sample size (80 browsing, 80 buying, 30 intent_override) large enough to
trust.** boundary moved negative, but N=10 there is small enough to be the exception, not
a contradiction.

**Not primarily a recall fix, despite the framing.** `never_retrieved` for browsing moved
18/80 -> **17/80** - one target recovered, not the dramatic fix "Step 3.4 recovers
targets BM25 never finds" implied. `pool_rank>100` (in the pool but deep) rose 7/80 ->
9/80, consistent with a couple of previously-absent targets being pulled into the pool
without yet surfacing to a scored position. The larger effect is elsewhere: browsing hit
rate 0.762 -> 0.825 and MRR 0.4867 -> 0.5337 - **ranking/conversion among targets the
fused pool already reached**, not recall of ones it didn't.

**Qualitatively different from the historical result.** The pre-state-machine
`dense_route.md` measurement recovered 0/10 missing targets and was net negative
(browsing MRR 0.855 -> 0.829, "a 4th RRF route full of semantically-plausible belts
dilutes the lexical ranking of a target that was already sitting at pool rank ~7"). Here
it is net positive and the dilution effect does not dominate - plausibly because this
branch's richer, policy-fixed disclosure (§1) gives the lexical routes a stronger signal
to fuse against, changing the RRF balance in the dense route's favour rather than against
it. Not confirmed, but consistent with §3c's parallel finding that query-text quality
matters more here than it did pre-state-machine.

**This is the strongest and most trustworthy result of the investigation so far** -
measured at full scale directly, on the exact scenario this whole branch exists to
stress, and it holds up where every S6 rerank variant did not.

### Robustness: cooperative simulator + holdout split

Two follow-up checks, both at trustworthy sample sizes:

| customer / split | router_on | dense_route_all | Δ |
|---|---|---|---|
| `official` (200, cooperative) | 0.92349 | 0.91931 | **−0.0042** |
| holdout (80, cooperative) | 0.9149 | 0.9084 | **−0.0065** |
| `heavy+browse-gated` (200, stressed) | 0.76086 | 0.78718 | **+0.0263** |

**Both cooperative checks agree, and land on the same mechanism.** Small, consistent,
within-noise cost - and in both, it's driven by the *same* thing: browsing MRR drops
(official: 0.8715 -> 0.8343; holdout: 0.87 -> 0.80). Hit@10 is unchanged at 1.000 in both
- no recall is lost, only some ranking precision. This is exactly the "dilutes good
lexical matches when the pool is already correct" pattern `dense_rerank.md` /
`dense_route.md` documented for the pre-state-machine agent, just smaller in magnitude
here (there: browsing MRR 0.855 -> 0.829 pre-state-machine cooperative; here: 0.87 ->
0.83-0.80).

**Verdict: this is a genuine, coherent trade-off, not noise on one side and a fluke on
the other.** The embedding costs a small, consistent amount when the lexical routes are
already working (cooperative/holdout) and buys a real, noise-clearing gain specifically
when they are not (paraphrased + browse-gated). That is the shape a *gated* signal is
for - `use_dense` currently fires unconditionally (§3d intro); Step 3.2 already built
`_dense_gate_open` for the S6 rerank term (`state.over_general`,
`intent_track`-exclusion) but it was never wired to the S5 retrieval route. Given this
trade-off, that extension - fire `use_dense` only when the state machine's own signals
say lexical retrieval is struggling - is the natural next step, not shipping
`dense_route_all` unconditionally. It should stay flag-gated and off by default (which is
already how it ships: `RetrievalConfig.use_dense: bool = False`) until that gate exists
and is measured.

---

## 4. What has not been checked

- **Gating `use_dense` on the state machine's own signals** (`state.over_general`,
  `intent_track`), the way `_dense_gate_open` already does for the S6 rerank term (§3b) -
  motivated directly by the robustness result above: `dense_route_all` fires
  unconditionally and has a real (if small) cooperative cost, so a gate that gives it up
  only when word-matching has visibly stopped working is the natural next step. Not
  designed or coded for the retrieval route yet.
- **Whether §3d's result is stable across a second run** - RRF fusion and ONNX inference
  are deterministic here in principle, but this has not been independently re-run to
  confirm, only reasoned about.
- **Whether combining §3d (S5 retrieval) with §3c (slots query text) does better than
  either alone** - untested. §3d used `full_text()` for its query (the default); §3c
  showed query-text quality plausibly matters more on this branch than it used to.
- **Full 200-session comparison of the ungated S6 term** (`dense_rr_10`, no
  `--targets generic` filter) - the original §3 result (net −0.016) has only ever been
  measured on the 21-session subset. Given how differently `dense_rr_slots` behaved at
  the two scales (§3c), this specific number should not be assumed to hold either way.
- **Plan Step 3.2 at full scale** - both gate variants (§3b) are 21-session-only; the
  pool-shape gate specifically needs the full set to be a fair test at all, since the
  generic-tail subset leaves it almost always open.
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

# §3 (21 sessions) and §3b's gate variants
python3 tools/stress_harness.py --customer paraphrase:heavy+browse-gated --targets generic \
    --configs router_on,dense_rr_10,dense_rr_gate,dense_rr_nobrowse,dense_rr_gate_nobrowse,dense_rr_slots

# §3c at full scale (slow - ~30-45 min on this environment's CPU)
python3 tools/stress_harness.py --customer paraphrase:heavy+browse-gated \
    --configs router_on,dense_rr_slots

# §3d at full scale - the one result that clears the noise floor
python3 tools/stress_harness.py --customer paraphrase:heavy+browse-gated \
    --configs router_on,dense_route_all

# §3d robustness: cooperative simulator + holdout split
python3 tools/stress_harness.py --customer official --configs router_on,dense_route_all
python3 tools/sweep.py --split holdout --configs router_on,dense_route_all
```

Everything above is off by default (`dense_weight=0.0`, `use_dense=False`, both
`dense_gate_*` fields `False`); the official evaluator and the full test suite are
unaffected regardless of artifact presence - `python3 -m evaluator.local_evaluator` still
reads **0.923487**, `python3 -m unittest discover -s tests -t .` still passes 123/123.
