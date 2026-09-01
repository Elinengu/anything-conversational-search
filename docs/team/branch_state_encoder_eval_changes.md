# Branch `state-encoder-eval` — embedding work: one signal that clears noise, rerank does not

Branch `state-encoder-eval` (from `dynamic-state-slot`). Purpose: re-test the bi-encoder
and cross-encoder work from branches `dense_rerank` / `semantic-rerank-experiment` against
the live state machine and the paraphrase/browse-gated stress harness, neither of which
existed when those branches were originally measured.

> **Status: the code this document measures no longer exists.** `src/embed.py`,
> `tools/build_embeddings.py`, the `dense_*` sweep rows and every `dense_*` /
> `use_dense` config knob were removed in change 20, because no configuration
> measured here ever cleared the noise floor. The commands below therefore only
> run against the branch this was written on. The document is kept as the
> measurement record — see `docs/team/agent_changes.md` change 20.

---

## 0. Plain-English: what is actually different from `main`, and why

This section assumes no prior recommendation-systems background. It answers one question:
if you diffed `main` against this branch, what would you actually be looking at?

`main` is the true starting point — it predates the state machine, the stress-testing
harness, and every line of embedding work. Everything in this branch is one of four
layers stacked on top of it, in the order they were built:

**Layer 1 — the state machine.** `main`'s agent treats every turn the same way: ask a
question, look at the answer, ask another question. This branch adds a small tracker
(`src/state.py`) that watches the conversation and classifies where it is — is the
customer's request still broad ("exploring"), narrowing down, converging on one answer, or
stuck ("stagnating")? It also tracks two other things per turn: `intent_track` — is this
customer actually trying to *buy* something, or just *browsing*? — and `over_general` —
have the last few answers stopped actually narrowing down the candidate list (the scores
for the top few products are all bunched together, i.e. word-matching has run out of
signal)? Nothing about search itself changes here; what changes is that the agent can now
*behave differently* depending on which of those states it's in — e.g. asking a more
pointed question once it detects it's stuck, or (see Layer 4) turning extra machinery on
only when a specific state fires. This is what makes items 3 and 4 below "conditional"
instead of "always on".

**Layer 2 — the stress-testing harness (`tools/stress_harness.py`).** `main` is evaluated
by a simulated customer who talks the way the product catalog is worded — they'll say
"leather" if the product listing says "leather". That's why `main` finds the right product
100% of the time: the words genuinely match. This branch adds a harness that simulates
*harder*, more realistic customers on top of the same 200 sessions:
  - **paraphrasing** (light/medium/heavy) — the simulated customer says "cowhide" instead
    of "leather", so a system that only matches exact words starts missing things;
  - **browse-gated** — a customer who is just browsing and won't proactively describe what
    they want; they only answer if you ask them a specific, on-target question. If you ask
    something broad ("tell me more"), they deflect.

  This harness is a *test*, not a feature — it doesn't run during scoring, it's how this
  branch measures whether a change actually holds up when the customer isn't perfectly
  cooperative. Running it against the state machine for the first time (Layer 1 had never
  been tested against it before this branch) surfaced a real bug: browsing-track customers
  had lost their targeted-questioning behavior in an earlier merge and were defaulting to
  a generic broad question the `browse-gated` customer refuses to answer. That's Layer 3.

**Layer 3 — a policy fix for browsing customers (`starter/agent.py`,
`_policy_for_state`).** Small, not embedding-related, but necessary before the embedding
question could be measured honestly: browsing-track sessions now get a policy tuned to ask
pointed questions early (`InfoGainPolicy(expected_broad_answers=4.0)`) instead of the
generic fallback. This recovered most of what the merge had broken (`heavy+browse-gated`
0.703 → 0.761) at a small, documented cost to the cooperative-customer score (§1 above has
the exact numbers). Everything the embedding work (Layer 4) is measured against uses this
already-fixed baseline, not the broken one.

**Layer 4 — embeddings (the actual ask of this investigation).** `main`'s search is
two stages: **retrieval** narrows the 50,000-product catalog down to ~300 candidates using
BM25 (word/keyword matching, with no notion of *meaning* — "leather" and "cowhide" are
unrelated strings to it); **rerank** then puts the most likely candidate on top, mostly via
exact-phrase matching. This branch adds a second way to search: a bi-encoder embedding
model (`bge-small-en-v1.5`) that converts text into a list of 384 numbers such that texts
with *similar meaning* land at nearby points, regardless of exact wording — so "leather"
and "cowhide" score as close. That gives it a way to survive paraphrasing that pure word
matching doesn't have. It can plug into either stage (`src/retrieval.py`,
`src/rerank.py`), and either unconditionally or gated behind Layer 1's state signals
(`over_general`, `intent_track`) so it only turns on when word-matching looks like it's
struggling. Every combination that was actually built and measured is in the table below
(§ "Status at a glance"); the short version is: turning it on inside rerank never clearly
helped, but turning it on inside retrieval — specifically gated to skip browsing-track
turns — produced a real, repeatable gain under the harder stress-tested customer while
costing nothing on the easy, cooperative one. It's implemented as an off-by-default flag,
not a new default behavior.

**In one sentence:** `main` is a two-stage word-matching search agent tested only against
a cooperative customer; this branch adds a state-tracking layer on top of it, a much
harder simulated customer to test against, a bug-fix the harder customer exposed, and an
optional (flag-gated, off by default) meaning-based search route that measurably helps
under that harder customer without hurting the easy one.

---

**Headline (revised after §3f/§3g - the earlier version of this line is superseded):
the embedding line closes. The S5 dense retrieval route gated to the buying track
(`dense_route_nobrowse`) was this investigation's one result that cleared the noise floor,
at +0.0257 on `heavy+browse-gated`. Three bugs ported in from a sibling branch (§3f) then
showed that gain was largely compensating for a broken lexical signal: with
`constraint_spans()` fixed, the same comparison re-run at 200 sessions gives
**+0.0042** - 16% of what it was - while the deterministic baseline it is measured against
rose +0.0098 for free. Official (−0.0002) and holdout (+0.0031) are unchanged by the fixes
and were always inside the ~0.02 noise floor. **No embedding configuration now clears
noise on any of the three checks**, and all four S6-rerank variants were already
net-zero-or-worse at full scale. The recommendation in §3e is withdrawn (§3g). The
cross-encoder remains blocked. The durable win from this work is the three bug fixes and
the tooling repair, not the model.**

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
| **S5 retrieval** — searches the full 50,000-product catalog by meaning, builds the candidate pool | `RetrievalConfig.use_dense` — a 5th RRF-fused route alongside the BM25 ones | **none** — fires unconditionally | ✅ measured, 200 sessions, three ways (§3d): stressed **+0.0263** (clears noise), official **−0.0042**, holdout **−0.0065**. A confirmed trade-off, not shipped unconditionally |
| S5 retrieval | same `use_dense` route | **state-gated** — withheld on `intent_track=="browsing"` | ⚠️ **re-measured after the §3f bug fixes (§3g)**: official **−0.0002**, holdout **+0.0031**, stressed **+0.0042** (was +0.0257). Nothing clears the noise floor any more - **recommendation withdrawn** |
| S5 retrieval | same `use_dense` route | **state-gated** on `state.over_general` | ✅ measured, 200 sessions, both customers (§3e) — **no-op again**, matches ungated almost exactly on both, same pattern as the S6 pool-shape gate |
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

## 3e. Gating the S5 route - `dense_route_nobrowse` resolves the trade-off

> **Superseded by §3g.** The numbers below are correct as measured, but the baseline they
> were measured against carried the two bugs fixed in §3f. Re-run on the fixed codebase,
> the stressed gain falls from +0.0257 to +0.0042 and the recommendation at the end of this
> section is withdrawn. The mechanism discussion still holds; the conclusion does not.

`RetrievalConfig` gains `dense_gate_over_general` / `dense_gate_exclude_browsing`,
identically named and worded to `RerankConfig`'s (§3b), and `retrieve()` reuses
`src.rerank._dense_gate_open` directly rather than duplicating the logic - no circular
import (`rerank.py` does not import `retrieval.py`), and the function is duck-typed
against the two config fields, so it works unchanged against a `RetrievalConfig`. Three
new `tools/sweep.py` rows mirror the S6 gate rows: `dense_route_gate` (pool-shape alone),
`dense_route_nobrowse` (track exclusion alone), `dense_route_gate_nobrowse` (both).
`starter/agent.py` threads `track=track_name` through both `retrieve()` call sites, and 4
new `DenseRouteTests` (`tests/test_components.py`) passed on the first run - the
track/`state.intent_track` precedence bug §3b caught and fixed is shared by construction,
so this reuse was protected from it automatically.

Measured at full 200-session scale, both customers:

| | router_on | dense_route_all | dense_route_gate | dense_route_nobrowse | dense_route_gate_nobrowse |
|---|---|---|---|---|---|
| **official** (200, cooperative) | 0.92349 | 0.91931 (−0.0042) | 0.91851 (−0.0050) | **0.92329 (−0.0002)** | 0.92315 (−0.0003) |
| **heavy+browse-gated** (200, stressed) | 0.76086 | 0.78718 (+0.0263) | 0.78629 (+0.0254) | **0.78652 (+0.0257)** | 0.78401 (+0.0232) |

**`dense_route_gate` (pool-shape alone) is a no-op again**, on both customers - matching
`dense_route_all` almost exactly everywhere. Same pattern as `dense_rr_gate` in §3b:
`state.over_general` does not discriminate the turns where this route actually helps or
hurts.

**`dense_route_nobrowse` (track exclusion alone) resolves the trade-off cleanly.** On
`official`, browsing MRR is 0.8716 vs `router_on`'s 0.8715 - an almost exact match, the
cooperative cost is gone. On `heavy+browse-gated`, it keeps +0.0257 of the ungated
route's +0.0263 - **98% of the stress-side gain survives**. `dense_route_gate_nobrowse`
(both gates together) is close behind but strictly worse than `nobrowse` alone on both
customers - the pool-shape gate adds restriction without benefit, consistent with it
being a no-op on its own.

**Why the same gate behaves so differently on the two customers - the mechanism from §3b,
now working in the exclusion's favour.** Under `heavy+browse-gated`, sessions promote out
of `intent_track=="browsing"` within 2-3 turns and stay `"buying"` for the rest of a
longer session (§3b, traced directly); most of the turns that decide the outcome happen
after promotion, outside the gate's reach, so most of the gain survives untouched. Under
the cooperative customer, disclosure and convergence are faster - more of a browsing
session's scoring-decisive turns land *before* promotion completes, which is exactly
where the cost was concentrated, so excluding that window removes it almost entirely.
Not a designed property of the gate; an empirical asymmetry this measurement surfaced.

**Confirmed on holdout too, and it goes further than "cost eliminated."**
`tools/sweep.py --split holdout --configs router_on,dense_route_all,dense_route_nobrowse`
(80 sessions - the split this project's own convention treats as the actual gate, dev
selects/holdout gates):

| | router_on | dense_route_all | dense_route_nobrowse |
|---|---|---|---|
| holdout (80) | 0.9149 | 0.9084 (−0.0065) | **0.9180 (+0.0031)** |
| browsing MRR (holdout) | 0.87 | 0.80 | **0.89** |

On holdout, `dense_route_nobrowse` doesn't just recover the ungated route's cost - it
scores *above* `router_on`, and browsing MRR (0.89) is the highest of all three
configurations, including the un-gated baseline. All three independent checks now point
the same way: official −0.0002 (noise-zero), holdout **+0.0031**, stressed **+0.0257**.

**Recommendation: `dense_route_nobrowse` is the S5 configuration worth keeping as a
documented, flag-gated option** - `RetrievalConfig(use_dense=True,
dense_gate_exclude_browsing=True)`. Not shipped as a new default; `use_dense` stays
`False` by default, unchanged. `dense_route_gate` and the redundant
`dense_route_gate_nobrowse` combination are not worth carrying forward as separate rows.

---

## 3f. Cross-branch audit vs `integration/gemini-stress-harness` - three bugs ported in

This branch and `integration/gemini-stress-harness` both branch from the same commit
(`9921650`, the `dynamic-state-slot` tip) and both independently merged `stress_harness`,
so both carry `tools/stress_harness.py` and `tools/stress_observe/`. Neither is an
ancestor of the other: 23 commits here, 16 there. `dynamic-state-slot` itself has **no**
unique commits - it is a strict ancestor of this branch, so there is nothing to port from
it.

The gemini branch went on to find four bugs. **Three were in code this branch also has,
and all three were still present here** - confirmed by running them, not by reading code.
Two of them corrupt exactly the paraphrase and browse-gated paths §3d/§3e were measured
on, which is why they are recorded here and not just in `agent_changes.md`.

| bug | file | gemini fix | status here |
|---|---|---|---|
| A. carrier framing glued onto real disclosures | `src/text.py` | `5b7c76f` | **was present** → ported (`0462f4d`) |
| B. browse-gated stall text pollutes the slot ledger and fakes a productive turn | `src/state.py` | `df080db` | **was present** → ported (`b6a334f`) |
| C. observe probes reject the kwargs the agent passes | `tools/stress_observe/runner.py`, `tools/observe.py` | `ec470db` | **was present, in both files** → ported (`e484cbe`) |
| D. LLM rerank temperature + reroute discards `llm_scores` | `src/llm.py`, `starter/agent.py` | `2224245` | **n/a** - no `src/llm.py` here, and this branch's reroute already threads `embed=`/`qvec=` through both calls |

**Bug A** was the significant one for ranking. `constraint_spans()` kept every ≥2-word
punctuation-split chunk with no stopword stripping - correct for the official evaluator's
templated wording, where a colon isolates the carrier, but wrong for the harness's
paraphrased carriers, which have no separator:

```
"One more thing - a breathable net weave."   was ['one more thing', 'a breathable net weave']
"I'd also want it to be synthetic sole."     was ['i d also want it to be synthetic sole']
```

The clean value was being destroyed, not merely accompanied by noise. `query_spans()`
feeds the reranker's span-coverage term - the primary deterministic ranking signal - and
a glued blob is far less likely to appear literally in a product's text than the clean
value, so this was diluting ranking under exactly the paraphrase stress this branch
exists to measure.

**Bug C** was the most misleading. Both probe copies had signatures predating this
branch's `track=`/`embed=`/`qvec=` kwargs, so every call raised `TypeError` inside
`Agent.respond()`'s catch-all and the turn returned an empty envelope. Measured on 5
`paraphrase:heavy+browse-gated` sessions, before → after the fix:

```
hit 0.000 / MRR 0.0000 / score 0.000000  ->  hit 1.000 / MRR 0.6750 / score 0.858500
```

Worse than a zero score: the empty envelope made the diagnostic classify all 5 as
`never_retrieved` - "target never entered the retrieval pool, a recall problem (S1/S5)".
That is the exact signal used to reason about where a dense route could help, so **any
`never_retrieved` figure taken from `tools/stress_observe` or `tools/observe.py` on this
branch before `e484cbe` is an artifact, not a measurement.** Aggregate numbers from
`tools/stress_harness.py` are unaffected - it does not install these probes, which is why
§3d/§3e's headline scores stand independently of this bug.

All three fixes are score-neutral on the official set: **0.923487 bit-identical**
(hit@10 1.000, MRR 0.880956, efficiency 0.796) after each, and harness `--verify` delta
`9.52e-08`. Tests went 127 → 134.

### Other differences (design divergence, not bugs)

| | `state-encoder-eval` | `integration/gemini-stress-harness` |
|---|---|---|
| official score | 0.923487 | 0.921497 |
| unique work | bi-encoder embeddings (`src/embed.py`, S5 route + S6 term) | DeepSeek LLM listwise rerank (`src/llm.py`, `src/router.py`) |
| dense/embedding code | yes | **none** |
| browsing-policy mechanism | live per-turn `state.intent_track` (§1) | kept `dual_tracking`'s `route_policies`, fixed at session open |
| `src/phrasing.py`, `policy.py`, `context_programming.py`, `router.py` | unchanged from base | all four changed |

The two branches solved the browsing-policy problem differently - live per-turn here,
fixed-at-opening there. Neither is ported; they are alternative designs, not a fix and a
bug.

---

## 3g. Re-measurement after the §3f fixes - the S5 result does not survive

Two of the three bugs in §3f corrupt exactly the paraphrase and browse-gated paths §3d/§3e
were measured on, so the headline comparison was re-run on the fixed codebase. Same
configs, same 200-session scale, same three checks:

| check | baseline `router_on` | `dense_route_nobrowse` | gain |
|---|---|---|---|
| official (200) — **before** fixes | 0.92349 | 0.92329 | −0.0002 |
| official (200) — **after** fixes | 0.92349 | 0.92329 | −0.0002 *(unchanged)* |
| holdout (80) — **before** | 0.9149 | 0.9180 | +0.0031 |
| holdout (80) — **after** | 0.9149 | 0.9180 | +0.0031 *(unchanged)* |
| `heavy+browse-gated` (200) — **before** | 0.76086 | 0.78652 | **+0.0257** |
| `heavy+browse-gated` (200) — **after** | **0.77065** | 0.77480 | **+0.0042** |

Two things happened at once on the stressed customer, and they point the same way:

1. **The baseline rose +0.0098** (0.76086 → 0.77065) with no model involved. That is the
   §3f fixes paying off directly: `constraint_spans()` now returns `synthetic sole` where
   it used to return `i d also want it to be synthetic sole`, and `query_spans()` feeds
   that straight into the reranker's span-coverage term.
2. **The dense gain fell to 16% of its former size** (+0.0257 → +0.0042). The route was in
   large part *substituting for the broken lexical signal*. Once the exact-match signal
   works on paraphrased wording again, there is much less left for the embedding to add.

Official and holdout are bit-identical before and after, which is the expected result and a
useful control: the fixes only fire on free-form customer wording, which the official
simulator never produces. It also means the two checks that were already inside the noise
floor stay there.

**Conclusion: the recommendation in §3e is withdrawn.** `dense_route_nobrowse`'s case rested
entirely on the stressed +0.0257 being the one number that cleared noise; at +0.0042 it no
longer does, and neither do official (−0.0002) or holdout (+0.0031). Combined with the four
S6-rerank variants already measured net-zero-or-worse at full scale (§3, §3b, §3c), **no
embedding configuration tested on this branch now clears the noise floor on any check.** The
code stays (flag-gated, `use_dense=False` by default, byte-identical off-path) so the result
is reproducible and the next person does not repeat it — but it should not be turned on.

This is also the clearest vindication of the project's own measurement discipline: the
+0.0257 was a real, correctly-run, full-scale measurement. It was just measured against a
baseline that was quietly broken, and only a cross-branch audit surfaced that. Per §3f, no
`never_retrieved` figure from `tools/observe.py` or `tools/stress_observe` predating
`e484cbe` should be trusted either.

Not re-run after the fixes: the four S6-rerank variants (§3, §3b, §3c). They were
net-zero-or-worse beforehand and the fixes raise the baseline they lost to, so re-running
them could only make their case worse, not better - but the specific post-fix numbers are
genuinely unmeasured rather than assumed.

---

## 4. What has not been checked

- **Whether §3d/§3e's results are stable across a second run** - RRF fusion and ONNX
  inference are deterministic here in principle, but this has not been independently
  re-run to confirm, only reasoned about.
- **Whether combining §3e (gated S5 retrieval) with §3c (slots query text) does better
  than either alone** - untested. §3d/§3e used `full_text()` for the query (the
  default); §3c showed query-text quality plausibly matters more on this branch than it
  used to.
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

# §3e - the recommended gated variant, both customers
python3 tools/stress_harness.py --customer official \
    --configs router_on,dense_route_all,dense_route_gate,dense_route_nobrowse,dense_route_gate_nobrowse
python3 tools/stress_harness.py --customer paraphrase:heavy+browse-gated \
    --configs router_on,dense_route_all,dense_route_gate,dense_route_nobrowse,dense_route_gate_nobrowse
```

Everything above is off by default (`dense_weight=0.0`, `use_dense=False`, all four
`dense_gate_*` fields `False`); the official evaluator and the full test suite are
unaffected regardless of artifact presence - `python3 -m evaluator.local_evaluator` still
reads **0.923487**, `python3 -m unittest discover -s tests -t .` still passes 127/127.
