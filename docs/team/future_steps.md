# Future Steps — Beyond the Public Evaluator

The public score is saturated (~0.92, hit@10 ~1.0). Marginal gains there are
noise-fitting. This is the map of what is actually worth doing, why, and what it
costs.

---

## 1. What the public evaluator rewards vs the four spec pillars

Judging is **Technical Execution 35% / Innovation 20% / Impact 20% / Feasibility
15% / Presentation 10%.** The automated `TechnicalScore` is explicitly "an
objective input to the Technical Execution assessment… not a separate judging
criterion" — so it is at most ~35% of one criterion.

| Pillar (from the problem statement) | Value on the public evaluator |
|---|---|
| I — Dual-track routing | ~0. The router only changes question wording; routing retrieval by intent was measured and dropped (dev flat, holdout -0.002). |
| I — LLM semantic ranking / vector similarity | ~0. BM25 recall is already ~100%; the dense route is off by default and never helped. |
| II — Incremental slots (accumulate constraints) | **The whole game.** 0.11 -> 0.78 came from this alone. |
| II — Slot *erasure* on override | **Negative.** The "erased" preference is still drawn from the target, so erasing it costs score; down-weighting / ignoring wins. |
| II — Proactive clarification / retrieval cutoff on over-generality | ~0. The simulator always answers "anything else?", so `ask_attribute="other"` strictly dominates targeted questions. |
| III — Personalized context distillation (user_profile) | ~0 to negative. Four independent attempts, all failed the holdout test. |
| III — Adaptive orchestration / runtime re-orchestration | Untested; the pipeline is deterministic and there is nothing to adapt to. |

**Why:** a deterministic, templated, fully-cooperative simulator cannot reward
dialogue intelligence — there is nothing to be clever about when the customer
always answers on request. The `user_profile` is aggregate *past* taste, weakly
linked to the specific target (only 26.5% of targets satisfy all their profile
tags; for `intent_override` the profile aligns with neither side). So the pillars
about *handling difficult customers* have no difficult customers to handle.

**Why this is probably deliberate, not a mistake:**

1. The number is <=35% of one of five criteria.
2. The pillars are listed under "Innovation Directions" — a pitch to human
   judges, not scored line-items.
3. The 800 private sessions "may paraphrase." If that simulator is harder
   (genuine decoys on override, sparser disclosure, paraphrased constraints),
   robust state handling, cue- (not template-) matching, and semantic retrieval
   become load-bearing insurance.
4. A demo of "anything else? … anything else? …" scoring 0.92 loses on
   Presentation and Innovation even while topping the objective input.

**Strategic implication:** the lean agent is the right *floor* — the feasibility
baseline and the public number. Build the pillar features as **opt-in layers
that do not regress the public score but demonstrably help on adversarial /
paraphrased cases.** That is the Innovation + Impact pitch and the private-set
hedge. Do not chase the last 0.005 on the leaderboard; 65% of judging is not
there.

---

## 2. The "other, then InfoGain for more" hybrid policy

Right instinct for realism, **zero score impact on this simulator.**
`customer_reply` has a hard cap of ~4 constraints per session. Once `other` has
drained them, InfoGain asking about anything returns "no additional preference" —
nothing left to extract. The premise ("users don't remember what they want until
asked") is true for humans, false for this simulator.

Build it as an opt-in that provably does not regress the public number: it is
realism for the demo, and genuine insurance for the private set if that simulator
discloses less on the first ask.

---

## 3. Improving the evaluator

You **cannot** change `evaluator/local_evaluator.py` for your reported score —
the spec forbids it and it invalidates the number. Two legitimate moves:

1. **Build a separate realism harness** (extend `tools/hard_cases.py`).
   Synthesise sessions the official simulator cannot produce: paraphrased
   constraint disclosure, a *genuine decoy* on override, partial disclosure
   (customer volunteers 2, agent must dig for the other 2), contradictory
   constraints, a customer who says "I'm not sure" a lot. Measure against *this*,
   not just the public 200.
2. **Put the calibration critique in the write-up.** "Innovation & Problem
   Insight" (20%) is scored on how clearly the team framed the challenge. A
   section "what the public simulator can and can't measure, and how we tested
   beyond it" is exactly that criterion.

---

## 4. Remaining work, ranked

| work | pillar / criterion | LLM needed? |
|---|---|---|
| Realism harness + testing against it | Problem Insight, Feasibility | No |
| Router that actually routes (narrow/confirm for buying, wide net for browsing) — currently only changes wording | I — Dual-Track | No |
| Proactive retrieval cutoff on over-generality — when the pool is huge and undifferentiated, ask a targeted question *before* showing a list | II — Proactive Guidance | Trigger: no. Question text: better with LLM |
| Explanation text — "these all have a stainless steel band and a date window" from the reranker's matched spans | listed Innovation Direction; Presentation | Optional (template works, LLM better) |
| LLM semantic rerank over the top ~20 when span coverage saturates (homogeneous clusters) | I — LLM Semantic Ranking | Yes |
| LLM-generated clarification questions from the pool's facet distribution | II — structured clarification | Yes |
| Adaptive orchestration — skip stages that cannot contribute; switch strategy on a stagnating session (diversify list / disambiguate) | III — Adaptive Orchestration | No |
| Paraphrase-robust constraint extraction (vs regex/cues) for the private set | II — state machine robustness | Yes |

---

## 5. Does realism require an LLM?

Split it:

- **Realism in what the agent *says*** — varied natural clarification questions
  instead of the fixed `QUESTION_TEXT` table; explanations; handling messy /
  paraphrased customer input — **yes, LLM-shaped.** The repeated "Is there
  anything else that matters for this one?" is the single biggest realism gap and
  there is no non-LLM fix that isn't a bigger template table.
- **Realism in what the agent *does*** — retrieval, core ranking, state
  tracking, orchestration — **no.** BM25 recall is ~100%, exact-span matching is
  a strong ranking signal, and the state machine is deterministic logic.

**Architecture:** keep the offline lexical core as the guaranteed path and the
score floor; add an LLM as **opt-in layers** on top — (1) question generation,
(2) semantic rerank over top-K, (3) explanation text, (4) extraction fallback —
each degrading to the current local behaviour if the model is unavailable. This
keeps the "$0, deterministic, offline" story *and* adds the "intelligent layer"
story, and a network-restricted scoring run still gets 0.92.

---

## 6. Model options and what to expect

### bge-small vs the BGE reranker

| model | type | params | on-disk | can generate text? |
|---|---|---|---|---|
| `bge-small-en-v1.5` | bi-encoder embedding | 33M | ~130 MB | No — outputs a 384-dim vector |
| `bge-reranker-base` | cross-encoder | 278M (~8x) | ~1.1 GB fp32 / ~560 MB fp16 | No — outputs one relevance score per (query, doc) pair |
| `bge-reranker-large` / `-v2-m3` | cross-encoder | ~560M | ~2.2 GB | No |
| `bge-reranker-v2-gemma` | LLM-based reranker | 2.5B | ~5 GB | Technically a decoder, but purpose-built to emit a score, not chat |

**Can BGE "do inference like an LLM"?** The embedding and cross-encoder models
cannot generate text at all — they are BERT-family encoders that output vectors
or scores. Only the `v2-gemma` / `v2-minicpm` rerankers wrap a decoder LLM, and
even those are tuned to score, not converse. For the clarification questions and
explanations you need an actual generative model.

### Speed (CPU, rough; GPU is 10-40x faster)

| operation | latency |
|---|---|
| current agent, one turn | ~17 ms |
| bge-small: encode one query | ~2-8 ms |
| bge-reranker-base: score 20 candidates | ~0.3-1.2 s per turn |
| Qwen2.5-0.5B (GGUF Q4): one ~25-token question | ~1-2 s |
| Qwen2.5-1.5B: same | ~2-4 s |
| Qwen2.5-3B / Llama-3.2-3B: same | ~5-9 s |
| Qwen2.5-7B / Llama-3.1-8B: same | ~12-25 s |

So a small generative LLM on CPU is **100-1000x slower per call** than
bge-small; the cross-encoder reranker sits in between (~10-100x bge-small, but
10-50x faster than a 3B LLM). A 3B model for question-gen on CPU adds
~5-15 s/turn -> ~1-2.5 min/session -> **hours for a full 200-session eval run**.
On a T4-class GPU this collapses to sub-second and is a non-issue. Latency and
cost are disclosed as feasibility metrics, not part of the score.

### Expected score improvement

| add | public | hard set | notes |
|---|---|---|---|
| bge-small dense retrieval route | ~0 | ~0 | BM25 recall already ~100%; cache + `dense_weight` already in the tree, defaulted off after measuring flat |
| bge-reranker cross-encoder over top-20 | ~0 to +0.01 | maybe +0.01-0.03 on `homogeneous_cluster` / `degenerate_card` | The winning rerank signal is *exact verbatim span match* — the disclosed constraints are literal catalog text. A semantic cross-encoder optimises for *meaning*, not string matching, so it may not beat span coverage on public and can even demote an exact-match target. Only clearly helps where span coverage saturates. |
| small LLM: clarification questions | **0** | 0 | The simulator ignores the English entirely. Pure Presentation / Innovation / demo value. |
| small LLM: extraction fallback | 0 | 0 | Helps only if the private simulator paraphrases; regex is fine on public. |

### Can a small LLM produce good responses?

For **narrow, well-scoped rewriting/formatting** — "turn these facet options into
a natural question", "explain why these 3 products match given these matched
phrases" — **yes**, a 1.5-3B instruct model (Qwen2.5, Llama-3.2, Phi-3.5-mini)
is acceptable, provided the actual content is supplied in the prompt and the
output is validated with a template fallback. 0.5B is too weak (unreliable
instruction following). For open-ended dialogue or nuanced reasoning a small
model is mediocre and hallucination-prone — do not ask it to *decide* anything,
only to *phrase* things whose content the deterministic pipeline already
produced. 7-8B is where quality gets genuinely good, at ~4-5 GB and ~2-4x the
latency.

**Bottom line on models:** an offline stack of `bge-small` (retrieval, already
cached) + `bge-reranker-base` (rerank tie-breaker over top-20) + a ~3B instruct
model (questions + explanations, template fallback) keeps everything local, adds
~1-2 s/turn on GPU, and buys demo realism plus a modest hard-set gain — not a
public-score jump.
