# Architecture

One customer turn, end to end. Every stage is a module under `src/`; the only
file the evaluator knows about is `starter/agent.py`, which owns the response
contract and wires the rest together.

## Data flow

```
                       reset(session_id, user_profile)
                                    |
                                    v
                          DialogState created
                                    |
       respond(session_id, user_message, turn, top_k)
                                    |
                                    v
  +---------------------------------------------------------------+
  |  S3  state.observe(turn, text)                                 |
  |      - append utterance with provenance                        |
  |      - detect reversal cues  -> down-weight prior turns        |
  |      - detect "no preference" -> mark attribute dead           |
  |      - detect new spans       -> mark turn productive          |
  +---------------------------------------------------------------+
                    |                         |
      full_text()   |                         |  focused_text()
   (all turns)      v                         v   (post-override only)
  +---------------------------------------------------------------+
  |  S5  retrieval.retrieve()                                      |
  |      terms route ---------> FTS5 bm25, pool 300                |
  |      focused route -------> FTS5 bm25 (only after an override) |
  |      fused by reciprocal rank fusion, k=60                     |
  +---------------------------------------------------------------+
                                    |
                          candidates: [(asin, score)]
                                    |
                                    v
  +---------------------------------------------------------------+
  |  S6  rerank.rerank()  (top 200)                                |
  |      + verbatim constraint-span coverage   <-- dominant signal |
  |      + normalised retrieval score                              |
  |      + popularity prior (tie-break only, weight 0.02)          |
  +---------------------------------------------------------------+
                    |                                  |
                    v                                  v
  +--------------------------------+   +--------------------------------+
  |  S4  policy.select()           |   |  S7  agent._shortlist()        |
  |      which attribute to ask    |   |      emit now, or hold?        |
  +--------------------------------+   +--------------------------------+
                    |                                  |
                    v                                  v
              ask_attribute                      recommendations
                    \                                  /
                     +----------------+---------------+
                                      v
                          {message, ask_attribute,
                           recommendations, usage}
```

## Stage boundaries

| Stage | Owns the decision | Input | Output | On failure |
|---|---|---|---|---|
| S2 `router` | Is this customer deciding or exploring? | opening message | `Route` (phrasing) | defaults to browsing |
| S3 `state` | What has the customer told us, and when? | raw turn text | utterances, dead attributes, override mark | empty message is recorded, not rejected |
| S5 `retrieval` | Which products are plausible? | conversation text | ranked pool of ≤300 | malformed FTS expression returns `[]` |
| S6 `rerank` | Which plausible product is *the* one? | pool + disclosed spans | reordered pool | no spans → pool passes through untouched |
| S4 `policy` | What is worth asking next? | pool + state | one legal attribute | illegal value coerced to `other` |
| S7 timing | Show a list, or ask again? | pool + turn | ≤`top_k` recommendations | before evidence, returns `[]` |

Failure at any stage degrades that turn rather than the session: `respond()`
wraps the whole path and returns a valid empty envelope on any exception, because
the evaluator scores a raised exception as a missed session.

## Why the boundaries fall here

**State is separate from retrieval** because the scoring lever is accumulation,
not search. Keeping provenance in `DialogState` rather than folding it into a
query string is what makes intent override expressible at all: the override
handler down-weights utterances, and the retrieval layer reads that weighting
through two different views of the same history (`full_text` and `focused_text`)
without knowing why they differ.

**Reranking is separate from retrieval** because the two need opposite things.
Retrieval needs recall and gets it from a permissive bag-of-words OR query.
Reranking needs precision and gets it from exact verbatim span matching, which is
useless as a retrieval route — measured at 47/80 recall against the terms route's
80/80 — but decisive as a rescoring signal.

**The policy reads the pool rather than the state** so that the question depends
on what is still ambiguous among live candidates, not on a fixed checklist. This
is what `InfoGainPolicy` exploits; `FixedPolicy` ignores the pool and is the
default only because it currently wins on held-out data.

## Extension points

- **A new retrieval route** implements `(index, state) -> [(asin, score)]` and is
  fused by adding one `_rrf()` call in `retrieval.retrieve()`. A dense route
  would attach here.
- **A new clarification policy** implements `select(state, candidates) -> str` and
  `question(attribute) -> str`, and is passed via `AgentConfig(policy=...)`.
- **A neural reranker** implements the same signature as `rerank.rerank()` and is
  selected in `AgentConfig`. This was built and measured as `src/semantic.py` (S6b,
  a cross-encoder over ambiguous clusters) and then removed: it lost on every split.
  The code is preserved on the `semantic-rerank` branch and the measurements are in
  `docs/team/rerank_signals.md` §9. The constraint any future attempt must respect
  is the one that shaped it: the core path carries a no-network guarantee, so a
  model call is an opt-in layer *above* the local scorer that falls back to it,
  never a dependency inside it.
