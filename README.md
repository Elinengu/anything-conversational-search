# Conversational Shopping Agent — TechJam 2026 Track 4

A multi-turn shopping agent for the TechJam Conversational E-Commerce Search
Challenge. It finds a hidden target product in a frozen 50,000-item Amazon
Clothing/Shoes/Jewelry catalog by asking the customer questions and narrowing a
candidate pool across turns.

**Public-set score: `0.8592`** against the supplied BM25 baseline's `0.1067` — an
8x improvement, with no model API, no network, and no dependencies beyond the
Python standard library.

| Split | Sessions | Hit@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---|---|---|---|---|---|
| Public (all) | 200 | 0.940 | 0.791 | 3.40 | 0.759 | **0.8592** |
| Dev (tuning) | 120 | 0.950 | 0.814 | 3.27 | 0.772 | 0.8738 |
| Holdout (untouched) | 80 | 0.925 | 0.756 | 3.60 | 0.740 | 0.8374 |
| *Supplied baseline* | 200 | 0.125 | 0.068 | 9.81 | 0.119 | *0.1067* |

Every number above is reproducible with the commands in
[Reproducing the results](#reproducing-the-results).

## The core insight

The supplied baseline is stateless. It answers each turn from that turn's message
alone, and it never sets `ask_attribute`.

That second part is what costs it the challenge. The simulated customer only
discloses a constraint when the agent asks for one, and each session holds exactly
four constraints drawn verbatim from the target product's own metadata. An agent
that never asks a question never receives more than the opening sentence, and an
agent that asks but forgets throws away each answer as it arrives.

So the largest win is not in retrieval. It is *asking a question and remembering
the answer*. That change alone — accumulate every turn, always ask something, hold
the shortlist until something has been disclosed — moved the score from `0.1067`
to `0.7811` before a single retrieval parameter was touched. Everything after that
is comparatively small.

The second insight follows from the first: because disclosed constraints are
*verbatim* product copy, a candidate whose text literally contains
`"stainless steel band"` is far more likely to be the target than one that merely
shares those tokens. Matching those spans as a reranking signal took the score from
`0.7799` to `0.8543`, and the remaining tuning to `0.8592`.

## Architecture

```
customer turn
     |
     v
 S2 router ......... buying vs browsing, from linguistic cues
     |
     v
 S3 state .......... accumulate turns, track provenance, handle intent override
     |
     v
 S5 retrieval ...... FTS5 bag-of-words + post-override focused route, fused by RRF
     |
     v
 S6 rerank ......... verbatim constraint-span coverage over the top 200
     |
     v
 S4 policy ......... which attribute to ask about next
     |
     v
 S7 timing ......... emit a shortlist, or hold and ask again
```

| Module | Stage | Responsibility |
|---|---|---|
| `src/index.py` | S1 | Catalog → in-memory FTS5 index + trimmed product records |
| `src/router.py` | S2 | Buying/browsing classification from cues, not templates |
| `src/state.py` | S3 | Turn accumulation, slot provenance, override handling |
| `src/policy.py` | S4 | Clarification policy (`FixedPolicy`, `InfoGainPolicy`) |
| `src/retrieval.py` | S5 | Multi-route candidate generation with RRF fusion |
| `src/rerank.py` | S6 | Verbatim span coverage, facet agreement, popularity tie-break |
| `src/facets.py` | — | Typed attribute extraction shared by S4 and S6 |
| `starter/agent.py` | — | The `Agent` the evaluator imports; owns the response contract |
| `tools/sweep.py` | S0 | Experiment harness and the dev/holdout split |
| `tools/observe.py` | — | Session tracer: annotated transcripts and failure diagnosis |
| `tools/build_stoplist.py` | S1 | Learns `src/stoplist.py` (metadata boilerplate) from the catalog |

See [agent_changes.md](agent_changes.md) for the running log of every change by author,
with before/after numbers. New entries are added with the `record-change` skill in
`.claude/skills/`, which carries the measurement rules the numbers depend on.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the data flow, the boundaries between
stages, and the failure behaviour at each one. See
[IMPLEMENTATION.md](IMPLEMENTATION.md) for a full stage-by-stage account of what
changed and why, written for readers new to search and recommender systems, with
enhancement ideas at the end of each stage.

## Setup

Python 3.10 or newer. **No third-party packages are required.**

```bash
# 1. Download catalog.jsonl.gz from the repository's GitHub Release, then:
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl

# 2. Verify against the published SHA256SUMS file.
```

There is nothing else to install, no API key to set, and no environment variable
to configure.

## Reproducing the results

```bash
# Official evaluator — writes results.json (~28s)
python3 -m evaluator.local_evaluator

# The dev/holdout split the tuning decisions were made against
python3 tools/sweep.py --split dev
python3 tools/sweep.py --split holdout

# Test suite: contract, components, and the end-to-end scoring floor (~14s)
python3 -m unittest discover -s tests -t .
```

`tools/sweep.py` compares several named configurations in one process, sharing a
single catalog index. Add a row to `build_configs()` to test a change.

## Inspecting sessions

Aggregate metrics say *how much* was lost, never *where*. `tools/observe.py`
records what happened in every session — what the customer said, what it
disclosed, how the agent ranked, and why the target was or was not found.

```bash
python3 tools/observe.py --verify              # all 200 sessions -> runs/<tag>-<stamp>/
python3 tools/observe.py --scenario intent_override
python3 tools/observe.py --only public_0008
python3 tools/observe.py --dataset data/hard_set.jsonl --tag hard
```

**The evaluator is not modified, copied, or forked.** `evaluate()` accepts the
agent as a parameter, so the observer wraps the agent and lets the organizer's own
evaluator drive every session — the reported score is the real one. `--verify`
re-runs untraced and asserts the two agree (they do, to the last digit). The
pipeline stages are probed by wrapping `classify` / `retrieve` / `rerank` in
`starter/agent.py`'s namespace for the duration of the run, so no production code
carries any tracing overhead.

| Flag | Default | Purpose |
|---|---|---|
| `--dataset` | `data/public_set.jsonl` | session file to run |
| `--catalog` | `data/catalog.jsonl` | catalog to index |
| `--scenario` | all | `buying` / `browsing` / `intent_override` / `boundary` |
| `--only` | all | comma-separated `sample_id`s |
| `--limit N` | all | first N sessions after filtering |
| `--tag` | `public` | names the run folder |
| `--out` | `runs` | run-folder root |
| `--top` | 10 | candidates recorded per turn |
| `--no-markdown` | off | skip the per-session files, keep `trace.jsonl` |
| `--verify` | off | re-run untraced and assert an identical score (~2x runtime) |

Each run writes `index.md` (worst sessions first), `sessions/<id>.md` (one
annotated transcript per session), `trace.jsonl`, `summary.json`, and a
self-contained offline `viewer.html` that filters by scenario, outcome, and
failure mode. `runs/` is git-ignored; regenerate rather than commit.

Any session file works, not just the public set: one JSON object per line with
`sample_id`, `scenario_type`, `ground_truth.parent_asin` and `user_profile`, with
every target present in the catalog. A sample carrying its own `intent_card` **and**
`behavior` bypasses derivation (`evaluator/local_evaluator.py:205`), and the
observer renders whichever card was actually in play.

Every session is classified into one failure mode, which is the point of the tool —
a miss caused by a target that was never retrieved is a different repair from one
that sat at rank 14 for six turns:

| Diagnosis | Meaning | Stage to repair |
|---|---|---|
| `hit` | found and scored | — |
| `never_retrieved` | target never entered the candidate pool | S1 / S5 |
| `ranked_out` | in the pool throughout, never reached the shown top 10 | S6 |
| `withheld_only` | reached the top 10, but the list was held back every time | S7 |
| `override_locked` | shown before the override fired, so the hit could not count | S3 |
| `exhausted` | 10 turns elapsed with the target outside the top 10 | — |

The runs also count **turns left on the table**: hits where the target was already
inside a convertible top 10 before the list was shown. Placements before an
override are excluded, since `evaluator/local_evaluator.py:252` cannot convert them.

### What the first full run showed

On the public set, all 12 misses are `ranked_out` and **none** are
`never_retrieved` — the target entered the pool in 200 of 200 sessions, at best
ranked positions of 13, 17, 19, 20, 20, 23, 28, 30, 32, 47, 53 and 117. Every
remaining point on this set is a ranking problem, not a search problem.

The same run counts **98 turns left on the table** across 89 sessions. Perfect
timing would move MTTC from `3.405` to `2.915`, worth `+0.0098` — more than any
other measured opportunity. The cause is legible in the transcripts: at those
turns the top-1/top-2 margin has a median of `0.017`, and every one of them falls
under `confidence_margin = 0.20` (`starter/agent.py:59`). In `public_0008` the
target held rank 1 on turn 2 with `4.8243` against `4.8053` — a 0.4% margin, so
the gate held and a turn was spent. Span coverage saturates, which bunches the
scores, and the gate is reading a scale it cannot discriminate on. That figure is
the ceiling of perfect foresight, not what a re-tuned fixed threshold would reach.

The adversarial set (`tools/hard_cases.py`) is where retrieval is genuinely under
stress: 96 sessions score `0.6842`, and 4 sessions are `never_retrieved`.

## Design decisions worth defending

**Intent override down-weights rather than erases.** The brief describes override
as slot erasure. We deliberately deviate. In this evaluator the discarded
preference is *still derived from the target product*, so erasing it destroys
usable signal; pre-override turns are weighted to `0.35` instead of dropped, and a
separate retrieval route runs over post-override turns alone. If the private
simulator uses genuine decoy values, the focused route already isolates them.

**Verbatim span matching lives in the reranker, not in retrieval.** As an FTS5
phrase query it recalls the target in only 47 of 80 sampled sessions, so fusing it
at retrieval time injects more noise than signal. Applied as a rescoring signal
over a pool the bag-of-words route already fills — where it recalls the target
80/80 at median rank 1 — the same evidence is pure gain.

**The clarification policy ships as the simpler of two implementations.**
`InfoGainPolicy` selects the question that most reduces uncertainty over the live
candidate pool, using a gain ratio (raw information gain is dominated by brand,
which has thousands of distinct values and which shoppers can rarely answer) times
an answerability prior. It is the more interesting design and it finds the target slightly *more* often on
the dev split (hit rate `0.958` against `0.950`), but it ranks it worse — MRR
`0.703` against `0.814` — and loses overall on both splits: `0.8369` against
`0.8738` on dev, `0.8141` against `0.8374` on holdout. Specific questions surface
fewer constraints per turn, and thinner evidence ranks worse even when it still
retrieves. We ship the policy that wins on data we did not tune against, and keep
the other selectable via `AgentConfig(policy=InfoGainPolicy(agent.facets))`.

**Intent routing does not touch retrieval, because measurement said not to.**
Widening the pool for browsing and narrowing it for buying changed the dev score
not at all and cost `0.002` on the holdout. Rather than keep it as decoration, the
router was scoped to what it genuinely does well: phrasing the agent's questions
appropriately for an exploring versus a decided customer.

**Span rarity weighting was measured and removed.** Weighting each disclosed span
by pool-local rarity — so that "buckle closure" counts for less than "two row
stitch" among belts — is principled and was implemented in full. It moved the dev
score by `0.0002` and the holdout not at all, because a pool retrieved by those
same terms has little rarity spread left to exploit. It was deleted rather than
left in as an off-by-default flag.

**Tuning constants sit mid-plateau, not at the argmax.** The confidence margin is
`0.20` because every value from `0.15` to `0.50` beats `0.0` on both splits and the
curve between them is flat. With 200 public sessions deciding nothing and 800
private ones deciding everything, picking a split's argmax is how you buy noise.

## Disclosure

| | |
|---|---|
| Model / API | **None.** No LLM, no network, no credentials. |
| Dependencies | Python standard library only |
| Estimated cost | $0.00 |
| Token usage | 0 prompt, 0 completion — `usage` is reported honestly as zero |
| Index build | 4.1s one-time, at `Agent.__init__` |
| Per-turn latency | mean 16.6ms, p95 19.8ms, max 21.0ms |
| Peak memory | ~282 MB resident |
| Full public evaluation | ~28s for 200 sessions |

The agent runs identically with the network disabled; there is no fallback path
because there is no online path. This was a deliberate choice: the submission rules
reserve the right to score under network restrictions, and an agent that scores
zero in that environment is worth less than one that scores `0.8592` everywhere.

A neural cross-encoder reranking stage was built, measured and **removed from this
branch**: it lost on every split (dev 0.9268 → 0.9211) at ~13× the latency. The
code lives on the `semantic-rerank` branch and the full decision log is
[`docs/team/rerank_signals.md`](docs/team/rerank_signals.md) §9.

## Limitations and what we would do next

- **Facet extraction is regex and keyword based.** It produces a usable partition
  of the candidate pool, not a correct product taxonomy. A material like
  "recycled polyester blend" resolves to `polyester`, and colour is taken from the
  first match in the text rather than the item's actual colourway.
- **`InfoGainPolicy` is not yet good enough to ship.** Its weakness is that it
  cannot estimate *how many* constraints an answer will surface, only how much a
  known answer would split the pool. That is visible in its results: it retrieves
  the target as often as the shipped policy but ranks it worse, because it gathers
  less evidence per turn. Modelling expected yield per attribute — from observed
  disclosure sizes rather than a fixed prior — is the obvious next step, and is
  what would let the more principled policy overtake the simpler one.
- **Intent override is the weakest scenario**, at `0.867` hit rate against `0.94`
  overall. Some of this is structural: those sessions cannot convert before the
  override arrives on turn 3 or 4, which puts a floor under MTTC. The remaining
  gap is genuine, and the focused route is a blunt instrument for it.
- **No dense retrieval.** A local sentence-transformer route was planned and cut:
  the lexical route recalls the target in **200 of 200** public sessions — measured
  exhaustively by `tools/observe.py`, not sampled — so the headroom is in ranking,
  not recall, and a model download would have compromised the no-network guarantee.
  The adversarial set is the caveat: 4 of its 96 sessions are `never_retrieved`, so
  the claim holds for the released distribution rather than universally.
- **The dev/holdout split is 120/80.** Differences below roughly `0.02` on the
  holdout are not distinguishable from noise at that size, and we have treated
  them as such throughout.

## Data

Catalog and sessions derive from Amazon Reviews 2023 by McAuley Lab, UCSD. See
[DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) before using or redistributing the data.
The evaluator, `data/`, and the five frozen files at the root of `docs/` are
organizer-owned and unmodified. Documents the team wrote live in
[`docs/team/`](docs/team/); [`docs/README.md`](docs/README.md) explains the split.
