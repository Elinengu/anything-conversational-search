# Documentation

This directory holds two kinds of file, and the difference matters.

## Frozen — supplied by the organizer

These five arrived with the challenge. **Do not edit them.** They define the
contract the agent is scored against, and the competition specification refers to
them by these exact paths, so they must not be moved or renamed either.
`tests/test_contract.py` loads the contract file directly.

| File | What it is |
|---|---|
| `competition_specification.md` | Participant rules and the evaluation protocol |
| `agent_api_contract.json` | Machine-readable Agent response contract |
| `evaluation_config.json` | Scoring configuration |
| `baseline_results.json` | Reproducible weak-starter reference score |
| `submission_rules.md` | Participant submission requirements |

`evaluator/` and `data/` are frozen on the same basis: modifying either invalidates
the score. Check with:

```bash
git status --porcelain evaluator/ data/ \
  docs/competition_specification.md docs/agent_api_contract.json \
  docs/evaluation_config.json docs/baseline_results.json docs/submission_rules.md
```

## Ours — written by the team

Everything under [`team/`](team/) is ours to edit freely.

| File | What it covers |
|---|---|
| [`team/agent_changes.md`](team/agent_changes.md) | The running change log: every change by author, with before/after numbers |
| [`team/ideas.md`](team/ideas.md) | Reranking and recommendation-strategy ideas, each with its measured result |
| [`team/hard_cases.md`](team/hard_cases.md) | Failure analysis of the adversarial set and the prioritised fix plan |
| [`team/category_tail_match.md`](team/category_tail_match.md) | The last public-set miss, and the evidence that the fix does not overfit |
| [`team/reranking_explained.md`](team/reranking_explained.md) | How the reranking stage works, from first principles |
| [`team/intent_override_retrieval.md`](team/intent_override_retrieval.md) | Handling the intent-override scenario in retrieval |
| [`team/bm25.md`](team/bm25.md) | How BM25 / BM25F lexical scoring works, grounded in `src/index.py`, and why recall is ~100% on the public set but brittle to paraphrase |
| [`team/stress_harness.md`](team/stress_harness.md) | (branch `stress_harness`) `tools/stress_harness.py` — paraphrase / browsing-gated / decoy customer stressors and the retrieval-vs-ranking diagnostic |
| [`team/dual_track_routing.md`](team/dual_track_routing.md) | (branch `dual_tracking`) making Buying/Browsing routing drive behaviour, and why it stayed on a branch |
| [`team/dense_rerank.md`](team/dense_rerank.md) | (branch `dense_rerank`) `bge-small` embedding cosine as an S6 rerank signal, tested under paraphrase: helps only the degenerate tail, wash on the full set |
| [`team/dense_route.md`](team/dense_route.md) | (branch `dense_rerank`) same `bge-small` as an S5 retrieval route, browsing-track-only — recovers **none** of the `never_retrieved` tail, slightly negative overall |
| [`team/branch_state_encoder_eval_changes.md`](team/branch_state_encoder_eval_changes.md) | (branch `state-encoder-eval`) re-running the dense work against the live state machine, and a cross-branch bug audit — the one embedding result that cleared the noise floor turned out to be compensating for a broken lexical signal, and does not survive fixing it; no embedding configuration now clears noise on any check |

Each `.md` has a `.pdf` rendering beside it; regenerate with
`python3 tools/md_to_pdf.py <file>.md <file>.pdf` (a lightweight reportlab
renderer — headings, tables, code, lists).

Three documents live at the repository root rather than here, because they are the
first things a reader opens: [`../README.md`](../README.md),
[`../IMPLEMENTATION.md`](../IMPLEMENTATION.md) (the full stage-by-stage account) and
[`../ARCHITECTURE.md`](../ARCHITECTURE.md).

## Adding to the change log

Use the `record-change` skill (`.claude/skills/record-change/SKILL.md`). It carries
the measurement rules the numbers in `team/agent_changes.md` depend on — chiefly that
a baseline is captured before editing, and that a teammate's commit landing
mid-measurement voids the comparison.
