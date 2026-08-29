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

Each `.md` has a `.pdf` rendering beside it.

Three documents live at the repository root rather than here, because they are the
first things a reader opens: [`../README.md`](../README.md),
[`../IMPLEMENTATION.md`](../IMPLEMENTATION.md) (the full stage-by-stage account) and
[`../ARCHITECTURE.md`](../ARCHITECTURE.md).

## Adding to the change log

Use the `record-change` skill (`.claude/skills/record-change/SKILL.md`). It carries
the measurement rules the numbers in `team/agent_changes.md` depend on — chiefly that
a baseline is captured before editing, and that a teammate's commit landing
mid-measurement voids the comparison.
