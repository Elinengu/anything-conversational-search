---
name: record-change
description: Record a change to the shopping agent in IMPLEMENTATION.md and docs/team/agent_changes.md, with the measurement that justifies it. Use after changing anything under src/, starter/ or tools/ that could move the score, or when the user asks to write up, log, or document a change.
---

# Recording a change to the agent

Two documents track this project, and a change is not finished until both are updated:

| File | Audience | What goes in it |
|---|---|---|
| `IMPLEMENTATION.md` | Someone new to search and recommender systems | *Why* the change exists, taught from first principles, filed under the pipeline stage it belongs to |
| `docs/team/agent_changes.md` | The team and the judges | *What* changed, by whom, with before/after numbers |

The same change gets a different treatment in each. Do not paste one into the other.

## Before you write anything: measure

A write-up without a measurement is a guess. Numbers claimed here are read by
judges, so every one must come from a run you actually did.

**1. Capture the baseline before you edit.** Not after, not from memory, not from a
number quoted earlier in the conversation.

```bash
python3 -m evaluator.local_evaluator        # public set, writes results.json
python3 tools/observe.py --tag base --no-markdown
python3 tools/observe.py --dataset data/hard_set.jsonl --tag base-hard --no-markdown
python3 tools/sweep.py --split dev
python3 tools/sweep.py --split holdout
```

**2. Check nobody else is editing while you measure.** This repo has several people
working in it at once, and a teammate's commit landing mid-experiment will be
silently attributed to your change. This has already happened once and produced a
spurious `+0.0084`.

```bash
git status --porcelain          # before AND after the run
git log --oneline -1
stat -f "%Sm %N" -t "%H:%M:%S" src/*.py starter/*.py
```

If any source file's mtime moved during your measurement, the comparison is void.
Re-run it. If `tools/sweep.py` and `evaluator/local_evaluator.py` disagree about the
same configuration, that is the alarm that the tree moved — investigate before
believing either.

**3. Prefer measuring both states in one process.** It holds everything else
constant and takes one run instead of two:

```python
import src.text as text
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent

ids, cats, prods = catalog_index("data/catalog.jsonl")
samples = load_jsonl("data/public_set.jsonl")
for label, value in (("before", OLD), ("after", NEW)):
    text.SOME_CONSTANT = value
    r = evaluate(Agent("data/catalog.jsonl"), samples, ids, cats, prods)
    print(label, r["recommended_technical_score"])
```

**4. Gate the change.** The holdout split must not regress. Differences below ~0.02
on the 80-session holdout are noise, not evidence of improvement. Use dev/holdout as
a gate, never to select a threshold or weight — that fits the 200 public sessions
through the back door, and the private 800 decide the real score.

**5. Run `tools/observe.py` on both states** and compare the failure-mode mix, not
only the score. A change that keeps the score but moves sessions from
`never_retrieved` to `ranked_out` has improved retrieval and hurt ranking, and the
write-up should say so.

**A measured no-change is a real result.** Report `0.000000` plainly and say what the
change buys instead — robustness, auditability, dead code removed. Do not hunt for a
split where it happens to look positive.

## Never touch

`evaluator/`, `data/` and **five files at the root of `docs/`** are organizer-owned:
`competition_specification.md`, `agent_api_contract.json`, `evaluation_config.json`,
`baseline_results.json`, `submission_rules.md`. Editing any of them invalidates the
score, and they must not be moved or renamed either — the competition specification
refers to them by path and `tests/test_contract.py` loads the contract file.

The rest of `docs/` is ours. Team documents live in `docs/team/`; see `docs/README.md`.

## Writing the `IMPLEMENTATION.md` entry

Find the stage the change belongs to in §5 (S0 harness, S1 index, S2 router, S3
state, S4 policy, S5 retrieval, S6 rerank, S7 timing, S9 robustness). Add to that
stage's **What changed**, following the existing shape:

- Plain language first, formula second. Every term is explained before it is used —
  the reader does not know what MRR, BM25, IDF or reranking mean.
- One concrete example from this catalog, with real product text or a real session id.
- Say why the change belongs in *this* stage rather than an adjacent one.
- End with **Measured effect** and the actual numbers.

Then **update that stage's "Ideas for this stage" list**. If the change implements an
idea already listed there, mark it done and point at the code — leaving it as an open
suggestion makes the document wrong. If the idea turned out to be misconceived, say
so and keep the evidence; a corrected idea is more useful than a deleted one.

If the change was tried and rejected, it goes in §6 with its numbers, not §5.

## Writing the `docs/team/agent_changes.md` entry

Add a row to the score-progression table, then a numbered section:

```markdown
## Change N — <short title> (<author>)

**Files:** `path/one.py`, `path/two.py` — commit `<sha>`

### Problem
What was wrong, with the evidence that showed it (a traced session, a count, a
failure mode from tools/observe.py).

### What changed
The mechanism, briefly, with a small code excerpt if it clarifies.

### Effect
| | before | after |
|---|---|---|
| Public set | 0.xxxxxx | 0.xxxxxx |
| Adversarial set | 0.xxxxxx | 0.xxxxxx |
| dev / holdout | 0.xxxx / 0.xxxx | 0.xxxx / 0.xxxx |

One sentence on what the numbers mean, including what did *not* move.
```

Keep the table and the sections in the same order, and keep the running net total at
the top current.

## Attribution

Use the author of the commit, not "the agent" or "Claude". Check with
`git log --oneline -5` and `git log -1 --format=%an <sha>`. If you cannot tell who
made a change, ask rather than guessing. Refer to people by the name in the commit
log; use they/them unless you know otherwise.

## Finishing

```bash
python3 -m unittest discover -s tests -t .
git status --porcelain evaluator/ data/ \
  docs/competition_specification.md docs/agent_api_contract.json \
  docs/evaluation_config.json docs/baseline_results.json docs/submission_rules.md
```

Report to the user: what changed, the measured effect with both numbers, what did not
move, and anything you could not verify.
