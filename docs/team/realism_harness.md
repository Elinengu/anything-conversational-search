# Realism harness — testing beyond the official simulator

`tools/sim_harness.py`

## Why

`evaluator/local_evaluator.py` is a fully-cooperative, templated, deterministic
customer: it answers every question, discloses constraints copied **verbatim**
from the target's own metadata, and its "intent override" replaces one
target-derived value with another (`behavior_for()` draws both from the same
card — never a real retraction). It measures Pillar IV (retrieval / ranking /
turn efficiency) and the accumulation half of Pillar II. It cannot measure:

- Pillar I routing intelligence (the sim ignores the agent's prose)
- Pillar II proactive guidance / genuine override
- Pillar III adaptation (nothing to adapt to)

The 800 private sessions "may paraphrase." This harness quantifies what that
costs.

## What it does

Drives the **unmodified** `Agent` through a faithful copy of `evaluate()`'s
session loop, swapping the customer policy:

| customer | behaviour |
|---|---|
| `official` | byte-identical replay of the simulator. `--verify` asserts it reproduces `local_evaluator` (delta 0.00000). |
| `paraphrase --level light` | same constraints, **verbatim tokens kept**, only the carrier sentence reworded ("For that, what matters is: X" → "It should be X."). |
| `paraphrase --level medium` | the constraint itself reworded ("color: blue" → "in blue"; "100% Cotton" → "all cotton"; "budget around $45" → "roughly 45 dollars"; "buckle closure" → "a buckle fastening"). Rule-based, deterministic per session. |
| `decoy` | `intent_override` sessions where the pre-override preference is a **genuine decoy** — a colour/material value the target does *not* have and whose token is absent from its text. The override becomes a real retraction. |

`tok_cov` = fraction of the target's ground-truth constraint tokens that reached
the agent's accumulated `full_text()` — a proxy for extraction erosion.

```bash
python3 tools/sim_harness.py --customer official --verify
python3 tools/sim_harness.py --all --dataset data/public_set.jsonl
python3 tools/sim_harness.py --all --dataset data/hard_set.jsonl
```

## Results (main @ dd9ba8a, FixedPolicy default)

### PUBLIC (200) — official score 0.93050

| customer | hit@10 | MRR | score | Δ | tok_cov |
|---|---|---|---|---|---|
| official | 1.000 | 0.9013 | 0.93050 | — | 0.846 |
| paraphrase:light | 0.995 | 0.8539 | **0.91056** | −0.0199 | 0.886 |
| paraphrase:medium | 0.980 | 0.8157 | **0.88711** | −0.0434 | 0.796 |
| decoy | 1.000 | 0.9005 | 0.92986 | −0.0006 | 0.846 |

decoy per scenario: `intent_override` hit 1.000 / mrr 0.894 / score 0.909
(26/30 override sessions got a real decoy).

### HARD (96) — official score 0.80198

| customer | hit@10 | MRR | score | Δ | tok_cov |
|---|---|---|---|---|---|
| official | 0.896 | 0.7198 | 0.80198 | — | 0.941 |
| paraphrase:light | 0.875 | 0.6406 | **0.75905** | −0.0429 | 0.972 |
| paraphrase:medium | 0.865 | 0.6298 | **0.74768** | −0.0543 | 0.810 |
| decoy | 0.885 | 0.7087 | 0.79260 | −0.0094 | 0.943 |

decoy per scenario: `intent_override` hit 0.812 / mrr 0.606 / score 0.704
(16/16 override sessions got a real decoy).

## Findings

1. **Paraphrasing is the largest untested exposure.** `light` paraphrase keeps
   every constraint token yet costs **−0.020 / −0.043**; `tok_cov` even *rises*
   (0.846 → 0.886) while the score falls. The loss is not missed constraints —
   it is `constraint_spans()` failing to yield a clean fragment for the
   verbatim-substring reranker (`src/rerank.py`, the dominant signal). `medium`
   costs **−0.043 / −0.054** and loses whole sessions (hit@10 1.000 → 0.980).
   **If the private simulator paraphrases, the real score is ~0.89–0.91.**

2. **Genuine decoy override is handled.** −0.001 / −0.009. This contradicts the
   `hard_cases.md` note that the override down-weight is "inert with nothing to
   express" — when the pre-override turn carries a real decoy, the
   `focused_text` route + `_facet_conflicts` recover. The official sim just never
   exercises it. (Hard-set override still drops to 0.704 — the decoy compounds
   with the bucket's existing difficulty.)

## Implications for the build

- **Robustness fix worth pursuing:** loosen the reranker's constraint match from
  exact substring to token-set overlap (or add the bge dense route, which is
  semantic and already measured to help the hard tail). Re-run this harness to
  confirm the paraphrase gap closes.
- **Write-up:** "Innovation & Problem Insight" (20%) is scored on how clearly
  the challenge is framed. This harness + these numbers are that section:
  *what the public simulator can't measure, and how we tested beyond it.*
- **Override machinery is not dead weight** — keep it; it is private-set
  insurance that the official metric hides.

## Extending

`Customer` is the base class. To add a stressor, subclass and override
`_render()` (what a disclosure looks like), `opening()`, or `reply()`:

- `PartialDisclosure` — return "no additional preference" for the first 2 `other`
  asks; only disclose on a *specific* attribute ask. Tests whether InfoGain /
  proactive clarification beats "ask other".
- `NoisyCustomer` — 40% "I'm not sure"; occasional contradiction with no
  "actually" cue. Tests state-machine robustness.
- `OverGeneral` — open with a bare category, disclose nothing unless asked a
  pointed question. Pillar II's over-generality trigger.

An LLM paraphraser (offline model or API, disclosed as an eval-time tool, never
part of the agent) would replace the rule-based `_reword_one` for a `heavy`
level.
