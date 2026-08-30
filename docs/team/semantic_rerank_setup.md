# Enabling the Semantic Reranker (S6b)

**Stage S6b — `src/semantic.py`, added in change 11 (`docs/team/agent_changes.md`)**

This is the step-by-step guide to switching on the optional neural
cross-encoder reranking stage: what to install, how to download the
open-source model, how to enable and verify it, and how to turn it back off.

---

## Read this first: what you are enabling

The stage is **off by default, and it is off for a measured reason**. Enabled,
it *lowers* the score on every split (dev 0.9268 → 0.9211, hard 0.7981 →
0.7944) and multiplies mean turn latency ~13× — the full numbers and the
mechanism are in `docs/team/rerank_signals.md` §9. Every score reported in
`README.md` and `IMPLEMENTATION.md` was measured with it off.

So why enable it at all?

- to **reproduce the negative result** — the measurement is the deliverable
  against the "semantic reranking" innovation direction in
  `docs/competition_specification.md`;
- to benchmark a **different model** in the same harness (swap one download);
- to re-test if the **query or document text changes** in a way that could
  cure the domain mismatch that sank it.

Never enable it for a submission run.

---

## What it is

When the symbolic ranking is visibly undecided — many candidates tied on span
coverage, one dominant (category, material, colour) cluster, no distinctive
span — the stage rescores the top of the pool with a neural **cross-encoder**:
a small model that reads the customer's request and one product's text
*together* and emits a relevance score. The neural ranking is fused with the
symbolic one by reciprocal-rank fusion (RRF), never by adding raw scores,
because cross-encoder logits are uncalibrated while one matched span is worth
~1.12 on the symbolic scale.

| | |
|---|---|
| Model | [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) |
| License | Apache-2.0 (open source, redistribution permitted) |
| Size | 22.7M parameters; **23.2 MB** as the int8 ONNX graph we run |
| Runtime | `onnxruntime` on CPU — no torch, no transformers, no GPU |
| Where the weights live | `models/ms-marco-MiniLM-L6-v2/` — **gitignored, never committed** |

The weights are not in the repo because `docs/submission_rules.md` allows only
"lightweight local assets" and reserves the right to score with the network
disabled. The agent is built so their absence costs nothing: on a clean clone
the stage silently no-ops and every result matches the README exactly.

---

## Step 1 — install the Python dependencies

From the repo root:

```bash
pip install -r requirements.txt
```

That installs exactly four packages, for this stage only:

| package | why |
|---|---|
| `onnxruntime` | runs the ONNX model on CPU |
| `numpy` | tensor input/output for onnxruntime |
| `tokenizers` | the model's WordPiece tokenizer (Rust, fast) |
| `huggingface_hub` | downloads the model files in step 2 |

The scored agent needs none of these — its pipeline is Python standard library
plus SQLite FTS5. If this install fails, nothing else in the project is
affected.

## Step 2 — download the open-source model

```bash
python3 tools/fetch_model.py
```

Expected output (arm64 Mac shown; x86_64 picks the AVX-512 variant):

```
cross-encoder/ms-marco-MiniLM-L6-v2  (arm64 -> onnx/model_qint8_arm64.onnx)
  model_quantized.onnx             23.2 MB
  tokenizer.json                    0.7 MB
  total                            23.9 MB   -> models/ms-marco-MiniLM-L6-v2
```

What the tool does, so you can audit or do it by hand:

1. Downloads two files from the model's Hugging Face repo — an **int8
   quantised ONNX graph** matched to your CPU architecture (the upstream repo
   publishes these pre-exported, which is why we need no torch and no export
   step) and `tokenizer.json`.
2. Copies them into `models/ms-marco-MiniLM-L6-v2/` under the names
   `src/semantic.py` looks for: `model_quantized.onnx` (preferred) or
   `model.onnx` (full-precision fallback).

Useful variants:

```bash
python3 tools/fetch_model.py --variant onnx/model.onnx      # full precision, 91 MB
python3 tools/fetch_model.py --model-id <other-hf-repo>     # benchmark another reranker
```

A different model must be a cross-encoder for sequence classification whose
ONNX graph takes `input_ids` / `attention_mask` (/ `token_type_ids`) and whose
repo carries a `tokenizer.json`. (For example,
[`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)
qualifies structurally but is ~20× larger and multilingual — see the "not in
scope" note in `rerank_signals.md` §9.)

No Hugging Face account or token is needed; you may see an unauthenticated
rate-limit warning, which is harmless.

**Offline check:** `git status --porcelain models/` must print nothing — the
directory is gitignored and stays that way.

## Step 3 — enable it

The switch is `SemanticConfig.enabled`, carried on `AgentConfig`:

```python
from starter.agent import Agent, AgentConfig
from src.semantic import SemanticConfig

agent = Agent("data/catalog.jsonl",
              AgentConfig(semantic=SemanticConfig(enabled=True)))
```

Or through the sweep harness, which is how the recorded numbers were produced:

```bash
python3 tools/sweep.py --split dev --configs semantic_off,semantic_on
```

The other rows: `semantic_loose` (gate thresholds from the original proposal,
fires ~73% of sessions), `semantic_tight` (stricter gate), `semantic_d20`
(rescore top 20 instead of 50 — half the latency).

Knobs on `SemanticConfig` (`src/semantic.py`): gate thresholds
(`tied_leaders`, `facet_cluster`, `distinctive_max`, `conditions_required`),
cost (`depth`, `max_length`, `batch_size`), fusion (`weight_semantic`), and
`model_dir` for a non-default weights location.

## Step 4 — verify it is actually running

The stage is built to fail *silently* into a no-op, so verify positively:

```python
from src.semantic import _scorer, DEFAULT_MODEL_DIR, SemanticConfig

s = _scorer(DEFAULT_MODEL_DIR)
print(s.ok)   # must be True; False means missing runtime or missing weights
print(s.score("leather belt buckle",
              ["leather belt with buckle", "cotton dress"], SemanticConfig()))
# sensible output: something like [7.2, -11.1] - relevant >> irrelevant
```

Then confirm end-to-end that enabling it changes results (that is the point —
with it on, dev drops to ~0.9211):

```bash
python3 tools/sweep.py --split dev --configs semantic_off,semantic_on
```

If `semantic_on` scores identically to `semantic_off`, the stage no-opped:
check `s.ok` above, and that `models/ms-marco-MiniLM-L6-v2/` contains both
`model_quantized.onnx` and `tokenizer.json`.

## Turning it off / removing it

- **Off:** it already is, unless you passed `enabled=True` — there is nothing
  persistent to undo.
- **Remove the weights:** `rm -rf models/` (23.9 MB back; the stage returns to
  no-op).
- **Remove the runtime:** `pip uninstall onnxruntime numpy tokenizers
  huggingface_hub` — the agent and all 68 tests run identically without them.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `semantic_on` ≡ `semantic_off` | stage no-opped: runtime or weights missing | steps 1-2, then the `s.ok` check |
| download fails | no network / HF rate limit | retry; or fetch the two files by hand from the model repo's `onnx/` folder and place them as `models/ms-marco-MiniLM-L6-v2/{model_quantized.onnx,tokenizer.json}` |
| turns suddenly ~10× slower | that is the measured cost (mean 30.7 → 389.8 ms) | expected; use `semantic_d20`, or turn it off |
| test suite aborts at exit with `recursive_mutex lock failed` | macOS onnxruntime teardown bug | should not occur — weights are checked before the runtime is imported (change 11, third note); if it recurs, `rm -rf models/` and report |

## Pointers

- `src/semantic.py` — implementation; module docstring carries the design and
  the measured verdict.
- `docs/team/rerank_signals.md` §9 — full decision log: oracle ceiling, gate
  measurements, ablation grid, why it failed.
- `docs/team/agent_changes.md` change 11 — the team-facing record.
- `README.md` → "Optional: semantic reranking (S6b)" — the short version of
  this document.
