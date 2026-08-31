# Conversational Shopping Agent

An offline-first, multi-turn product-search agent built for **TikTok TechJam
2026 Track 4: Conversational E-Commerce Search**. The agent asks useful
follow-up questions, remembers the customer's constraints, adapts when the
customer changes their mind, and recommends the hidden target product from a
frozen 50,000-item Amazon catalog.

The committed default configuration keeps LLM reranking **off**. It makes no
model API calls, requires no credentials, and reports zero token usage. On the
repository's bundled frozen local evaluator, this offline configuration
achieves a **0.954975 TechnicalScore** and **1.000 Hit Rate@10**.

| System | Hit@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Supplied BM25 baseline | 0.125 | 0.068 | 9.81 | 0.119 | 0.1067 |
| This project (offline default, frozen local evaluator) | **1.000** | **0.961** | **2.67** | **0.834** | **0.954975** |

The result above is recorded in [`results.json`](results.json) and can be
reproduced with the commands in [Reproducing the results](#reproducing-the-results).

## Latest customer evaluation results

The following are the team's latest 200-session comparisons. “Official” uses
the official customer behavior. The stress customer paraphrases constraints and
uses **gated browsing**: a browsing customer discloses information only when the
agent asks for a specific `ask_attribute` rather than a broad `other` question.

| Customer | Sessions | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Official customer | 200 | 1.000 | 0.9609 | 2.665 | **0.954975** |
| Stress: paraphrase + gated browsing | 200 | 0.990 | 0.8379 | 3.370 | **0.899070** |

Both rows are the current committed default, which now combines two changes —
**sniper list sizing** (one candidate per turn until turn 5) and the
**coarse-category pool retrieval route**. Measured over the same 200 official
sessions in one process:

| configuration | TechnicalScore | gain |
|---|---:|---:|
| neither | 0.9235 | — |
| category pool only | 0.9346 | +0.0111 |
| sniper sizing only | 0.9401 | +0.0166 |
| **both (ships)** | **0.9550** | **+0.0315** |

The two gains sum to `0.0277` and together deliver `0.0315`, so they are
**super-additive**: a one-candidate slate is only worth anything if that
candidate is right, and the pool is what makes the turn-1 candidate good.
Hit@10 does not move on any evaluated set; the whole gain is MRR
(`0.881 → 0.961`) and MTTC (`3.04 → 2.67`). See `docs/team/agent_changes.md`
changes 18 and 19.

### Optional LLM reranking layer

The repository also carries an opt-in LLM reranking layer, **off by default**
(`llm_weight=0.0`), which needs a `DEEPSEEK_API_KEY`. It was measured against
the agent as it stood *before* both changes above and has not been re-run,
so these numbers are historical and are not comparable to the table above:

| Customer | LLM reranking | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---|---:|---:|---:|---:|
| Official customer | Off | 1.000 | 0.8810 | 3.040 | 0.923487 |
| Official customer | On — gated | 1.000 | 0.8930 | 3.045 | 0.927012 |
| Stress: paraphrase + gated browsing | Off | 0.880 | 0.6628 | 4.410 | 0.770651 |
| Stress: paraphrase + gated browsing | On — gated | 0.880 | 0.6734 | 4.405 | 0.773924 |

It was worth `+0.003525` on the official customer and `+0.003273` under stress,
in both cases through better ordering of already-retrieved candidates rather
than through finding anything new. Since both shipped changes take their gain
from the same place — the rank a hit is scored at, and what is in the pool to
rank — the layer is unlikely to still be additive, and it stays off.

## Project overview

### Problem statement

For each evaluation session, one product is selected as the customer's hidden
target. The agent has at most ten conversational turns to place that product in
a ranked Top 10 list. The customer may begin with a clear buying requirement,
browse without a firm preference, reverse an earlier preference, or decline to
answer a question. Performance combines:

- **Hit Rate@10:** whether the target is found;
- **MRR:** how highly the target is ranked; and
- **MTTC:** how quickly the target is found.

Only an exact `parent_asin` match counts as a hit.

### How the solution addresses the problem

The supplied baseline searches each message independently and does not request
additional attributes. This project treats the task as a stateful conversation:

1. **Intent routing** distinguishes buying from browsing and updates the route
   as the conversation becomes more specific.
2. **Conversation state** accumulates useful constraints, records their turn and
   source, ignores “no preference” replies, and handles intent overrides without
   losing all earlier context.
3. **Adaptive clarification** asks broad or targeted questions according to the
   current intent, candidate ambiguity, and conversation progress.
4. **Multi-route retrieval** searches the full conversation, the opening anchor,
   post-override text, and an adaptive structured-state view. SQLite FTS5/BM25
   results are combined with Reciprocal Rank Fusion.
5. **Evidence-based reranking** rewards verbatim constraint spans, preserved
   attribute/value pairs, facet and category agreement, and appropriate
   popularity evidence while penalising explicit facet conflicts.
6. **Adaptive recommendation timing** avoids locking in a weak rank too early.
   A narrow first slate and an elimination scan prevent the same failed products
   from being shown repeatedly.
7. **Failure-aware orchestration** detects convergence, stagnation, and override
   recovery, then adjusts the retrieval route, question policy, and slate timing.

The key insight is that asking and remembering is more valuable than treating
every message as a new search. In this evaluator, a customer reveals further
constraints only after the agent sets `ask_attribute`; those disclosures are
then strong evidence for identifying the exact catalog item.

## Architecture

```text
Customer message
      |
      v
Intent router -----> buying / browsing / override transition
      |
      v
Dialog state ------> active constraints, declined slots, provenance, phase
      |
      v
Adaptive orchestrator --> retrieval route, question policy, recommendation gate
      |
      v
FTS5/BM25 routes --> Reciprocal Rank Fusion --> candidate pool (up to 300)
      |
      v
Constraint/facet/category/popularity reranker
      |
      v
Elimination scan + slate-size policy
      |
      v
message + ask_attribute + ranked parent_asin recommendations
```

| Component | Responsibility |
|---|---|
| `starter/agent.py` | Official `Agent` interface and end-to-end orchestration |
| `src/index.py` | In-memory SQLite FTS5 catalog index and BM25 search |
| `src/router.py` | Buying/browsing classification and turn-level intent changes |
| `src/state.py` | Multi-turn state, constraints, declines, overrides, and phases |
| `src/context_programming.py` | Context distillation and adaptive strategy selection |
| `src/policy.py` | Fixed and information-gain clarification policies |
| `src/retrieval.py` | Lexical, focused, anchor, structured, and optional dense routes |
| `src/rerank.py` | Constraint, facet, category, conflict, and popularity scoring |
| `src/phrasing.py` | Natural, pool-aware clarification wording |
| `evaluator/local_evaluator.py` | Frozen local simulator and official metric calculation |

For a more detailed data-flow description, see
[`ARCHITECTURE.md`](ARCHITECTURE.md). The complete design rationale and measured
experiments are in [`IMPLEMENTATION.md`](IMPLEMENTATION.md).

## Technology used

### Development tools

- **Python 3.10+** for the agent, evaluator, experiments, and test suite;
- **Git and GitHub** for source control and team collaboration;
- **Terminal/command-line tooling** for evaluation, configuration sweeps,
  adversarial-data generation, and session tracing;
- **Python `unittest`** plus custom regression and stress harnesses for testing;
- **Markdown** for architecture, experiment, attribution, and implementation
  documentation.

The project is editor-agnostic and does not require Colab, Jupyter, or a
notebook workflow. It can be opened in VS Code or any Python editor.

### APIs and models

| API/model | Use | Required for default result? |
|---|---|---|
| None | The submitted configuration is entirely local and deterministic. | — |
| DeepSeek `deepseek-chat` API | Optional, ambiguity-gated listwise reranking experiment through an OpenAI-compatible chat-completions endpoint. | No; off by default |
| `BAAI/bge-small-en-v1.5` | Optional ONNX sentence embeddings for dense retrieval or reranking experiments. | No; off by default |

The optional DeepSeek layer is **off by default** and fails closed: if it is
disabled, lacks a key, times out, or returns invalid output, the lexical ranking
is retained. Adding a key does not silently enable it; `LLMConfig.enabled` and
`RerankConfig.llm_weight` must also be set explicitly. Configuration details are
in [`llm_config_readme.md`](llm_config_readme.md).

### Libraries and frameworks

The default pipeline uses only the Python standard library:

- `sqlite3` with **FTS5/BM25** for full-text retrieval;
- `dataclasses`, `json`, `re`, and standard collection utilities for state and
  feature processing;
- `urllib` for the disabled-by-default DeepSeek client; and
- `unittest` for automated tests.

Optional dense-search experiments use:

- `numpy` for vector storage and cosine scoring;
- `onnxruntime` for local encoder inference; and
- `tokenizers` for BGE tokenisation.

PyTorch, pandas, scikit-learn, and Hugging Face Transformers are not runtime
dependencies. `huggingface_hub` is only an optional build-time route for
obtaining a different embedding model.

### Datasets and assets

| Dataset/asset | Description | Size |
|---|---|---:|
| Amazon Reviews 2023 `Clothing_Shoes_and_Jewelry` catalog | Frozen text and structured product metadata; target key is `parent_asin`. | 50,000 products |
| `data/public_set.jsonl` | Labelled local-development sessions: Buying, Browsing, Intent Override, and Boundary. | 200 sessions |
| `data/generated_test_set.jsonl` | Deterministically generated supplemental evaluation sessions. | 200 sessions |
| `data/generated_adversarial_set.jsonl` | Generated adversarial sessions for robustness checks. | 200 sessions |
| `data/hard_set.jsonl` | Six difficult retrieval/ranking buckets generated from the catalog. | 96 sessions |
| Optional BGE ONNX artifacts | Local encoder, tokenizer, and catalog-vector matrix used only by dense configurations. | Not committed |

The source data is **Amazon Reviews 2023**, published by McAuley Lab at UCSD.
The competition package contains product text and structured metadata, not real
shopping conversations, raw user identifiers, images, or private evaluation
labels. See [`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md) for attribution and use
conditions.

## Setup and installation

### 1. Clone the repository

```bash
git clone https://github.com/Elinengu/anything-conversational-search.git
cd anything-conversational-search
```

### 2. Create a virtual environment

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 --version
```

There is no package-install step for the default offline agent.

### 3. Verify the included catalog

`data/catalog.jsonl` is already included in the repository. No separate
download or decompression step is required. Verify the expected row count:

```bash
wc -l data/catalog.jsonl
```

The command should print `50000`.

### 4. Optional dense-search dependencies

Skip this step when reproducing the submitted offline result. To run the
`dense_*` experiment configurations:

```bash
python3 -m pip install -r requirements.txt
```

Dense configurations also require the corresponding model and embedding
artifacts under `data/embeddings/`. They self-disable if either the artifacts or
libraries are absent. See `tools/build_embeddings.py` for the one-time build.

### 5. Enable the optional LLM reranker

LLM reranking is disabled by default. The named `llm_rerank_gated`
configuration already contains the measured settings, including
`llm_weight=1.0` and `llm_gate_margin=0.05`; users do not need to edit Python or
construct an `AgentConfig` themselves.

1. Create a file named `.env` in the **repository root**—the same directory as
   this README—and add your API key:

   ```dotenv
   DEEPSEEK_API_KEY=sk-your-key-here
   ```

   The repository-root `.env` file is loaded automatically by `src/llm.py` and
   is ignored by Git. Never commit the file or paste a real key into source code.
   An exported `DEEPSEEK_API_KEY` environment variable takes precedence if both
   locations are present.

2. Select the existing named configuration from the command line:

   ```bash
   python3 tools/observe.py --config llm_rerank_gated --tag llm_on
   ```

   `--config llm_rerank_gated` enables the transport and applies the measured
   ambiguity gate. The model is called only when the lexical leaders are close.

3. To compare the default and LLM configurations across customer harnesses, use
   the plural `--configs` comparison option:

   ```bash
   # Official customer: default off versus gated LLM reranking
   python3 tools/stress_harness.py \
       --customer official \
       --configs router_on,llm_rerank_gated

   # Paraphrased, gated-browsing stress customer
   python3 tools/stress_harness.py \
       --customer paraphrase:heavy+browse-gated \
       --configs router_on,llm_rerank_gated
   ```

   Live API results can vary slightly between runs even with temperature `0.0`.

The named configuration already enables the required code-level switches. If it
is not selected, or if the key or ambiguity gate is unavailable, the agent
retains the local lexical ranking. See
[`llm_config_readme.md`](llm_config_readme.md) for the full configuration guide.

## Reproducing the results

Run all commands from the repository root.

### Frozen local public-set reproduction

```bash
python3 -m evaluator.local_evaluator
```

This evaluates all 200 public sessions and writes the detailed output to
`results.json`. The committed default is expected to report approximately:

```text
Hit Rate@10:    1.000000
MRR:            0.960917
MTTC:           2.665000
Efficiency:     0.833500
TechnicalScore: 0.954975
Token usage:    0
```

The score is calculated as:

```text
Efficiency = clip((11 - MTTC) / 10, 0, 1)
TechnicalScore = 0.50 * HitRate@10 + 0.30 * MRR + 0.20 * Efficiency
```

### Automated tests

```bash
python3 -m unittest discover -s tests -t .
```

Run the default test suite without `DEEPSEEK_API_KEY` or a DeepSeek key in the
repository `.env`; API behavior is tested with mocks and no live call is needed.

### Dev/holdout experiment split

The 200 public sessions are divided deterministically and by scenario into a
120-session development set and an 80-session holdout set:

```bash
python3 tools/sweep.py --split dev --configs router_on
python3 tools/sweep.py --split holdout --configs router_on
```

Use `--configs a,b` to compare named configurations in the same process while
sharing one catalog index.

### Robustness evaluation

```bash
# Frozen evaluator on the hard set
python3 -m evaluator.local_evaluator --dataset data/hard_set.jsonl

# Customer-language and browsing-behavior stress tests
python3 tools/stress_harness.py --all

# Trace and diagnose individual sessions
python3 tools/observe.py --only public_0008 --tag example
```

The observer writes an offline HTML viewer and per-session traces under
`runs/`. A complete testing and evaluation reference is available in
[`test_guide.md`](test_guide.md).

## Results and evaluation discipline

The evaluator was not modified. Changes were selected on the development split,
checked on the untouched holdout split, and challenged on generated and hard
sets. The repository records unsuccessful experiments as well as successful
ones; for example, unconditional dense retrieval and neural reranking were not
made defaults when they failed to improve consistently across splits.

The latest measured customer results are summarised in
[Latest customer evaluation results](#latest-customer-evaluation-results). The
gated DeepSeek reranker remains off by default even though it substantially
improves the official-customer run: it requires user-supplied credentials and
network access, adds latency and API cost, and is less reproducible than the
offline path. The stress result also shows that an LLM reranker cannot by itself
repair every paraphrase-driven retrieval or clarification failure.

## Limitations and reflection

The solution performs strongly on the supplied simulator, but the result should
not be interpreted as a solved real-world shopping problem.

- **Simulator wording is unusually favourable to lexical matching.** Disclosed
  constraints are often copied from target-product metadata. Real customers
  paraphrase, omit details, make spelling errors, and express subjective needs.
- **The public set is small.** There are only 200 labelled sessions, so repeated
  tuning can overfit even with a dev/holdout split. The private 800-session set
  is the more meaningful generalisation test.
- **The evaluation is simulated.** There has been no human usability study, and
  natural question quality does not affect the deterministic customer's answer
  in the local evaluator.
- **Popularity is only a tie-break signal, but can still introduce bias.** New,
  niche, or lightly reviewed products may be disadvantaged.
- **Semantic search is not part of the default.** The robust fallback is lexical,
  so heavy paraphrasing can still hurt retrieval and ranking. Dense and LLM
  routes improve some stress cases but introduce new accuracy trade-offs.
- **The system is text-only and catalog-specific.** It does not use product
  images, availability, live pricing, or cross-category behavioral signals.
- **The in-memory index targets this 50,000-item benchmark.** A production-scale
  catalog would require persistent indexing, incremental updates, monitoring,
  and stricter latency and memory controls.

Given more time, the team would run human conversation studies, evaluate on a
larger untouched paraphrase set, calibrate a gated lexical/dense hybrid on truly
unseen data, audit popularity and category bias, add grounded explanations for
recommendations, and move the index to a persistent service suitable for live
catalog updates.

## Team member contributions

Contributions below are summarised from the Git history and the measured change
log in [`docs/team/agent_changes.md`](docs/team/agent_changes.md).

| Team member | Main contributions |
|---|---|
| **Eline Ngu Xiang Ee (`Elinengu`)** | Project setup and integration; observer tooling; category-tail, stoplist, negative-facet, pair-span, slate-ramp, popularity, structured-state, adaptive orchestration, robustness fixes, and optional DeepSeek reranking work; experiment analysis and documentation. |
| **Kwong Weng** | Elimination-scan recommendation strategy; retrieval-recall fixes; natural clarification phrasing; realism/stress harness; dual-track experiments; optional BGE embedding, dense retrieval, and dense reranking infrastructure; testing and technical documentation. |
| **`corainexia`** | BM25 field-weight tuning; facet-agreement and category-agreement reranking; customer-query facet extraction. |
| **`xiaotong0329`** | Intent-router logic; multi-route anchor retrieval; dynamic context programming and track-aware early-recommendation gating. |

## Repository structure

```text
starter/       Agent entry point imported by the evaluator
src/           State, routing, retrieval, reranking, policies, and context logic
evaluator/     Frozen local customer simulator and scorer
tests/         Contract, component, regression, state, LLM, and tooling tests
tools/         Sweeps, tracing, stress tests, data generation, and weight fitting
data/          Public, generated, adversarial, hard, and local catalog data
docs/          Competition contract, evaluation configuration, and team notes
```

## Cost and runtime disclosure

| Item | Offline default |
|---|---|
| Model/API | None |
| Network required | No |
| Credentials required | No |
| Prompt/completion tokens | 0 / 0 |
| Estimated API cost | $0.00 |
| Required dependencies | Python 3.10+ standard library |

The optional DeepSeek and dense-embedding configurations are not requirements
for the offline default. DeepSeek reranking must be explicitly enabled even when
a valid key exists in the repository-root `.env` file.
