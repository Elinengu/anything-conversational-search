# LLM rerank config — how to switch it

How to configure the optional S6 LLM semantic-reranking layer (`src/llm.py`,
`src/rerank.py`), for the three situations that come up in practice:

1. **No LLM rerank at all** — the shipped default, every existing config.
2. **Gated LLM rerank switched on** — the measured, recommended way to run it.
3. **LLM rerank on all the time (ungated)** — exists in code, not recommended,
   not shipped as a named config.

Background, the full design rationale, and the measured numbers for situation 2
are in `docs/team/agent_changes.md` Change 17 and `IMPLEMENTATION.md` §S6. This
file is only the "how do I actually flip this" reference.

All commands are run from the repo root with `python3`, same as `test_guide.md`.

---

## The three knobs

Two dataclasses control this end to end. Nothing else needs to change.

```python
# src/llm.py
LLMConfig(
    enabled: bool = False,          # the transport switch — off = no network, ever
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com/chat/completions",
    api_key_env: str = "DEEPSEEK_API_KEY",   # env var name, never the key itself
    timeout: float = 8.0,
    max_tokens: int = 400,
    temperature: float = 0.0,
)

# src/rerank.py
RerankConfig(
    llm_weight: float = 0.0,        # the fusion switch — 0.0 = layer is a no-op
    llm_gate_margin: float = 0.05,  # the gating switch — see "gated" below
    llm_depth: int = 8,             # how many top candidates go in the prompt
)
```

`LLMReranker.available` (what `rerank()` actually checks before calling out) is
`True` only when **both** `LLMConfig.enabled=True` **and** the environment
variable it names (`DEEPSEEK_API_KEY` by default) is actually set. So there are
two independent gates before any network call can happen at all:

```
enabled=True  AND  DEEPSEEK_API_KEY set  AND  llm_weight > 0.0  AND  gate open
   (transport)         (credential)          (fusion)         (per-turn)
```

Miss any one of the first three and the layer never calls out — same as
situation 1. The fourth (the gate) is what separates situations 2 and 3.

---

## Situation 1 — no LLM rerank at all

**This is already the default.** You don't need to change anything — every
config that doesn't explicitly set `llm=` or `rerank.llm_weight` gets this:

```python
from starter.agent import Agent, AgentConfig

agent = Agent("data/catalog.jsonl", AgentConfig())   # no LLM, ever
```

```bash
python3 tools/sweep.py --configs router_on            # named config, same thing
python3 -m evaluator.local_evaluator                  # the official run — always this
```

If you've been experimenting and want to be certain a config has it off, set it
explicitly rather than relying on omission:

```python
from src.rerank import RerankConfig

AgentConfig(rerank=RerankConfig(llm_weight=0.0))       # explicit off
```

You do **not** need `DEEPSEEK_API_KEY` set, network access, or the `LLMConfig`
import for this situation. This is the only situation the submission rules'
network-restricted scoring guarantee depends on — see `README.md` "Disclosure".

---

## Situation 2 — gated LLM rerank (recommended)

Fires the model only when the pool is genuinely ambiguous: `state.leader_margin
< llm_gate_margin` (default `0.05`), the same live pool-shape signal the dense
embedding term is gated on. This is the configuration that was actually
measured — see Change 17 for the split-by-split numbers.

**1. Set the API key** (never commit it, never put it in a config default):

```bash
export DEEPSEEK_API_KEY="sk-..."
```

**2. Build the config:**

```python
from starter.agent import Agent, AgentConfig
from src.llm import LLMConfig
from src.rerank import RerankConfig

config = AgentConfig(
    use_router=True,
    llm=LLMConfig(enabled=True),
    rerank=RerankConfig(llm_weight=1.0, llm_gate_margin=0.05),
)
agent = Agent("data/catalog.jsonl", config)
```

**3. Or use the named config that already exists** (`tools/sweep.py`
`build_configs()`):

```bash
python3 tools/sweep.py --split holdout --configs router_on,llm_rerank_gated
python3 tools/observe.py --config llm_rerank_gated --tag llm_on   # traced, produces viewer.html
python3 tools/stress_harness.py --customer paraphrase:heavy+browse-gated \
    --configs router_on,llm_rerank_gated
```

**Tuning the gate width:** `llm_gate_margin` is a threshold on `leader_margin`
(0 = perfectly tied leaders, larger = a more confident pool). Raising it (e.g.
`0.10`) makes the layer fire on *more* turns, including some with a mild lead
already; lowering it (e.g. `0.02`) restricts it to only the most contested
turns. `0.05` is the one value that's been measured — moving it needs its own
before/after run, same rules as any other weight in this repo (dev/holdout
gate, `docs/team/agent_changes.md` write-up).

**Tuning the prompt window:** `llm_depth` (default `8`) bounds how many of the
already lexically-sorted head candidates go into the prompt — and therefore
latency, cost, and how far down the model can promote a candidate from. It can
never reorder a candidate ranked below this window to the top.

---

## Situation 3 — LLM rerank on all the time (not recommended, not shipped)

Disables the gate so the model is asked on **every** turn, regardless of how
confident the lexical ranking already is. Exists in code (it was needed to
measure situation 2 against something), but it is not a config this repo
recommends or ships as a default anywhere.

```python
config = AgentConfig(
    use_router=True,
    llm=LLMConfig(enabled=True),
    rerank=RerankConfig(llm_weight=1.0, llm_gate_margin=0.0),   # 0.0 = gate disabled
)
```

```bash
python3 tools/sweep.py --configs llm_rerank_always
```

Why it's discouraged: a confident lexical leader has nothing to gain from a
nondeterministic network call and everything to lose (latency, cost, and a
model call that can disagree with a ranking that was already correct). This
project's own design note (`docs/team/ideas_to_integrate_llm.md` §Tier 2 #3)
calls for exactly the opposite: an opt-in layer that only spends the network
call where the deterministic signal has run out of discriminating power — which
is what the gate in situation 2 implements. `llm_rerank_always` has **not**
been re-measured on the current (post–Change 16) codebase; treat any number
from it as unverified until someone runs it and writes it up.

---

## Quick reference

| I want… | Config | Needs `DEEPSEEK_API_KEY`? |
|---|---|---|
| No LLM rerank (default, network-restricted-safe) | `AgentConfig()` or `router_on` | No |
| Gated LLM rerank (recommended, measured) | `llm_rerank_gated` | Yes |
| Always-on LLM rerank (not recommended) | `llm_rerank_always` | Yes |

| I want to know… | Run |
|---|---|
| Is the layer actually off in my config | check `config.rerank.llm_weight == 0.0` — nothing else matters |
| Did gated LLM rerank help | `python3 tools/sweep.py --split holdout --configs router_on,llm_rerank_gated` |
| *Why* did it help/hurt on one session | `python3 tools/observe.py --config llm_rerank_gated` then open `runs/latest/viewer.html` |
| How robust is it under paraphrase/browse-gating | `python3 tools/stress_harness.py --customer paraphrase:heavy+browse-gated --configs router_on,llm_rerank_gated` |
| Did I break anything | `python3 -m unittest discover -s tests -t .` — `tests/test_llm.py` and `LLMRerankTests` in `tests/test_components.py` cover this layer, no network needed (the transport is mocked) |
