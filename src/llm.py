"""Shared LLM adapter for the shopping agent.

This is the only module allowed to talk to the remote DeepSeek API. The
routing, policy, and orchestration layers should depend on this adapter
rather than issuing HTTP requests directly.

Two layers live here:

1. ``DeepSeekClient`` - a thin, low-level wrapper around DeepSeek's
   OpenAI-compatible chat-completions API (``generate`` / ``generate_json``).
   ``src/router.py`` and ``src/phrasing.py`` call this directly for a single
   free-text completion (route hinting, clarification-wording polish).
2. ``LLMClient`` - a higher-level client used by ``starter/agent.py`` for the
   three purpose-built features from the competition's LLM tier:
       - Grounded Clarification Generation
       - Transparent Recommendation Explanations
       - Listwise semantic reranking of the candidate pool (S6 ``llm_weight``)
   It is built on top of ``DeepSeekClient`` (same API key, same network
   surface) rather than a second HTTP implementation, and degrades to a
   deterministic fallback string/score whenever no key is configured, the
   circuit breaker has tripped, or the call fails - exactly like
   ``DeepSeekClient``.

Both clients share the same off-by-default, fail-open contract: no
``DEEPSEEK_API_KEY`` (or a dead network) means zero behavioural change from
the fully offline agent.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency.
    load_dotenv = None

try:
    import requests
except ImportError:  # pragma: no cover - graceful degradation in minimal envs.
    requests = None


def _load_dotenv_if_present() -> None:
    """Load key-value pairs from .env if present, without requiring python-dotenv."""
    if load_dotenv is not None:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        return
    # Fallback: read directly from the repo .env if python-dotenv is not installed.
    for path in [Path(".env"), Path(__file__).resolve().parent.parent / ".env"]:
        if path.exists():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = val
            except Exception:
                pass


_load_dotenv_if_present()


DEFAULT_DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
_DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"


class DeepSeekClient:
    """Thin client for DeepSeek-backed LLM features.

    The rest of the project should call this adapter instead of reaching into
    DeepSeek's (OpenAI-compatible) chat-completions API directly. When the
    environment variable ``DEEPSEEK_API_KEY`` is not set, the client degrades
    gracefully and returns ``None`` so the deterministic rule-based pipeline
    continues to work.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model or DEFAULT_DEEPSEEK_MODEL
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and requests is not None

    def _url(self) -> str:
        return _DEFAULT_API_URL

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key or ''}",
        }

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str | None:
        """Generate a text response from DeepSeek.

        Returns ``None`` when no API key is configured or the request fails so the
        calling code can safely fall back to deterministic logic.
        """
        if not self.is_configured:
            return None

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "stream": False,
        }

        response = None
        try:
            response = requests.post(
                self._url(),
                headers=self._headers(),
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            choices = payload.get("choices") or []
            if not choices:
                return None
            message = choices[0].get("message") or {}
            text = message.get("content")
            return text if isinstance(text, str) else None
        except Exception as exc:  # pragma: no cover - debug aid for runtime configuration issues.
            print(f"DeepSeek request failed for model {self.model}: {type(exc).__name__}: {exc}")
            if response is not None:
                print(f"DeepSeek status: {response.status_code}")
                print(response.text[:500])
            return None

    def generate_json(self, prompt: str, *, system_prompt: str | None = None) -> dict[str, Any] | None:
        """Generate a JSON object from DeepSeek when the prompt asks for structured output."""
        text = self.generate(prompt, system_prompt=system_prompt)
        if text is None:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw_text": text}

    def usage_estimate(self, prompt: str, text: str) -> dict[str, int]:
        """Rough token accounting when DeepSeek's response doesn't carry ``usage``.

        DeepSeek's chat-completions response does include real ``usage``, but the
        thin ``generate()`` wrapper above only surfaces text, not the raw payload,
        so callers that need approximate token counts (for the harness's cost
        reporting) use this word-count heuristic instead.
        """
        return {
            "prompt_tokens": len(prompt.split()) * 2,
            "completion_tokens": len((text or "").split()) * 2,
        }


def get_llm_client() -> DeepSeekClient:
    """Return the shared default DeepSeek client for the application."""
    return DeepSeekClient()


DEFAULT_LLM_CLIENT = get_llm_client()


@dataclass
class LLMConfig:
    """Configuration for the higher-level LLM integration (``LLMClient`` below).

    ``provider`` only distinguishes "mock" (deterministic canned response, used
    in tests/CI without a network) from "deepseek" (the one real backend). It
    used to also support gemini/openai/ollama; the project standardised on
    DeepSeek (see docs/team/agent_changes.md), so those providers were dropped
    rather than kept as dead code paths.
    """

    enabled: bool = False
    provider: str = "deepseek"  # "deepseek" or "mock"
    model: str = DEFAULT_DEEPSEEK_MODEL
    temperature: float = 0.2
    max_tokens: int = 1000
    timeout_seconds: float = 8.0


class LLMClient:
    """DeepSeek-backed client with built-in deterministic offline fallback.

    Wraps ``DeepSeekClient`` with the three purpose-built prompts the agent
    needs (clarification generation, recommendation explanation, listwise
    rerank) plus a circuit breaker so a dead network costs at most a few
    timeouts per session rather than one per turn.
    """

    #: Consecutive failed calls after which the client stops trying for its lifetime.
    #: A dead-but-slow network otherwise costs (timeout x retries) on every
    #: turn, and evaluator timeouts count as misses; three strikes bounds that.
    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._deepseek = DeepSeekClient(model=self.config.model, timeout=int(self.config.timeout_seconds) or 30)
        self._consecutive_failures = 0

    def is_available(self) -> bool:
        if not self.config.enabled:
            return False
        if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            return False
        if self.config.provider == "mock":
            return True
        if self.config.provider in ("deepseek", "auto"):
            return self._deepseek.is_configured
        return False

    def explain_recommendations(
        self,
        query_spans: list[str],
        products: list[dict],
    ) -> tuple[str, dict[str, int]]:
        """Generate human-like transparent justification for recommended products."""
        if not products:
            return "", {"prompt_tokens": 0, "completion_tokens": 0}

        # 1. Deterministic Grounded Fallback
        matched_constraints = ", ".join(query_spans[:3]) if query_spans else "your preferences"
        top_titles = [p.get("text", "").split("\n")[0][:40] for p in products[:2]]
        fallback_msg = f"I selected these options because they match {matched_constraints}."

        if not self.is_available():
            return fallback_msg, {"prompt_tokens": 0, "completion_tokens": 0}

        # 2. LLM Prompt Construction
        prompt = (
            f"You are a helpful shopping assistant. Explain in 1 short, conversational sentence why these "
            f"products match the customer's requested constraints: [{matched_constraints}].\n"
            f"Product titles: {top_titles}\n"
            f"Response:"
        )

        text, usage = self._call_llm(prompt)
        # On an attempted-but-failed call return "" (not fallback_msg): the agent
        # keeps its original clarify() message, so a network failure degrades to
        # the exact baseline conversation rather than a canned sentence.
        return text.strip(), usage

    def rerank_candidates(
        self,
        conversation_text: str,
        candidates: list[dict],
    ) -> tuple[dict[str, float], dict[str, int]]:
        """Perform listwise semantic reranking over candidate products."""
        if not candidates or not self.is_available():
            return {}, {"prompt_tokens": 0, "completion_tokens": 0}

        candidate_lines = []
        for i, c in enumerate(candidates[:15]):
            asin = c.get("parent_asin") or c.get("asin", f"CAND_{i}")
            title = c.get("title") or c.get("text", "").split("\n")[0][:80]
            candidate_lines.append(f"[{asin}] {title}")

        prompt = (
            f"You are an expert e-commerce product search reranker.\n"
            f"Customer Dialogue: \"{conversation_text}\"\n\n"
            f"Candidate Products:\n" + "\n".join(candidate_lines) + "\n\n"
            f"Task: Rank the candidates by how closely they satisfy all constraints in the dialogue.\n"
            f"Output ONLY a JSON array of the product IDs from best match to worst match.\n"
            f"Example format: [\"B09...\", \"B08...\"]\n"
            f"JSON:"
        )

        text, usage = self._call_llm(prompt)
        scores: dict[str, float] = {}
        try:
            clean_text = text.strip()
            if "[" in clean_text and "]" in clean_text:
                json_str = clean_text[clean_text.find("["):clean_text.rfind("]")+1]
                ranked_ids = json.loads(json_str)
                for rank, pid in enumerate(ranked_ids):
                    pid_clean = str(pid).strip()
                    scores[pid_clean] = 1.0 / (rank + 1)
        except Exception:
            pass

        return scores, usage

    def generate_clarification(
        self,
        opening_query: str,
        split_facets: dict[str, list[str]],
        fallback_question: str,
    ) -> tuple[str, dict[str, int]]:
        """Generate a natural conversational clarification question grounded in pool splits."""
        if not self.is_available() or not split_facets:
            return fallback_question, {"prompt_tokens": 0, "completion_tokens": 0}

        facet_desc = "; ".join(f"{attr}: {', '.join(vals)}" for attr, vals in split_facets.items())
        prompt = (
            f"You are a helpful e-commerce shopping assistant helping a customer find: '{opening_query}'.\n"
            f"The available products are split across these attributes: [{facet_desc}].\n"
            f"Ask ONE short, polite, conversational question (under 20 words) asking the customer's preference "
            f"to help narrow down the choices.\n"
            f"Question:"
        )

        text, usage = self._call_llm(prompt)
        return (text.strip() if text else fallback_question), usage

    def _call_llm(self, prompt: str) -> tuple[str, dict[str, int]]:
        """Execute a request against DeepSeek with defensive error handling.

        An empty response counts as a failure toward the circuit breaker
        (``MAX_CONSECUTIVE_FAILURES``); any successful response resets it.
        """
        text, usage = "", {"prompt_tokens": 0, "completion_tokens": 0}
        try:
            if self.config.provider == "mock":
                # Simulated realistic LLM response for testing & CI.
                text = "I selected these options to match your exact requested features and style."
                usage = {"prompt_tokens": len(prompt.split()) * 2, "completion_tokens": 14}
            elif self.config.provider in ("deepseek", "auto") and self._deepseek.is_configured:
                result = self._deepseek.generate(prompt)
                text = result or ""
                usage = self._deepseek.usage_estimate(prompt, text)
        except Exception:
            pass
        if text:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
        return text, usage
