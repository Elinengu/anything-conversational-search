"""LLM Layer: Transparent Explanations, Grounded Clarifications & Semantic Extraction.

This module provides pluggable LLM integrations conforming to the competition specification:
1. Grounded Clarification Generation: Conversational, pool-aware clarification prompts.
2. Transparent Recommendation Explanations: Natural justifications for recommended products.
3. Paraphrase-Robust Slot Extraction: Semantic attribute extraction for unconventional queries.
4. Guaranteed Offline Fallback: Zero latency, zero cost deterministic fallback when offline.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load_dotenv_if_present() -> None:
    """Load key-value pairs from .env if present without requiring third-party libraries."""
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


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    enabled: bool = False
    provider: str = "auto"  # "gemini", "openai", "ollama", "mock", "auto"
    model: str = "gemini-3.6-flash"
    temperature: float = 0.2
    max_tokens: int = 1000
    timeout_seconds: float = 8.0
    endpoint_url: str | None = None  # e.g. "http://localhost:11434/v1/chat/completions" for Ollama


class LLMClient:
    """Multi-provider LLM client with built-in deterministic offline fallback."""

    #: Consecutive failed calls after which the client stops trying for its lifetime.
    #: A dead-but-slow network otherwise costs (timeout x model fallbacks) on every
    #: turn, and evaluator timeouts count as misses; three strikes bounds that.
    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(self, config: LLMConfig | None = None) -> None:
        _load_dotenv_if_present()
        self.config = config or LLMConfig()
        self.gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self._consecutive_failures = 0

    def is_available(self) -> bool:
        if not self.config.enabled:
            return False
        if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            return False
        if self.config.provider == "mock":
            return True
        if self.config.provider in ("gemini", "auto") and self.gemini_key:
            return True
        if self.config.provider in ("openai", "auto") and self.openai_key:
            return True
        if self.config.provider == "ollama" and self.config.endpoint_url:
            return True
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
        """Execute request across configured providers with defensive error handling.

        An empty response counts as a failure toward the circuit breaker
        (``MAX_CONSECUTIVE_FAILURES``); any successful response resets it.
        """
        text, usage = "", {"prompt_tokens": 0, "completion_tokens": 0}
        try:
            if self.config.provider in ("gemini", "auto") and self.gemini_key:
                text, usage = self._call_gemini(prompt)
            elif self.config.provider in ("openai", "auto") and self.openai_key:
                text, usage = self._call_openai(prompt)
            elif self.config.provider == "ollama" or (self.config.endpoint_url and self.config.enabled):
                text, usage = self._call_openai_compatible(prompt)
            elif self.config.provider == "mock":
                # Simulated realistic LLM response for testing & CI
                mock_text = "I selected these options to match your exact requested features and style."
                text, usage = mock_text, {"prompt_tokens": len(prompt.split()) * 2, "completion_tokens": 14}
        except Exception:
            pass
        if text:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
        return text, usage

    def _call_gemini(self, prompt: str) -> tuple[str, dict[str, int]]:
        models_to_try = [self.config.model, "gemini-3.6-flash", "gemini-3.7-flash", "gemini-flash-latest"]
        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": self.config.temperature,
                    "maxOutputTokens": self.config.max_tokens,
                },
            }
            headers = {"Content-Type": "application/json", "x-goog-api-key": self.gemini_key}
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    text = ""
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            text = parts[0].get("text", "")
                    usage_meta = data.get("usageMetadata", {})
                    prompt_tokens = usage_meta.get("promptTokenCount", len(prompt.split()) * 2)
                    comp_tokens = usage_meta.get("candidatesTokenCount", len(text.split()) * 2)
                    return text.strip(), {"prompt_tokens": prompt_tokens, "completion_tokens": comp_tokens}
            except Exception:
                continue
        return "", {"prompt_tokens": 0, "completion_tokens": 0}

    def _call_openai(self, prompt: str) -> tuple[str, dict[str, int]]:
        endpoint = "https://api.openai.com/v1/chat/completions"
        return self._call_openai_compatible(prompt, endpoint=endpoint, auth_header=f"Bearer {self.openai_key}")

    def _call_openai_compatible(
        self,
        prompt: str,
        endpoint: str | None = None,
        auth_header: str | None = None,
    ) -> tuple[str, dict[str, int]]:
        url = endpoint or self.config.endpoint_url or "http://localhost:11434/v1/chat/completions"
        payload = {
            "model": self.config.model if endpoint else "llama3.2:1b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if auth_header:
            headers["Authorization"] = auth_header

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", len(prompt.split()) * 2)
            comp_tokens = usage.get("completion_tokens", len(text.split()) * 2)
            return text, {"prompt_tokens": prompt_tokens, "completion_tokens": comp_tokens}
