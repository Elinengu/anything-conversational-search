"""Shared LLM adapter for the shopping agent.

This is the only module allowed to talk to the remote DeepSeek API. The
routing, policy, and orchestration layers should depend on this adapter
rather than issuing HTTP requests directly.
"""

from __future__ import annotations

import json
import os
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


if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
else:
    # Fallback: read directly from the repo .env if python-dotenv is not installed.
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


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
        except Exception:
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


def get_llm_client() -> DeepSeekClient:
    """Return the shared default DeepSeek client for the application."""
    return DeepSeekClient()


DEFAULT_LLM_CLIENT = get_llm_client()
