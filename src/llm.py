"""Optional S6 layer - LLM semantic reranking (DeepSeek).

Tier 2, item #3 of docs/team/ideas_to_integrate_llm.md: "A cross-encoder ... or
a listwise LLM prompt scores the final shortlist; fused with the existing
rerank() score (not replacing it), local scorer as fallback." That design
constraint is the whole contract of this module:

  * **Off by default.** ``LLMConfig.enabled=False`` and ``RerankConfig.
    llm_weight=0.0`` both have to be turned on for a single byte to go over
    the network. Every existing config, test and the shipped default keep
    calling the offline BM25 + span pipeline exactly as before - see
    ``README.md`` "Disclosure": the submission rules reserve the right to
    score under network restrictions, and the offline path is the floor that
    guarantee depends on.
  * **Fail closed, always.** Every way this can go wrong - no API key, no
    network, a timeout, a non-200 response, a reply that isn't the JSON array
    asked for, an id the model invented - is caught here and turned into
    ``None``. The caller (``src/rerank.py``) treats ``None`` as "no opinion"
    and keeps the lexical order untouched. There is no path where a flaky
    model call can make a turn score worse than turning the layer off.
  * **Stdlib only.** ``urllib.request``, not ``requests`` - this stays an
    optional layer a bare Python interpreter can still import, matching
    ``src/embed.py``'s pattern of degrading to unavailable rather than adding
    a hard dependency.

Talks to DeepSeek's OpenAI-compatible chat-completions endpoint
(https://api.deepseek.com/chat/completions, model "deepseek-chat"). The API
key is read from an environment variable (default ``DEEPSEEK_API_KEY``),
falling back to a ``.env`` file at the repo root if the process environment
doesn't have it (`` .env`` is in ``.gitignore`` - see ``_read_dotenv``) -
never from a config file or a repo-tracked default, so it can never be
committed by accident either way.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASE_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_dotenv(name: str) -> str:
    """Best-effort ``.env`` fallback for one variable, repo root only.

    Not a general dotenv implementation - just enough to read
    ``KEY=value`` / ``export KEY=value`` lines, skip blanks and ``#``
    comments, and strip one layer of surrounding quotes. Never raises: a
    missing file, a missing key, or a malformed line all just mean "not
    found here", exactly like a missing environment variable does.
    """
    try:
        text = (_REPO_ROOT / ".env").read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if key != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value
    return ""


@dataclass
class LLMConfig:
    """Everything needed to talk to the reranking model. Off unless asked."""

    enabled: bool = False
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    #: Name of the environment variable holding the API key - never the key
    #: itself, so nothing secret can end up in a config default or a sweep row.
    api_key_env: str = "DEEPSEEK_API_KEY"
    #: Per-call socket timeout. A turn that hangs past this degrades to the
    #: lexical order exactly like any other failure - see module docstring.
    timeout: float = 8.0
    max_tokens: int = 400
    #: 0.0 - the model is doing a ranking judgment, not open-ended generation;
    #: determinism here matters more than diversity.
    temperature: float = 0.0


class LLMReranker:
    """Thin, fail-closed client for one listwise reranking call per turn."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._api_key = ""
        if self.config.enabled:
            # An explicitly exported environment variable always wins - the
            # .env file is a convenience fallback for when it isn't set, not
            # an override of it.
            self._api_key = os.environ.get(self.config.api_key_env, "") or _read_dotenv(
                self.config.api_key_env
            )

    @property
    def available(self) -> bool:
        """True only when explicitly enabled AND a key is actually present.

        ``RerankConfig.llm_weight`` still has to be > 0.0 on top of this -
        ``available`` says the *transport* is usable, not that a caller wants
        it used this turn.
        """
        return bool(self.config.enabled and self._api_key)

    def rank(self, query_text: str, candidates: list[dict]) -> list[str] | None:
        """Ask the model to order ``candidates`` best-match-first.

        ``candidates`` is a list of ``{"asin": str, "text": str}``. Returns a
        de-duplicated list of the candidates' asins (a permutation, or a
        subset if the model dropped some) - or ``None`` on any failure at
        all, which the caller must treat identically to "the model had no
        opinion this turn" and leave the existing order alone.
        """
        if not self.available or not candidates:
            return None
        try:
            content = self._call(query_text, candidates)
            order = self._parse_order(content)
        except Exception:
            return None
        valid_asins = {item["asin"] for item in candidates}
        seen: set[str] = set()
        cleaned: list[str] = []
        for asin in order:
            if asin in valid_asins and asin not in seen:
                cleaned.append(asin)
                seen.add(asin)
        return cleaned or None

    # ---- internals ------------------------------------------------------------

    def _call(self, query_text: str, candidates: list[dict]) -> str:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": self._prompt(query_text, candidates)}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        request = urllib.request.Request(
            self.config.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]

    @staticmethod
    def _prompt(query_text: str, candidates: list[dict]) -> str:
        # 220 chars keeps a depth-8 prompt small - the product record's own
        # ``text`` field already leads with title and key attributes, so the
        # truncation rarely costs the details a ranking judgment needs.
        listing = "\n".join(
            f"{item['asin']}: {item['text'][:220]}" for item in candidates
        )
        return (
            "You are reranking candidate products for a shopping assistant.\n"
            "Customer request, accumulated over the conversation so far:\n"
            f"{query_text}\n\n"
            "Candidates (id: description):\n"
            f"{listing}\n\n"
            "Return ONLY a JSON array of the candidate ids, ordered from most "
            "to least likely to be what the customer wants. Include every id "
            "exactly once. No other text, no markdown fence."
        )

    @staticmethod
    def _parse_order(content: str) -> list[str]:
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:]
        data = json.loads(content.strip())
        if not isinstance(data, list):
            raise ValueError("expected a JSON array of candidate ids")
        return [str(item) for item in data]
