"""Unit tests for src/llm.py - the DeepSeek transport and response parsing.

No test here touches the network: urllib.request.urlopen is monkeypatched.
Live behaviour against the real API is exercised separately (see
docs/team/agent_changes.md Change 16) and is deliberately not part of the
hermetic suite - CI and any network-restricted scoring run must still pass.
"""

from __future__ import annotations

import json
import unittest
from contextlib import contextmanager
from unittest import mock

from src.llm import LLMConfig, LLMReranker


class _FakeResponse:
    """Minimal stand-in for the object urlopen()'s context manager yields."""

    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> None:
        return None


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


@contextmanager
def _patched_urlopen(result=None, raises: Exception | None = None):
    def _fake(_request, timeout=None):  # noqa: ARG001
        if raises is not None:
            raise raises
        return result

    with mock.patch("urllib.request.urlopen", side_effect=_fake):
        yield


class AvailabilityTests(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        self.assertFalse(LLMReranker(LLMConfig()).available)

    def test_disabled_even_with_a_key_present(self, ) -> None:
        with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
            self.assertFalse(LLMReranker(LLMConfig(enabled=False)).available)

    def test_enabled_but_no_key_is_unavailable(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(LLMReranker(LLMConfig(enabled=True)).available)

    def test_enabled_with_key_is_available(self) -> None:
        with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
            self.assertTrue(LLMReranker(LLMConfig(enabled=True)).available)

    def test_custom_key_env_var_name_is_honoured(self) -> None:
        with mock.patch.dict("os.environ", {"OTHER_KEY": "sk-test"}, clear=True):
            config = LLMConfig(enabled=True, api_key_env="OTHER_KEY")
            self.assertTrue(LLMReranker(config).available)


class RankTests(unittest.TestCase):
    def _reranker(self) -> LLMReranker:
        with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
            return LLMReranker(LLMConfig(enabled=True))

    def test_unavailable_returns_none_without_calling_out(self) -> None:
        reranker = LLMReranker(LLMConfig(enabled=False))
        with mock.patch("urllib.request.urlopen") as urlopen:
            self.assertIsNone(reranker.rank("query", [{"asin": "a", "text": "x"}]))
            urlopen.assert_not_called()

    def test_no_candidates_returns_none(self) -> None:
        reranker = self._reranker()
        self.assertIsNone(reranker.rank("query", []))

    def test_happy_path_returns_the_model_order(self) -> None:
        reranker = self._reranker()
        response = _FakeResponse(_chat_response('["b", "a"]'))
        with _patched_urlopen(result=response):
            order = reranker.rank("query", [{"asin": "a", "text": "x"}, {"asin": "b", "text": "y"}])
        self.assertEqual(order, ["b", "a"])

    def test_markdown_fenced_reply_is_unwrapped(self) -> None:
        reranker = self._reranker()
        response = _FakeResponse(_chat_response('```json\n["b", "a"]\n```'))
        with _patched_urlopen(result=response):
            order = reranker.rank("query", [{"asin": "a", "text": "x"}, {"asin": "b", "text": "y"}])
        self.assertEqual(order, ["b", "a"])

    def test_unknown_ids_are_dropped(self) -> None:
        reranker = self._reranker()
        response = _FakeResponse(_chat_response('["b", "made-up-asin", "a"]'))
        with _patched_urlopen(result=response):
            order = reranker.rank("query", [{"asin": "a", "text": "x"}, {"asin": "b", "text": "y"}])
        self.assertEqual(order, ["b", "a"])

    def test_duplicate_ids_are_deduplicated(self) -> None:
        reranker = self._reranker()
        response = _FakeResponse(_chat_response('["a", "a", "b"]'))
        with _patched_urlopen(result=response):
            order = reranker.rank("query", [{"asin": "a", "text": "x"}, {"asin": "b", "text": "y"}])
        self.assertEqual(order, ["a", "b"])

    def test_non_json_reply_returns_none(self) -> None:
        reranker = self._reranker()
        response = _FakeResponse(_chat_response("sure, here you go: a then b"))
        with _patched_urlopen(result=response):
            order = reranker.rank("query", [{"asin": "a", "text": "x"}])
        self.assertIsNone(order)

    def test_json_object_instead_of_array_returns_none(self) -> None:
        reranker = self._reranker()
        response = _FakeResponse(_chat_response('{"a": 1}'))
        with _patched_urlopen(result=response):
            order = reranker.rank("query", [{"asin": "a", "text": "x"}])
        self.assertIsNone(order)

    def test_network_error_returns_none(self) -> None:
        reranker = self._reranker()
        with _patched_urlopen(raises=TimeoutError("timed out")):
            order = reranker.rank("query", [{"asin": "a", "text": "x"}])
        self.assertIsNone(order)

    def test_malformed_response_body_returns_none(self) -> None:
        reranker = self._reranker()
        with _patched_urlopen(result=_FakeResponse({"unexpected": "shape"})):
            order = reranker.rank("query", [{"asin": "a", "text": "x"}])
        self.assertIsNone(order)

    def test_all_ids_unknown_returns_none_not_empty_list(self) -> None:
        reranker = self._reranker()
        response = _FakeResponse(_chat_response('["nope"]'))
        with _patched_urlopen(result=response):
            order = reranker.rank("query", [{"asin": "a", "text": "x"}])
        self.assertIsNone(order)


if __name__ == "__main__":
    unittest.main()
