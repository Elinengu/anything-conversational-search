"""Unit tests for src/llm.py - the DeepSeek transport and response parsing.

No test here touches the network: urllib.request.urlopen is monkeypatched.
Live behaviour against the real API is exercised separately (see
docs/team/agent_changes.md Change 17) and is deliberately not part of the
hermetic suite - CI and any network-restricted scoring run must still pass.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from src.llm import LLMConfig, LLMReranker, _read_dotenv


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
        # Both sources must be silenced, not just the environment: since the
        # .env fallback was added, clearing os.environ alone left this test
        # passing only on machines without a real key file - i.e. it failed for
        # anyone set up the way llm_config_readme.md tells them to set up.
        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch("src.llm._read_dotenv", return_value=""):
            self.assertFalse(LLMReranker(LLMConfig(enabled=True)).available)

    def test_dotenv_alone_is_enough_to_be_available(self) -> None:
        """The environment being empty is not itself a reason to be unavailable."""
        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch("src.llm._read_dotenv", return_value="sk-from-dotenv"):
            self.assertTrue(LLMReranker(LLMConfig(enabled=True)).available)

    def test_enabled_with_key_is_available(self) -> None:
        with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
            self.assertTrue(LLMReranker(LLMConfig(enabled=True)).available)

    def test_custom_key_env_var_name_is_honoured(self) -> None:
        with mock.patch.dict("os.environ", {"OTHER_KEY": "sk-test"}, clear=True):
            config = LLMConfig(enabled=True, api_key_env="OTHER_KEY")
            self.assertTrue(LLMReranker(config).available)


class DotenvFallbackTests(unittest.TestCase):
    """RerankConfig.llm_weight aside, LLMReranker falls back to a repo-root
    .env file only when the environment variable itself isn't set - never
    touches the real repo root's .env (if one exists locally); every test
    here points _read_dotenv/_REPO_ROOT at an isolated temp directory."""

    def _write_env_file(self, tmpdir: str, content: str) -> None:
        (Path(tmpdir) / ".env").write_text(content, encoding="utf-8")

    def test_read_dotenv_missing_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("src.llm._REPO_ROOT", Path(tmpdir)):
                self.assertEqual(_read_dotenv("DEEPSEEK_API_KEY"), "")

    def test_read_dotenv_plain_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_env_file(tmpdir, "DEEPSEEK_API_KEY=sk-from-dotenv\n")
            with mock.patch("src.llm._REPO_ROOT", Path(tmpdir)):
                self.assertEqual(_read_dotenv("DEEPSEEK_API_KEY"), "sk-from-dotenv")

    def test_read_dotenv_export_prefix_and_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_env_file(
                tmpdir,
                '# a comment\nexport DEEPSEEK_API_KEY="sk-quoted"\n\nOTHER=1\n',
            )
            with mock.patch("src.llm._REPO_ROOT", Path(tmpdir)):
                self.assertEqual(_read_dotenv("DEEPSEEK_API_KEY"), "sk-quoted")

    def test_read_dotenv_missing_key_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_env_file(tmpdir, "SOME_OTHER_VAR=x\n")
            with mock.patch("src.llm._REPO_ROOT", Path(tmpdir)):
                self.assertEqual(_read_dotenv("DEEPSEEK_API_KEY"), "")

    def test_reranker_falls_back_to_dotenv_when_env_var_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_env_file(tmpdir, "DEEPSEEK_API_KEY=sk-from-dotenv\n")
            with mock.patch("src.llm._REPO_ROOT", Path(tmpdir)), \
                 mock.patch.dict("os.environ", {}, clear=True):
                reranker = LLMReranker(LLMConfig(enabled=True))
                self.assertTrue(reranker.available)
                self.assertEqual(reranker._api_key, "sk-from-dotenv")

    def test_exported_env_var_takes_precedence_over_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_env_file(tmpdir, "DEEPSEEK_API_KEY=sk-from-dotenv\n")
            with mock.patch("src.llm._REPO_ROOT", Path(tmpdir)), \
                 mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-from-export"}):
                reranker = LLMReranker(LLMConfig(enabled=True))
                self.assertEqual(reranker._api_key, "sk-from-export")

    def test_no_env_var_and_no_dotenv_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("src.llm._REPO_ROOT", Path(tmpdir)), \
                 mock.patch.dict("os.environ", {}, clear=True):
                self.assertFalse(LLMReranker(LLMConfig(enabled=True)).available)

    def test_dotenv_never_consulted_when_disabled(self) -> None:
        """enabled=False must short-circuit before even looking at .env."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_env_file(tmpdir, "DEEPSEEK_API_KEY=sk-from-dotenv\n")
            with mock.patch("src.llm._REPO_ROOT", Path(tmpdir)), \
                 mock.patch.dict("os.environ", {}, clear=True):
                self.assertFalse(LLMReranker(LLMConfig(enabled=False)).available)


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


class UsageAccountingTests(unittest.TestCase):
    """reported_token_usage is a metric the competition collects, so a layer
    that spends tokens must say so. This was hardcoded to zero before."""

    def _client(self):
        with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
            return LLMReranker(LLMConfig(enabled=True))

    def test_usage_starts_at_zero(self) -> None:
        self.assertEqual(self._client().usage,
                         {"prompt_tokens": 0, "completion_tokens": 0})

    def test_usage_accumulates_what_the_provider_reports(self) -> None:
        client = self._client()
        body = {"choices": [{"message": {"content": '["A1"]'}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 7}}
        with _patched_urlopen(result=_FakeResponse(body)):
            client.rank("q", [{"asin": "A1", "text": "t"}])
            client.rank("q2", [{"asin": "A1", "text": "t"}])
        self.assertEqual(client.usage, {"prompt_tokens": 240, "completion_tokens": 14})

    def test_a_response_without_usage_is_not_an_error(self) -> None:
        client = self._client()
        body = {"choices": [{"message": {"content": '["A1"]'}}]}
        with _patched_urlopen(result=_FakeResponse(body)):
            self.assertEqual(client.rank("q", [{"asin": "A1", "text": "t"}]), ["A1"])
        self.assertEqual(client.usage, {"prompt_tokens": 0, "completion_tokens": 0})


class CacheTests(unittest.TestCase):
    def test_a_cache_hit_replays_without_calling_or_recounting_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
                client = LLMReranker(LLMConfig(enabled=True, cache_dir=tmp))
            body = {"choices": [{"message": {"content": '["A2","A1"]'}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 5}}
            items = [{"asin": "A1", "text": "t1"}, {"asin": "A2", "text": "t2"}]
            with _patched_urlopen(result=_FakeResponse(body)):
                first = client.rank("q", items)
            # A second call must not need the network at all.
            with _patched_urlopen(raises=AssertionError("network was used")):
                second = client.rank("q", items)
            self.assertEqual(first, ["A2", "A1"])
            self.assertEqual(second, first)
            # The tokens were spent once; counting them twice would inflate a
            # metric the competition collects.
            self.assertEqual(client.usage, {"prompt_tokens": 100, "completion_tokens": 5})
            self.assertEqual(client.stats["cache_hits"], 1)


class ExtractionTests(unittest.TestCase):
    """Arm B: reading the opening, which is the job the model is better at."""

    def _client(self):
        with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
            return LLMReranker(LLMConfig(enabled=True))

    def test_extracts_structured_search_material(self) -> None:
        payload = ('{"category": "necklaces", "constraints": ["alloy"], '
                   '"expanded_terms": ["pendant", "chain"]}')
        body = {"choices": [{"message": {"content": payload}}]}
        with _patched_urlopen(result=_FakeResponse(body)):
            out = self._client().extract("I'm looking for Jewelry Necklaces.")
        self.assertEqual(out["category"], "necklaces")
        self.assertEqual(out["constraints"], ["alloy"])
        self.assertIn("pendant", out["expanded_terms"])

    def test_every_failure_is_None_not_an_exception(self) -> None:
        client = self._client()
        with _patched_urlopen(raises=TimeoutError("slow")):
            self.assertIsNone(client.extract("anything"))
        with _patched_urlopen(result=_FakeResponse(
                {"choices": [{"message": {"content": "not json"}}]})):
            self.assertIsNone(client.extract("anything"))
        with _patched_urlopen(result=_FakeResponse(
                {"choices": [{"message": {"content": "{}"}}]})):
            self.assertIsNone(client.extract("anything"), "an empty parse is no opinion")

    def test_unavailable_client_never_calls(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch("src.llm._read_dotenv", return_value=""):
            self.assertIsNone(LLMReranker(LLMConfig(enabled=True)).extract("x"))
