"""Unit tests for the LLM integration layer."""

import unittest
from src.llm import LLMClient, LLMConfig
from starter.agent import Agent, AgentConfig


class TestLLMLayer(unittest.TestCase):
    def test_offline_fallback(self):
        # Default offline config: zero network, zero tokens, instant fallback
        config = LLMConfig(enabled=False)
        client = LLMClient(config)
        self.assertFalse(client.is_available())

        spans = ["waterproof", "leather", "size 10"]
        products = [{"text": "Timberland Mens Waterproof Leather Boots Size 10"}]
        explanation, usage = client.explain_recommendations(spans, products)

        self.assertIn("waterproof, leather, size 10", explanation)
        self.assertEqual(usage["prompt_tokens"], 0)
        self.assertEqual(usage["completion_tokens"], 0)

    def test_mock_llm_mode(self):
        # Mock mode simulates live LLM API for testing & presentation without network
        config = LLMConfig(enabled=True, provider="mock")
        client = LLMClient(config)
        self.assertTrue(client.is_available())

        spans = ["running", "breathable mesh"]
        products = [{"text": "Nike Air Zoom Pegasus Lightweight Running Shoe"}]
        explanation, usage = client.explain_recommendations(spans, products)

        self.assertTrue(len(explanation) > 10)
        self.assertGreater(usage["prompt_tokens"], 0)
        self.assertGreater(usage["completion_tokens"], 0)

    def test_agent_with_mock_llm(self):
        agent_config = AgentConfig(
            llm=LLMConfig(enabled=True, provider="mock"),
            first_recommend_turn=1,
            confidence_margin=0.0,  # force recommendation
        )
        agent = Agent("data/catalog.jsonl", config=agent_config)
        agent.reset("session_mock", {})
        response = agent.respond("session_mock", "Looking for Nike running shoes.", turn=1, top_k=5)

        self.assertIn("recommendations", response)
        self.assertIn("usage", response)
        self.assertIn("prompt_tokens", response["usage"])
    def test_llm_reranking(self):
        config = LLMConfig(enabled=True, provider="mock")
        client = LLMClient(config)
        cands = [
            {"parent_asin": "B001", "title": "Nike Air Running Shoes"},
            {"parent_asin": "B002", "title": "Adidas Ultraboost"},
        ]
        scores, usage = client.rerank_candidates("Looking for Nike shoes", cands)
        self.assertIsInstance(scores, dict)
        self.assertGreater(usage["prompt_tokens"], 0)


class TestFallbackGuarantee(unittest.TestCase):
    """Drift these tests catch: an LLM failure that changes the scored behavior.

    The contract is that any failed call - network down, quota exhausted, garbage
    output - degrades to the exact baseline envelope: same message, same
    ask_attribute, same recommendations, zero reported tokens.
    """

    TURNS = [
        "I'm looking for casual pants. A key requirement is: cotton blend.",
        "For that, what matters is: relaxed fit; machine washable.",
        "For that, what matters is: dark colors.",
    ]

    @staticmethod
    def _run_session(config):
        agent = Agent("data/catalog.jsonl", config=config)
        agent.reset("fallback_test", {})
        envelopes = []
        for turn, message in enumerate(TestFallbackGuarantee.TURNS, start=1):
            envelopes.append(agent.respond("fallback_test", message, turn=turn, top_k=10))
        return agent, envelopes

    def test_network_failure_reproduces_baseline_envelopes(self):
        import urllib.error
        import urllib.request

        _, baseline = self._run_session(AgentConfig())

        from src.rerank import RerankConfig
        llm_config = AgentConfig(
            rerank=RerankConfig(llm_weight=0.5),
            llm=LLMConfig(enabled=True, provider="gemini"),
        )

        def dead_network(*args, **kwargs):
            raise urllib.error.URLError("network unreachable (simulated)")

        real_urlopen = urllib.request.urlopen
        urllib.request.urlopen = dead_network
        try:
            agent, degraded = self._run_session(llm_config)
            # A key may be absent in CI; force the client to believe one exists so
            # the network path is actually attempted.
            if not agent.llm.gemini_key:
                agent.llm.gemini_key = "test-key"
                agent.reset("fallback_test", {})
                degraded = []
                for turn, message in enumerate(self.TURNS, start=1):
                    degraded.append(agent.respond("fallback_test", message, turn=turn, top_k=10))
        finally:
            urllib.request.urlopen = real_urlopen

        self.assertEqual(baseline, degraded)
        for envelope in degraded:
            self.assertEqual(envelope["usage"], {"prompt_tokens": 0, "completion_tokens": 0})

    def test_circuit_breaker_stops_retrying(self):
        import urllib.error
        import urllib.request

        client = LLMClient(LLMConfig(enabled=True, provider="gemini"))
        client.gemini_key = client.gemini_key or "test-key"

        def dead_network(*args, **kwargs):
            raise urllib.error.URLError("network unreachable (simulated)")

        real_urlopen = urllib.request.urlopen
        urllib.request.urlopen = dead_network
        try:
            self.assertTrue(client.is_available())
            for _ in range(LLMClient.MAX_CONSECUTIVE_FAILURES):
                client._call_llm("test")
            self.assertFalse(client.is_available())
        finally:
            urllib.request.urlopen = real_urlopen

    def test_failed_explanation_keeps_original_message(self):
        client = LLMClient(LLMConfig(enabled=True, provider="gemini"))
        client.gemini_key = client.gemini_key or "test-key"
        client._call_llm = lambda prompt: ("", {"prompt_tokens": 0, "completion_tokens": 0})
        text, _ = client.explain_recommendations(["cotton"], [{"text": "Cotton Pants"}])
        self.assertEqual(text, "")


if __name__ == "__main__":
    unittest.main()
