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


if __name__ == "__main__":
    unittest.main()
