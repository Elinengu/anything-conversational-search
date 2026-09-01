"""Contract and robustness tests for the agent's response envelope.

The evaluator converts any raised exception into a missed session
(``evaluator/local_evaluator.py``), so a crash is silent and expensive. These
tests assert the agent stays inside the published contract even when it is fed
input the contract says it will never see.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from starter.agent import Agent


REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT = json.loads((REPO_ROOT / "docs" / "agent_api_contract.json").read_text())
ALLOWED = set(CONTRACT["turn_response"]["properties"]["ask_attribute"]["enum"]) - {None}


class ContractTests(unittest.TestCase):
    agent: Agent

    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = Agent(REPO_ROOT / "data" / "catalog.jsonl")

    def assert_valid(self, response: object) -> None:
        self.assertIsInstance(response, dict)
        self.assertIsInstance(response["message"], str)

        attribute = response["ask_attribute"]
        self.assertTrue(attribute is None or attribute in ALLOWED, f"illegal attribute {attribute!r}")

        recommendations = response["recommendations"]
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 10)
        seen: set[str] = set()
        for item in recommendations:
            self.assertIsInstance(item, dict)
            parent_asin = item["parent_asin"]
            self.assertIsInstance(parent_asin, str)
            self.assertTrue(parent_asin)
            self.assertNotIn(parent_asin, seen, "duplicate parent_asin in one response")
            seen.add(parent_asin)
            self.assertIn(parent_asin, self.agent.index.products, "recommended id not in catalog")

        usage = response["usage"]
        for key in ("prompt_tokens", "completion_tokens"):
            self.assertIsInstance(usage[key], int)
            self.assertGreaterEqual(usage[key], 0)

    def test_full_session_stays_in_contract(self) -> None:
        self.agent.reset("session-a", {"preference_tags": ["fit"], "summary": "x"})
        messages = [
            "I'm looking for Watches Wrist Watches. Stainless Steel Band",
            "For that, what matters is: Water Resistant; 3 Year Battery.",
            "For that, what matters is: Day / Date Indicator.",
            "I don't have an additional preference for color.",
        ]
        for turn, message in enumerate(messages, start=1):
            self.assert_valid(self.agent.respond("session-a", message, turn, 10))

    def test_hostile_input_never_raises(self) -> None:
        self.agent.reset("session-b", {})
        for turn, message in enumerate(["", "   ", "!!! ??? ***", "a" * 5000, "SELECT * FROM products"], start=1):
            self.assert_valid(self.agent.respond("session-b", message, turn, 10))

    def test_respond_without_reset_is_survivable(self) -> None:
        # The contract guarantees reset() first; a missing session must still not
        # take down the run.
        self.assert_valid(self.agent.respond("never-reset", "I want a blue cotton shirt", 1, 10))

    def test_sessions_do_not_leak_into_each_other(self) -> None:
        self.agent.reset("session-c", {})
        self.agent.reset("session-d", {})
        self.agent.respond("session-c", "I'm looking for leather hiking boots", 1, 10)
        state = self.agent._states["session-d"]
        self.assertEqual(state.turn_count, 0, "state bled across sessions")

    def test_opening_turn_risks_only_one_candidate(self) -> None:
        """Turn 1 guesses, but with a single candidate.

        The agent used to hold every slate until turn 3, because a wide early
        list banks whatever rank the target held at the time. Under sniper
        sizing that reasoning inverts: a one-item slate can only ever score
        rank 1, so an early guess is cheap (a turn) and an early hit is worth
        the full reciprocal rank. What must not happen is a *wide* opening
        slate on no evidence.
        """
        self.agent.reset("session-e", {})
        first = self.agent.respond(
            "session-e", "I'm looking for necklaces, but I'm still exploring.", 1, 10)
        self.assertLessEqual(
            len(first["recommendations"]), 1,
            "risked more than one candidate before any constraint was disclosed")

    def test_top_k_is_respected(self) -> None:
        self.agent.reset("session-f", {})
        self.agent.respond("session-f", "I'm looking for a leather belt", 1, 10)
        self.agent.respond("session-f", "For that, what matters is: full grain leather.", 2, 10)
        third = self.agent.respond("session-f", "For that, what matters is: buckle closure.", 3, 5)
        self.assertLessEqual(len(third["recommendations"]), 5)


if __name__ == "__main__":
    unittest.main()
