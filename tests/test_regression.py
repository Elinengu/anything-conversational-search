"""End-to-end scoring regression.

Guards the committed floor. The pipeline has several interacting knobs (route
weights, rerank weights, recommendation timing) and it is easy to make a local
improvement that quietly costs score overall - this test is what catches that.

Runs on the held-out split rather than the full public set: it is the honest
number to defend, and it keeps the test around ten seconds.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402
from tools.sweep import split_samples  # noqa: E402


#: The committed floor: the simple accumulate-and-ask agent measured 0.7811 on the
#: full public set. Any change that drops below this is a regression, not a tweak.
MINIMUM_SCORE = 0.78
MINIMUM_HIT_RATE = 0.88


class RegressionTests(unittest.TestCase):
    result: dict

    @classmethod
    def setUpClass(cls) -> None:
        catalog = REPO_ROOT / "data" / "catalog.jsonl"
        samples = split_samples(load_jsonl(REPO_ROOT / "data" / "public_set.jsonl"), "holdout")
        catalog_ids, categories, products = catalog_index(catalog)
        cls.result = evaluate(Agent(catalog), samples, catalog_ids, categories, products)

    def test_score_holds_above_the_committed_floor(self) -> None:
        score = self.result["recommended_technical_score"]
        self.assertGreaterEqual(score, MINIMUM_SCORE, f"technical score regressed to {score:.4f}")

    def test_hit_rate_holds(self) -> None:
        hit_rate = self.result["hit_rate_at_10"]
        self.assertGreaterEqual(hit_rate, MINIMUM_HIT_RATE, f"hit rate regressed to {hit_rate:.4f}")

    def test_every_scenario_still_works(self) -> None:
        # A collapse concentrated in one scenario - typically intent_override - is
        # the signature of broken state handling, and the overall average can hide it.
        for scenario, metrics in self.result["scenario_metrics"].items():
            with self.subTest(scenario=scenario):
                self.assertGreaterEqual(metrics["hit_rate_at_10"], 0.70, f"{scenario} collapsed")

    def test_offline_path_reports_no_token_usage(self) -> None:
        # The core pipeline must not silently acquire a network dependency.
        self.assertEqual(self.result["reported_token_usage"]["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
