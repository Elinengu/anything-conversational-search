"""Unit tests for the isolated stress-observer integration."""

from __future__ import annotations

import random
import unittest

from tools.stress_harness import StressCustomer
from tools.stress_observe.runner import ObservedCustomer, _outcome


CARD = {
    "hard_constraints": ["100% Leather", "color: black"],
    "soft_preferences": ["Buckle closure", "Wide width"],
}
CATEGORIES = {"T": ["Clothing, Shoes & Jewelry", "Men", "Belts"]}


class _AnnotationSink:
    def __init__(self):
        self.annotation = None

    def set_next_annotation(self, annotation):
        self.annotation = annotation


def _customer(**spec):
    defaults = {"paraphrase": "", "browse_gated": False, "decoy": False}
    defaults.update(spec)
    return StressCustomer(
        sample={"scenario_type": "browsing", "user_profile": {}},
        card=CARD,
        behavior={},
        categories=CATEGORIES,
        target="T",
        rng=random.Random(0),
        index_products=None,
        **defaults,
    )


class CustomerAnnotationTests(unittest.TestCase):
    def test_paraphrased_disclosure_keeps_original_constraint(self):
        sink = _AnnotationSink()
        customer = ObservedCustomer(
            _customer(paraphrase="heavy", browse_gated=True), sink
        )
        customer.opening()
        message = customer.reply(1, "material")
        self.assertNotIn("100% Leather", message)
        self.assertEqual(sink.annotation["kind"], "disclosed")
        self.assertEqual(sink.annotation["revealed"], ["100% Leather"])

    def test_gated_broad_question_is_stalled(self):
        sink = _AnnotationSink()
        customer = ObservedCustomer(_customer(browse_gated=True), sink)
        customer.opening()
        customer.reply(1, "other")
        self.assertEqual(sink.annotation, {"kind": "stalled", "revealed": []})


class OutcomeTests(unittest.TestCase):
    def test_best_rank_is_recovered_from_reciprocal_rank(self):
        result = {
            "hit": True,
            "first_hit_turn": 4,
            "reciprocal_rank": 0.25,
            "token_coverage": 0.8,
            "pool_rank": 2,
            "ranked_rank": 4,
        }
        outcome = _outcome(
            {"sample_id": "s", "scenario_type": "browsing"}, result
        )
        self.assertEqual(outcome["best_rank"], 4)


if __name__ == "__main__":
    unittest.main()
