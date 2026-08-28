"""Unit tests for the pipeline stages that carry the scoring behaviour."""

from __future__ import annotations

import unittest

from src.facets import extract
from src.policy import ALLOWED_ATTRIBUTES, FixedPolicy, InfoGainPolicy
from src.rerank import RerankConfig, rerank
from src.router import classify
from src.state import PRE_OVERRIDE_WEIGHT, DialogState
from src.text import constraint_spans, terms


class TextTests(unittest.TestCase):
    def test_spans_survive_punctuation_and_casing(self) -> None:
        spans = constraint_spans("For that, what matters is: Day / Date Indicator; Stainless Steel Band.")
        self.assertIn("day date indicator", spans)
        self.assertIn("stainless steel band", spans)

    def test_spans_are_token_joined_for_substring_matching(self) -> None:
        # "100% Leather" must normalise the same way the index normalises product
        # text, or the reranker's substring match silently never fires.
        self.assertIn("100 leather", constraint_spans("what matters is: 100% Leather."))

    def test_terms_drop_filler_and_deduplicate(self) -> None:
        result = terms("I am looking for a blue blue shirt", drop_boilerplate=True)
        self.assertEqual(result.count("blue"), 1)
        self.assertNotIn("looking", result)


class StateTests(unittest.TestCase):
    def test_override_downweights_rather_than_erases(self) -> None:
        state = DialogState("s")
        state.observe(1, "I'm looking for belts. Buckle closure")
        state.observe(2, "Actually, ignore my earlier preference. What I need is: full grain leather.")
        self.assertEqual(state.override_turn, 2)
        self.assertEqual(state.utterances[0].weight, PRE_OVERRIDE_WEIGHT)
        # The earlier turn is still reachable: erasing it would discard signal.
        self.assertIn("buckle", state.full_text().lower())
        self.assertNotIn("buckle", state.focused_text().lower())

    def test_declined_attribute_is_not_asked_again(self) -> None:
        state = DialogState("s")
        state.observe(1, "I'm looking for socks")
        state.record_ask("budget")
        state.observe(2, "I don't have an additional preference for budget.")
        self.assertIn("budget", state.dead_attributes)

    def test_productivity_tracks_new_information_only(self) -> None:
        state = DialogState("s")
        state.observe(1, "I'm looking for watches")
        state.observe(2, "For that, what matters is: water resistant case.")
        self.assertEqual(state.productive_turns, 1)
        state.observe(3, "For that, what matters is: water resistant case.")
        self.assertEqual(state.productive_turns, 1, "repeat disclosure counted as new")


class FacetTests(unittest.TestCase):
    def test_extracts_typed_values(self) -> None:
        values = extract({
            "text": "mens leather belt black buckle closure for work",
            "store": "Acme", "price": 24.99,
            "categories": ["Clothing, Shoes & Jewelry", "Men", "Belts"],
        })
        self.assertEqual(values["material"], "leather")
        self.assertEqual(values["color"], "black")
        self.assertEqual(values["budget"], "15 to 30")
        self.assertEqual(values["category"], "belts")

    def test_absent_price_yields_no_budget_facet(self) -> None:
        # 78.9% of this catalog has a null price; the policy relies on that
        # absence to stop asking about budget.
        self.assertNotIn("budget", extract({"text": "a scarf", "price": None, "categories": []}))


class PolicyTests(unittest.TestCase):
    def test_policies_only_emit_legal_attributes(self) -> None:
        state = DialogState("s")
        state.observe(1, "I'm looking for a jacket")
        for policy in (FixedPolicy(), InfoGainPolicy(_StubFacets())):
            self.assertIn(policy.select(state, [("A", 1.0), ("B", 0.5)]), ALLOWED_ATTRIBUTES)

    def test_fixed_policy_skips_declined_attributes(self) -> None:
        state = DialogState("s")
        state.observe(1, "hello")
        state.dead_attributes.add("other")
        self.assertNotEqual(FixedPolicy(("other", "feature")).select(state, []), "other")

    def test_gain_ratio_is_not_dominated_by_cardinality(self) -> None:
        # Raw entropy favours brand purely because the catalog holds thousands of
        # distinct stores; the gain ratio must not.
        policy = InfoGainPolicy(_StubFacets())
        state = DialogState("s", profile={})
        state.observe(1, "I'm looking for a watch")
        state.productive_turns = 2
        state.asked.append("other")
        gains = policy.scores(state, [(str(i), 1.0) for i in range(20)])
        self.assertLess(gains["brand"], gains["color"])


class RerankTests(unittest.TestCase):
    def test_verbatim_span_outranks_a_stronger_retrieval_score(self) -> None:
        index = _StubIndex({
            "weak": {"text": "a watch with a stainless steel band and date window", "average_rating": 4.0,
                     "rating_number": 10},
            "strong": {"text": "an unrelated cotton shirt", "average_rating": 5.0, "rating_number": 9999},
        })
        state = DialogState("s")
        state.observe(1, "I'm looking for watches")
        state.observe(2, "For that, what matters is: stainless steel band.")
        ranked = rerank(index, state, [("strong", 1.0), ("weak", 0.9)], RerankConfig())
        self.assertEqual(ranked[0][0], "weak")

    def test_rerank_is_a_no_op_without_disclosed_spans(self) -> None:
        index = _StubIndex({"a": {"text": "x"}, "b": {"text": "y"}})
        state = DialogState("s")
        state.observe(1, "I'm looking for shoes")
        original = [("a", 1.0), ("b", 0.5)]
        self.assertEqual(rerank(index, state, original, RerankConfig()), original)


class RouterTests(unittest.TestCase):
    def test_cue_based_classification(self) -> None:
        self.assertEqual(classify("I'm looking for boots. A key requirement is: leather.").name, "buying")
        self.assertEqual(classify("I'm looking for boots, but I'm still exploring.").name, "browsing")

    def test_unknown_phrasing_defaults_to_browsing(self) -> None:
        # Misreading a vague customer as a buyer commits to constraints they never
        # stated; the reverse costs at most one question.
        self.assertEqual(classify("hello there").name, "browsing")


class _StubFacets:
    """Two colours, many brands - the cardinality trap in miniature."""

    def get(self, parent_asin: str) -> dict[str, str]:
        return {"color": "black" if int(parent_asin) % 2 else "white", "brand": f"brand-{parent_asin}"}


class _StubIndex:
    def __init__(self, products: dict[str, dict]) -> None:
        self.products = products


if __name__ == "__main__":
    unittest.main()
