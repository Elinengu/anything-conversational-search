"""Unit tests for the pipeline stages that carry the scoring behaviour."""

from __future__ import annotations

import unittest

from src.facets import extract
from src.policy import ALLOWED_ATTRIBUTES, FixedPolicy, InfoGainPolicy
from src.rerank import RerankConfig, rerank
from src.router import classify, detect_turn_intent, extract_opening_facets
from src.state import PRE_OVERRIDE_WEIGHT, DialogState
from src.text import constraint_spans, pair_spans, terms


class TextTests(unittest.TestCase):
    def test_spans_survive_punctuation_and_casing(self) -> None:
        spans = constraint_spans("For that, what matters is: Day / Date Indicator; Stainless Steel Band.")
        self.assertIn("day date indicator", spans)
        self.assertIn("stainless steel band", spans)

    def test_spans_are_token_joined_for_substring_matching(self) -> None:
        # "100% Leather" must normalise the same way the index normalises product
        # text, or the reranker's substring match silently never fires.
        self.assertIn("100 leather", constraint_spans("what matters is: 100% Leather."))

    def test_pair_spans_keep_key_value_associations(self) -> None:
        msg = ("For that, what matters is: color: grey; Heather Grey: 90% Cotton, "
               "10% Polyester; All Other Heathers: 50% Cotton, 50% Polyester.")
        spans = pair_spans(msg)
        self.assertIn("heather grey 90 cotton 10 polyester", spans)
        self.assertIn("all other heathers 50 cotton 50 polyester", spans)
        # Fragments produced by constraint_spans must NOT reappear here.
        self.assertNotIn("90 cotton", spans)

    def test_pair_spans_split_on_sentence_separators_only(self) -> None:
        spans = pair_spans("Solid colors: 100% Cotton; Machine Wash Warm Water.")
        self.assertIn("solid colors 100 cotton", spans)
        self.assertIn("machine wash warm water", spans)

    def test_pair_spans_strip_leading_filler(self) -> None:
        # The simulator framing sits before the first colon; without the strip
        # it would glue itself to the first pair.
        spans = pair_spans("For that, what matters is: Heather Grey: 90% Cotton.")
        self.assertIn("heather grey 90 cotton", spans)
        self.assertNotIn("for that what matters is heather grey 90 cotton", spans)
        # Two-token leftovers (e.g. "color grey" after the strip) stay below
        # min_words - that association is carried by facet extraction instead.
        self.assertEqual(pair_spans("For that, what matters is: color: grey."), [])

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

    def test_conflicting_facet_is_demoted(self) -> None:
        # Same span coverage; only "wrong" resolves a colour that contradicts
        # the stated grey.
        index = _StubIndex({
            "wrong": {"text": "cotton shirt classic fit black only"},
            "right": {"text": "cotton shirt classic fit heather grey"},
        })
        state = DialogState("s")
        state.observe(1, "I'm looking for shirts")
        state.observe(2, "For that, what matters is: cotton shirt; color: grey.")
        ranked = rerank(index, state, [("wrong", 1.0), ("right", 0.9)], RerankConfig())
        self.assertEqual(ranked[0][0], "right")

    def test_multi_value_product_is_not_punished(self) -> None:
        # extract() keeps only the first colour (black), but the text still
        # contains grey - the substring guard must keep the penalty at zero.
        multi = {"text": "reversible belt black grey two tone"}
        plain = {"text": "reversible belt grey two tone"}
        index = _StubIndex({"multi": multi, "plain": plain})
        state = DialogState("s")
        state.observe(1, "I'm looking for belts")
        state.observe(2, "For that, what matters is: reversible belt; color: grey.")
        with_conflict = RerankConfig(facet_conflict_weight=1.0)
        without = RerankConfig(facet_conflict_weight=0.0)
        pool = [("multi", 1.0), ("plain", 0.9)]
        self.assertEqual(
            rerank(index, state, pool, with_conflict),
            rerank(index, state, pool, without),
        )

    def test_silence_about_a_facet_is_not_a_conflict(self) -> None:
        # "mute" never mentions any colour, so it does not resolve the facet
        # and must not be penalised relative to a colour-free config.
        index = _StubIndex({
            "mute": {"text": "cotton shirt classic fit"},
            "match": {"text": "cotton shirt classic fit grey"},
        })
        state = DialogState("s")
        state.observe(1, "I'm looking for shirts")
        state.observe(2, "For that, what matters is: cotton shirt; color: grey.")
        pool = [("mute", 1.0), ("match", 0.9)]
        with_conflict = rerank(index, state, pool, RerankConfig(facet_conflict_weight=1.0))
        without = rerank(index, state, pool, RerankConfig(facet_conflict_weight=0.0))
        self.assertEqual(with_conflict, without)

    def test_override_discards_stale_facet_for_conflict_scoring(self) -> None:
        # After the customer reverses from black to grey, a grey-only product
        # must not be penalised for contradicting the discarded black.
        index = _StubIndex({
            "old": {"text": "leather wallet slim black"},
            "new": {"text": "leather wallet slim grey"},
        })
        state = DialogState("s")
        state.observe(1, "I'm looking for wallets")
        state.observe(2, "For that, what matters is: leather wallet; color: black.")
        state.observe(3, "Actually, ignore my earlier preference. What I need is: color: grey.")
        ranked = rerank(index, state, [("old", 1.0), ("new", 0.9)],
                        RerankConfig(facet_conflict_weight=1.0))
        self.assertEqual(ranked[0][0], "new")

    def test_intact_association_outranks_recombined_fragments(self) -> None:
        # Both candidates contain every fragment ("heather grey", "90 cotton",
        # "10 polyester"), but only "intact" states the composition about that
        # colour. Fragment coverage ties; the pair span decides.
        index = _StubIndex({
            "shuffled": {"text": "tee in heather grey solid is 90 cotton trim 10 polyester"},
            "intact": {"text": "tee heather grey 90 cotton 10 polyester classic"},
        })
        state = DialogState("s")
        state.observe(1, "I'm looking for tees")
        state.observe(2, "For that, what matters is: Heather Grey: 90% Cotton, 10% Polyester.")
        ranked = rerank(index, state, [("shuffled", 1.0), ("intact", 0.9)],
                        RerankConfig(pair_weight=0.8))
        self.assertEqual(ranked[0][0], "intact")

    def test_span_matching_is_word_bounded(self) -> None:
        # "90 cotton" must not match "190 cotton".
        index = _StubIndex({
            "midtoken": {"text": "thread count 190 cotton sateen sheet"},
            "real": {"text": "soft 90 cotton 10 polyester blend tee"},
        })
        state = DialogState("s")
        state.observe(1, "I'm looking for tees")
        state.observe(2, "For that, what matters is: 90% Cotton.")
        ranked = rerank(index, state, [("midtoken", 1.0), ("real", 0.9)], RerankConfig())
        self.assertEqual(ranked[0][0], "real")

    def test_pair_weight_zero_reproduces_fragment_ranking(self) -> None:
        index = _StubIndex({
            "a": {"text": "tee heather grey 90 cotton 10 polyester"},
            "b": {"text": "tee heather grey solid 90 cotton and 10 polyester"},
        })
        state = DialogState("s")
        state.observe(1, "I'm looking for tees")
        state.observe(2, "For that, what matters is: Heather Grey: 90% Cotton, 10% Polyester.")
        pool = [("a", 1.0), ("b", 0.5)]
        ranked = rerank(index, state, pool, RerankConfig(pair_weight=0.0))
        # With the pair term off, equal fragment coverage leaves retrieval
        # order in charge.
        self.assertEqual([asin for asin, _ in ranked], ["a", "b"])

    def test_conflict_weight_zero_matches_previous_behaviour(self) -> None:
        index = _StubIndex({
            "a": {"text": "cotton shirt grey"},
            "b": {"text": "cotton shirt black"},
        })
        state = DialogState("s")
        state.observe(1, "I'm looking for shirts")
        state.observe(2, "For that, what matters is: cotton shirt; color: grey.")
        pool = [("a", 1.0), ("b", 0.5)]
        zeroed = RerankConfig(facet_conflict_weight=0.0)
        ranked = rerank(index, state, pool, zeroed)
        self.assertEqual([asin for asin, _ in ranked], ["a", "b"])


class RouterTests(unittest.TestCase):
    def test_cue_based_classification(self) -> None:
        self.assertEqual(classify("I'm looking for boots. A key requirement is: leather.").name, "buying")
        self.assertEqual(classify("I'm looking for boots, but I'm still exploring.").name, "browsing")

    def test_unknown_phrasing_defaults_to_browsing(self) -> None:
        # Misreading a vague customer as a buyer commits to constraints they never
        # stated; the reverse costs at most one extra question.
        self.assertEqual(classify("hello there").name, "browsing")

    def test_facet_density_triggers_buying(self) -> None:
        route = classify("Looking for a 100% cotton black shirt size XL under $30.")
        self.assertTrue(route.is_buying)
        self.assertIn("material", route.detected_facets)
        self.assertIn("color", route.detected_facets)
        self.assertIn("price", route.detected_facets)

    def test_browsing_hesitation_overrides_accidental_facets(self) -> None:
        route = classify("I'm looking for some casual black shoes, but I'm not sure, just exploring ideas.")
        self.assertTrue(route.is_browsing)
        self.assertGreater(route.browsing_score, 0)

    def test_scenario_hints(self) -> None:
        override_route = classify("Actually, ignore my earlier choice. I need running shoes.")
        self.assertEqual(override_route.scenario_hint, "intent_override")

        boundary_route = classify("I don't have a preference for material, use your judgment.")
        self.assertEqual(boundary_route.scenario_hint, "boundary")

    def test_dynamic_turn_intent_transition(self) -> None:
        # Turn 1: Browsing
        t1_route = classify("I'm looking for shoes, but still exploring.")
        self.assertTrue(t1_route.is_browsing)

        # Turn 2: Customer provides concrete constraints -> transitions to buying
        t2_route = detect_turn_intent(
            "For that, what matters is: full grain leather; waterproof.",
            turn=2,
            current_track="browsing",
            productive_turns=2,
        )
        self.assertTrue(t2_route.is_buying)


class _StubFacets:
    """Two colours, many brands - the cardinality trap in miniature."""

    def get(self, parent_asin: str) -> dict[str, str]:
        return {"color": "black" if int(parent_asin) % 2 else "white", "brand": f"brand-{parent_asin}"}


class _StubIndex:
    def __init__(self, products: dict[str, dict]) -> None:
        self.products = products


if __name__ == "__main__":
    unittest.main()
