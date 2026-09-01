"""Unit tests for the pipeline stages that carry the scoring behaviour."""

from __future__ import annotations

import collections
import unittest

from src.facets import extract
from src.policy import ALLOWED_ATTRIBUTES, FixedPolicy, InfoGainPolicy
from src.rerank import RerankConfig, _llm_gate_open, rerank
from src.retrieval import RetrievalConfig, retrieve
from src.router import classify, detect_turn_intent, extract_opening_facets
from src.state import PRE_OVERRIDE_WEIGHT, DialogState
from src.text import constraint_spans, pair_spans, terms
from starter.agent import Agent, AgentConfig


class TextTests(unittest.TestCase):
    def test_spans_survive_punctuation_and_casing(self) -> None:
        spans = constraint_spans("For that, what matters is: Day / Date Indicator; Stainless Steel Band.")
        self.assertIn("day date indicator", spans)
        self.assertIn("stainless steel band", spans)

    def test_spans_are_token_joined_for_substring_matching(self) -> None:
        # "100% Leather" must normalise the same way the index normalises product
        # text, or the reranker's substring match silently never fires.
        self.assertIn("100 leather", constraint_spans("what matters is: 100% Leather."))

    def test_constraint_spans_official_template_unchanged(self) -> None:
        # Anchor: the bidirectional stopword-strip added for paraphrased
        # carrier sentences must be a no-op here - the colon already isolates
        # carrier framing from the value on the official template.
        spans = constraint_spans(
            "For that, what matters is: Day / Date Indicator; Stainless Steel Band."
        )
        self.assertEqual(spans, ["day date indicator", "stainless steel band"])

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

    def test_shipped_weights_are_pinned(self) -> None:
        """The defaults are measured quantities (rerank_signals.md §10); a
        silent change to any of them is a scoring change and must fail here."""
        config = RerankConfig()
        self.assertEqual(
            (config.span_weight, config.pair_weight, config.retrieval_weight,
             config.popularity_weight, config.facet_weight, config.category_weight,
             config.tail_weight, config.facet_conflict_weight),
            (1.0, 0.8, 1.0, 0.4, 0.3, 0.4, 0.8, 0.4),
        )

    def test_popularity_breaks_lexical_ties_toward_reviewed_products(self) -> None:
        """The tie-break regime: identical span evidence, the reviewed product
        wins under the shipped weight and cannot win at the old 0.02 against a
        retrieval-score lead."""
        popular = {"text": "cotton shirt grey crew neck",
                   "average_rating": 4.6, "rating_number": 900}
        thin = {"text": "cotton shirt grey crew neck",
                "average_rating": 0.0, "rating_number": 0}
        index = _StubIndex({"pop": popular, "thin": thin})
        state = DialogState("s")
        state.observe(1, "I'm looking for shirts")
        state.observe(2, "For that, what matters is: cotton shirt grey.")
        pool = [("thin", 1.0), ("pop", 0.9)]  # retrieval prefers the thin listing
        ranked = rerank(index, state, pool, RerankConfig())
        self.assertEqual(ranked[0][0], "pop")
        old = rerank(index, state, pool, RerankConfig(popularity_weight=0.02))
        self.assertEqual(old[0][0], "thin")

    def test_track_kwarg_defaults_to_todays_behaviour(self) -> None:
        # A bare call and track=None must be byte-identical, and hard_filter is
        # inert on any track but "buying".
        index = _StubIndex({
            "wrong": {"text": "cotton shirt classic fit black only"},
            "right": {"text": "cotton shirt classic fit heather grey"},
        })
        state = DialogState("s")
        state.observe(1, "I'm looking for shirts")
        state.observe(2, "For that, what matters is: cotton shirt; color: grey.")
        pool = [("wrong", 1.0), ("right", 0.9)]
        base = rerank(index, state, list(pool), RerankConfig(hard_filter=True))
        self.assertEqual(base, rerank(index, state, list(pool), RerankConfig(hard_filter=True), track=None))
        self.assertEqual(base, rerank(index, state, list(pool), RerankConfig(hard_filter=True), track="browsing"))

    def test_hard_filter_banishes_a_contradicting_candidate_on_the_buying_track(self) -> None:
        index = _StubIndex({
            "wrong": {"text": "cotton shirt classic fit black only"},
            "right": {"text": "cotton shirt classic fit heather grey"},
        })
        state = DialogState("s")
        state.observe(1, "I'm looking for shirts")
        state.observe(2, "For that, what matters is: cotton shirt; color: grey.")
        # Retrieval strongly prefers the black-only impostor.
        ranked = rerank(index, state, [("wrong", 9.0), ("right", 0.1)],
                        RerankConfig(hard_filter=True), track="buying")
        self.assertEqual(ranked[0][0], "right")
        self.assertEqual(ranked[-1][0], "wrong")

    def test_dense_weight_zero_is_a_noop(self) -> None:
        index = _StubIndex({"a": {"text": "cotton shirt grey"}, "b": {"text": "cotton shirt black"}})
        state = DialogState("s")
        state.observe(1, "I'm looking for shirts")
        state.observe(2, "For that, what matters is: cotton shirt.")
        pool = [("a", 1.0), ("b", 0.5)]
        base = rerank(index, state, list(pool), RerankConfig())
        self.assertEqual(base, rerank(index, state, list(pool), RerankConfig(dense_weight=0.0),
                                      embed=_StubEmbed(), qvec=[1.0]))

    def test_dense_term_reorders_toward_the_semantic_match(self) -> None:
        # identical lexical evidence; the stub embed says "b" is the closer meaning.
        index = _StubIndex({"a": {"text": "cotton shirt classic"}, "b": {"text": "cotton shirt classic"}})
        state = DialogState("s")
        state.observe(1, "I'm looking for shirts")
        state.observe(2, "For that, what matters is: cotton shirt.")
        embed = _StubEmbed({"a": 0.5, "b": 0.9})
        ranked = rerank(index, state, [("a", 1.0), ("b", 0.9)],
                        RerankConfig(dense_weight=2.0), embed=embed, qvec=[1.0])
        self.assertEqual(ranked[0][0], "b")

    # -- Step 3.2: dense_gate_over_general / dense_gate_exclude_browsing -------

    def _dense_setup(self):
        index = _StubIndex({"a": {"text": "cotton shirt classic"}, "b": {"text": "cotton shirt classic"}})
        state = DialogState("s")
        state.observe(1, "I'm looking for shirts")
        state.observe(2, "For that, what matters is: cotton shirt.")
        embed = _StubEmbed({"a": 0.5, "b": 0.9})
        pool = [("a", 1.0), ("b", 0.9)]
        return index, state, embed, pool

    def test_over_general_gate_closed_is_byte_identical_to_dense_off(self) -> None:
        index, state, embed, pool = self._dense_setup()
        self.assertFalse(state.over_general)  # default - the gate under test starts closed
        base = rerank(index, state, list(pool), RerankConfig())
        gated = rerank(index, state, list(pool),
                       RerankConfig(dense_weight=2.0, dense_gate_over_general=True),
                       embed=embed, qvec=[1.0])
        self.assertEqual(base, gated)

    def test_over_general_gate_open_reorders(self) -> None:
        index, state, embed, pool = self._dense_setup()
        state.over_general = True
        ranked = rerank(index, state, list(pool),
                        RerankConfig(dense_weight=2.0, dense_gate_over_general=True),
                        embed=embed, qvec=[1.0])
        self.assertEqual(ranked[0][0], "b")

    def test_exclude_browsing_gate_withholds_on_the_browsing_track(self) -> None:
        index, state, embed, pool = self._dense_setup()
        base = rerank(index, state, list(pool), RerankConfig())
        gated = rerank(index, state, list(pool),
                       RerankConfig(dense_weight=2.0, dense_gate_exclude_browsing=True),
                       track="browsing", embed=embed, qvec=[1.0])
        self.assertEqual(base, gated)

    def test_exclude_browsing_gate_falls_back_to_state_intent_track(self) -> None:
        """No `track` kwarg passed - the gate must still read state.intent_track."""
        index, state, embed, pool = self._dense_setup()
        state.intent_track = "browsing"
        base = rerank(index, state, list(pool), RerankConfig())
        gated = rerank(index, state, list(pool),
                       RerankConfig(dense_weight=2.0, dense_gate_exclude_browsing=True),
                       embed=embed, qvec=[1.0])
        self.assertEqual(base, gated)

    def test_exclude_browsing_gate_allows_buying(self) -> None:
        index, state, embed, pool = self._dense_setup()
        ranked = rerank(index, state, list(pool),
                        RerankConfig(dense_weight=2.0, dense_gate_exclude_browsing=True),
                        track="buying", embed=embed, qvec=[1.0])
        self.assertEqual(ranked[0][0], "b")

    # -- Step 3.3: dense_query="slots" ------------------------------------------

    def test_dense_query_slots_encodes_authoritative_text(self) -> None:
        class _RecordingEmbed(_StubEmbed):
            def __init__(self) -> None:
                super().__init__()
                self.queries: list[str] = []

            def encode_query(self, text: str):
                self.queries.append(text)
                return [1.0]

        index, state, _unused_embed, pool = self._dense_setup()
        embed = _RecordingEmbed()
        # qvec is the full_text() vector the agent would have cached - "slots"
        # must ignore it and encode authoritative_text() fresh instead.
        rerank(index, state, list(pool), RerankConfig(dense_weight=1.0, dense_query="slots"),
              embed=embed, qvec=[0.0, 0.0])
        self.assertEqual(embed.queries, [state.authoritative_text()])

    def test_hard_filter_keeps_the_slate_when_every_candidate_contradicts(self) -> None:
        index = _StubIndex({
            "a": {"text": "cotton shirt black only"},
            "b": {"text": "cotton shirt navy only"},
        })
        state = DialogState("s")
        state.observe(1, "I'm looking for shirts")
        state.observe(2, "For that, what matters is: cotton shirt; color: grey.")
        pool = [("a", 1.0), ("b", 0.9)]
        ranked = rerank(index, state, list(pool), RerankConfig(hard_filter=True), track="buying")
        self.assertEqual(sorted(a for a, _ in ranked), ["a", "b"])  # no drop, no duplicate


class LLMRerankTests(unittest.TestCase):
    """Tier-2 opt-in layer (src/llm.py) - RerankConfig.llm_weight / llm_gate_margin.

    _StubLLM never touches the network; src/llm.py's own transport/parsing is
    covered by tests/test_llm.py.
    """

    def _setup(self, leader_margin: float = 0.0):
        index = _StubIndex({
            "a": {"text": "cotton shirt classic"},
            "b": {"text": "cotton shirt classic"},
        })
        state = DialogState("s")
        state.observe(1, "I'm looking for shirts")
        state.observe(2, "For that, what matters is: cotton shirt.")
        state.leader_margin = leader_margin
        pool = [("a", 1.0), ("b", 0.9)]
        return index, state, pool

    def test_llm_weight_zero_is_a_noop(self) -> None:
        index, state, pool = self._setup()
        llm = _StubLLM(order=["b", "a"])
        base = rerank(index, state, list(pool), RerankConfig())
        gated = rerank(index, state, list(pool), RerankConfig(llm_weight=0.0), llm=llm)
        self.assertEqual(base, gated)
        self.assertEqual(llm.calls, [])  # never even called

    def test_llm_reorders_the_head_when_weight_positive(self) -> None:
        index, state, pool = self._setup()
        llm = _StubLLM(order=["b", "a"])
        ranked = rerank(index, state, list(pool), RerankConfig(llm_weight=5.0), llm=llm)
        self.assertEqual(ranked[0][0], "b")

    def test_unavailable_llm_is_a_noop(self) -> None:
        index, state, pool = self._setup()
        llm = _StubLLM(order=["b", "a"], available=False)
        base = rerank(index, state, list(pool), RerankConfig())
        gated = rerank(index, state, list(pool), RerankConfig(llm_weight=5.0), llm=llm)
        self.assertEqual(base, gated)
        self.assertEqual(llm.calls, [])

    def test_no_llm_passed_is_a_noop(self) -> None:
        index, state, pool = self._setup()
        base = rerank(index, state, list(pool), RerankConfig())
        gated = rerank(index, state, list(pool), RerankConfig(llm_weight=5.0))
        self.assertEqual(base, gated)

    def test_llm_returning_none_falls_back_to_lexical_order(self) -> None:
        """Any failure inside LLMReranker.rank() surfaces as None - see
        src/llm.py. This must degrade exactly like llm_weight=0.0."""
        index, state, pool = self._setup()
        llm = _StubLLM(order=None)
        base = rerank(index, state, list(pool), RerankConfig())
        gated = rerank(index, state, list(pool), RerankConfig(llm_weight=5.0), llm=llm)
        self.assertEqual(base, gated)
        self.assertEqual(len(llm.calls), 1)  # it *was* called - just had no opinion

    def test_gate_blocks_a_confident_pool(self) -> None:
        index, state, pool = self._setup(leader_margin=0.5)  # confident leader
        llm = _StubLLM(order=["b", "a"])
        gated = rerank(index, state, list(pool),
                       RerankConfig(llm_weight=5.0, llm_gate_margin=0.05), llm=llm)
        self.assertEqual(gated[0][0], "a")  # lexical order stands
        self.assertEqual(llm.calls, [])

    def test_gate_opens_on_an_ambiguous_pool(self) -> None:
        index, state, pool = self._setup(leader_margin=0.01)  # near-tied leader
        llm = _StubLLM(order=["b", "a"])
        gated = rerank(index, state, list(pool),
                       RerankConfig(llm_weight=5.0, llm_gate_margin=0.05), llm=llm)
        self.assertEqual(gated[0][0], "b")
        self.assertEqual(len(llm.calls), 1)

    def test_gate_margin_zero_disables_the_gate(self) -> None:
        index, state, pool = self._setup(leader_margin=0.9)  # would otherwise block
        llm = _StubLLM(order=["b", "a"])
        gated = rerank(index, state, list(pool),
                       RerankConfig(llm_weight=5.0, llm_gate_margin=0.0), llm=llm)
        self.assertEqual(gated[0][0], "b")

    def test_llm_depth_bounds_the_reorderable_window(self) -> None:
        # "c" sits outside a depth-2 window and must not be promotable to the
        # front even though the model puts it first.
        index = _StubIndex({
            "a": {"text": "cotton shirt classic"},
            "b": {"text": "cotton shirt classic"},
            "c": {"text": "cotton shirt classic"},
        })
        state = DialogState("s")
        state.observe(1, "I'm looking for shirts")
        state.observe(2, "For that, what matters is: cotton shirt.")
        pool = [("a", 1.0), ("b", 0.9), ("c", 0.1)]
        llm = _StubLLM(order=["c", "b", "a"])
        ranked = rerank(index, state, list(pool),
                        RerankConfig(llm_weight=5.0, llm_depth=2), llm=llm)
        self.assertEqual([asin for asin, _ in ranked], ["b", "a", "c"])


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




class _StubTermsIndex:
    """Fixed FTS5 ranking, ignores the query - just enough for retrieve()."""

    def __init__(self, ranked: list[tuple[str, float]]) -> None:
        self._ranked = ranked
        self.products = {asin: {} for asin, _ in ranked}

    def match_pool(self, category_text, limit=1500):
        """No coarse-category pools in this stub - the route sees no opinion."""
        return []

    def search_terms(self, text: str, limit: int) -> list[tuple[str, float]]:
        return self._ranked[:limit]


class DenseRouteTests(unittest.TestCase):
    """S5 dense retrieval route (RetrievalConfig.use_dense)."""

    @staticmethod
    def _state() -> DialogState:
        state = DialogState("s")
        state.observe(1, "a lightweight waterproof jacket")
        return state

    def test_use_dense_false_is_byte_identical_to_the_no_kwarg_call(self) -> None:
        index = _StubTermsIndex([("a", 3.0), ("b", 2.0), ("c", 1.0)])
        state = self._state()
        base = retrieve(index, state, RetrievalConfig())
        self.assertEqual(retrieve(index, state, RetrievalConfig(), embed=None, qvec=None), base)
        # A usable embed present but use_dense off -> still the exact lexical pool.
        embed = _StubEmbed({"z": 0.99, "a": 0.2})
        self.assertEqual(
            retrieve(index, state, RetrievalConfig(use_dense=False), embed=embed, qvec=[1.0]),
            base,
        )

    def test_use_dense_changes_the_fused_pool(self) -> None:
        index = _StubTermsIndex([("a", 3.0), ("b", 2.0), ("c", 1.0)])
        state = self._state()
        base = retrieve(index, state, RetrievalConfig())
        embed = _StubEmbed({"z": 0.99, "a": 0.2})  # "z" is dense-only
        fused = retrieve(index, state, RetrievalConfig(use_dense=True), embed=embed, qvec=[1.0])
        self.assertNotEqual(fused, base)
        self.assertIn("z", [asin for asin, _ in fused])

    def test_dense_path_swallows_embed_failure(self) -> None:
        class _Boom:
            available = True

            def search(self, qvec, limit):
                raise RuntimeError("no onnx here")

        index = _StubTermsIndex([("a", 3.0), ("b", 2.0)])
        state = self._state()
        base = retrieve(index, state, RetrievalConfig())
        self.assertEqual(
            retrieve(index, state, RetrievalConfig(use_dense=True), embed=_Boom(), qvec=[1.0]),
            base,
        )

    # -- gating: RetrievalConfig.dense_gate_* (mirrors RerankConfig's, reuses
    #    the same _dense_gate_open - see docs/team/branch_state_encoder_eval_changes.md §3d) --

    def test_over_general_gate_closed_is_byte_identical_to_dense_off(self) -> None:
        index = _StubTermsIndex([("a", 3.0), ("b", 2.0), ("c", 1.0)])
        state = self._state()
        self.assertFalse(state.over_general)
        base = retrieve(index, state, RetrievalConfig())
        embed = _StubEmbed({"z": 0.99, "a": 0.2})
        gated = retrieve(index, state, RetrievalConfig(use_dense=True, dense_gate_over_general=True),
                         embed=embed, qvec=[1.0])
        self.assertEqual(gated, base)

    def test_over_general_gate_open_changes_the_pool(self) -> None:
        index = _StubTermsIndex([("a", 3.0), ("b", 2.0), ("c", 1.0)])
        state = self._state()
        state.over_general = True
        embed = _StubEmbed({"z": 0.99, "a": 0.2})
        fused = retrieve(index, state, RetrievalConfig(use_dense=True, dense_gate_over_general=True),
                         embed=embed, qvec=[1.0])
        self.assertIn("z", [asin for asin, _ in fused])

    def test_exclude_browsing_gate_withholds_on_the_browsing_track(self) -> None:
        index = _StubTermsIndex([("a", 3.0), ("b", 2.0), ("c", 1.0)])
        state = self._state()
        base = retrieve(index, state, RetrievalConfig())
        embed = _StubEmbed({"z": 0.99, "a": 0.2})
        gated = retrieve(index, state, RetrievalConfig(use_dense=True, dense_gate_exclude_browsing=True),
                         track="browsing", embed=embed, qvec=[1.0])
        self.assertEqual(gated, base)

    def test_exclude_browsing_gate_allows_buying(self) -> None:
        index = _StubTermsIndex([("a", 3.0), ("b", 2.0), ("c", 1.0)])
        state = self._state()
        embed = _StubEmbed({"z": 0.99, "a": 0.2})
        fused = retrieve(index, state, RetrievalConfig(use_dense=True, dense_gate_exclude_browsing=True),
                         track="buying", embed=embed, qvec=[1.0])
        self.assertIn("z", [asin for asin, _ in fused])


class _StubShortlistState:
    """The only state ``Agent._shortlist`` reads once the first turn has passed."""

    def __init__(self, session_id: str = "session-1") -> None:
        self.session_id = session_id
        self.override_turn = None


class ShortlistRampTests(unittest.TestCase):
    """``list_size_ramp`` is indexed by turn, clamped to its last entry.

    A narrow first slate defers commitment: the evaluator ends a session the
    moment the target is shown and freezes MRR at that position, so revealing
    ten candidates on turn 3 banks whatever rank the target holds *then*.
    Showing fewer costs a turn and buys the rank the next disclosed constraint
    earns. The elimination scan is what makes it free of coverage risk - the
    candidates held back are still reached on the following turn.
    """

    @staticmethod
    def _agent(ramp: tuple[int, ...]) -> Agent:
        """An Agent without its 50,000-row index - _shortlist never touches it."""
        agent = Agent.__new__(Agent)
        # first_recommend_turn is pinned so the probed turns (3, 4, 5, ...) line
        # up with ramp indices 0, 1, 2. These cases are about ramp *indexing*,
        # not about the shipped emit turn - see ShipedSniperRampTests for that.
        agent.config = AgentConfig(list_size_ramp=ramp, first_recommend_turn=3)
        agent._shown = {}
        agent._shown_override = {}
        agent._disclosed_count = {}
        return agent

    @staticmethod
    def _candidates(count: int = 200) -> list[tuple[str, float]]:
        return [(f"A{index:03d}", float(count - index)) for index in range(count)]

    def _sizes(self, ramp: tuple[int, ...]) -> list[int]:
        agent = self._agent(ramp)
        state = _StubShortlistState()
        candidates = self._candidates()
        return [len(agent._shortlist(state, candidates, turn, 10)) for turn in (3, 4, 5, 6)]

    def test_flat_ramp_shows_ten_every_turn(self) -> None:
        self.assertEqual(self._sizes((10,)), [10, 10, 10, 10])

    def test_narrow_first_slate_then_widens(self) -> None:
        self.assertEqual(self._sizes((4, 10)), [4, 10, 10, 10])

    def test_last_entry_applies_to_every_later_turn(self) -> None:
        self.assertEqual(self._sizes((5, 5, 10)), [5, 5, 10, 10])

    def test_ramp_never_exceeds_the_requested_top_k(self) -> None:
        agent = self._agent((10,))
        shortlist = agent._shortlist(_StubShortlistState(), self._candidates(), 3, 5)
        self.assertEqual(len(shortlist), 5)

    def test_narrow_slate_defers_rather_than_drops_candidates(self) -> None:
        """Nothing held back on turn 3 is lost - the scan reaches it on turn 4."""
        agent = self._agent((4, 10))
        state = _StubShortlistState()
        candidates = self._candidates()
        first = [item["parent_asin"] for item in agent._shortlist(state, candidates, 3, 10)]
        second = [item["parent_asin"] for item in agent._shortlist(state, candidates, 4, 10)]
        self.assertEqual(first, [f"A{index:03d}" for index in range(4)])
        self.assertEqual(second, [f"A{index:03d}" for index in range(4, 14)])
        self.assertFalse(set(first) & set(second))


class ShippedSniperRampTests(unittest.TestCase):
    """The shipped defaults emit one candidate per turn, widening at turn 5.

    A one-item slate is what converts an eventual hit into rank 1: the
    evaluator ends the session on the first slate containing the target and
    scores its position within that slate alone.
    """

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent.config = AgentConfig()
        agent._shown = {}
        agent._shown_override = {}
        agent._disclosed_count = {}
        return agent

    def test_singles_until_turn_five_then_widens(self) -> None:
        agent = self._agent()
        state = _StubShortlistState()
        candidates = [(f"A{index:03d}", float(200 - index)) for index in range(200)]
        sizes = [len(agent._shortlist(state, candidates, turn, 10)) for turn in range(1, 8)]
        self.assertEqual(sizes, [1, 1, 1, 1, 10, 10, 10])

    def test_singles_walk_distinct_candidates(self) -> None:
        """Four singles plus the wide turn reach 14 distinct products, not 10."""
        agent = self._agent()
        state = _StubShortlistState()
        candidates = [(f"A{index:03d}", float(200 - index)) for index in range(200)]
        seen: list[str] = []
        for turn in range(1, 6):
            seen += [item["parent_asin"] for item in agent._shortlist(state, candidates, turn, 10)]
        self.assertEqual(len(seen), len(set(seen)), "the scan re-showed a candidate")
        self.assertEqual(seen, [f"A{index:03d}" for index in range(14)])


class _StubFacets:
    """Two colours, many brands - the cardinality trap in miniature."""

    def get(self, parent_asin: str) -> dict[str, str]:
        return {"color": "black" if int(parent_asin) % 2 else "white", "brand": f"brand-{parent_asin}"}


class _StubIndex:
    def __init__(self, products: dict[str, dict]) -> None:
        self.products = products


class _StubLLM:
    """Stands in for src.llm.LLMReranker - a fixed ranking opinion, no network."""

    def __init__(self, order: list[str] | None = None, available: bool = True) -> None:
        self.order = order
        self.available = available
        self.calls: list[tuple[str, list[dict]]] = []

    def rank(self, query_text: str, candidates: list[dict]) -> list[str] | None:
        self.calls.append((query_text, list(candidates)))
        return self.order


class _StubEmbed:
    """Stands in for src.embed.EmbeddingIndex - fixed cosine per asin."""

    available = True

    def __init__(self, sims: dict[str, float] | None = None) -> None:
        self._sims = sims or {}

    def encode_query(self, text: str):
        return [1.0]

    def similarities(self, qvec, asins):
        return {a: self._sims.get(a, 0.0) for a in asins}

    def search(self, qvec, limit):
        ranked = sorted(self._sims.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:limit]


class _SplitFacets:
    """Half the pool leather, half canvas - a clean material split, nothing else."""

    def get(self, parent_asin: str) -> dict[str, str]:
        return {"material": "leather" if int(parent_asin) % 2 else "canvas"}


class PhrasingTests(unittest.TestCase):
    """src/phrasing.py - natural text must agree with ``ask_attribute``."""

    def _state(self, turn: int, productive: int = 2) -> DialogState:
        state = DialogState("s")
        state.observe(1, "I'm looking for a belt")
        for t in range(2, turn + 1):
            state.observe(t, "For that, what matters is: full grain leather; buckle closure.")
        state.productive_turns = productive
        return state

    def test_off_reproduces_the_fixed_question_byte_for_byte(self) -> None:
        from src.phrasing import clarify
        from src.policy import QUESTION_TEXT
        from src.router import BUYING

        cfg = AgentConfig(natural_questions=False)
        state = self._state(3)
        got = clarify("other", state, [], _SplitFacets(), BUYING, cfg)
        expect = BUYING.tone + QUESTION_TEXT["other"][0].lower() + QUESTION_TEXT["other"][1:]
        self.assertEqual(got, expect)
        # route None -> no tone prefix, exactly the old behaviour
        self.assertEqual(
            clarify("size", state, [], _SplitFacets(), None, cfg), QUESTION_TEXT["size"]
        )

    def test_grounded_question_names_the_pool_split(self) -> None:
        from src.phrasing import clarify

        cfg = AgentConfig(natural_questions=True)
        state = self._state(3)
        pool = [(str(i), 1.0) for i in range(40)]
        msg = clarify("other", state, pool, _SplitFacets(), None, cfg)
        self.assertIsInstance(msg, str)
        self.assertIn("leather", msg)
        self.assertIn("canvas", msg)
        self.assertIn("another detail", msg.lower())

    def test_specific_grounded_question_only_voices_that_attribute(self) -> None:
        from src.phrasing import clarify

        cfg = AgentConfig(natural_questions=True)
        state = self._state(3)
        state.record_ask("material")  # mirrors Agent._respond ordering
        pool = [(str(i), 1.0) for i in range(40)]
        msg = clarify("material", state, pool, _SplitFacets(), None, cfg)
        self.assertIn("leather", msg)
        self.assertIn("canvas", msg)
        self.assertIn("material", msg)
        self.assertNotIn("another detail", msg.lower())

    def test_specific_attribute_does_not_voice_an_unrelated_pool_split(self) -> None:
        from src.phrasing import SPECIFIC_BANK, clarify

        cfg = AgentConfig(natural_questions=True)
        state = self._state(3)
        state.record_ask("size")
        pool = [(str(i), 1.0) for i in range(40)]
        msg = clarify("size", state, pool, _SplitFacets(), None, cfg)
        self.assertIn(msg, SPECIFIC_BANK["size"])
        self.assertNotIn("leather", msg)
        self.assertNotIn("canvas", msg)

    def test_phrasing_failure_preserves_the_requested_attribute(self) -> None:
        from src.phrasing import SPECIFIC_BANK, clarify

        class BrokenFacets:
            def get(self, _parent_asin):
                raise RuntimeError("broken facet store")

        cfg = AgentConfig(natural_questions=True)
        state = self._state(3)
        state.record_ask("size")
        pool = [(str(i), 1.0) for i in range(40)]
        msg = clarify("size", state, pool, BrokenFacets(), None, cfg)
        self.assertIn(msg, SPECIFIC_BANK["size"])

    def test_never_raises_on_degenerate_input(self) -> None:
        from src.phrasing import clarify

        cfg = AgentConfig(natural_questions=True)
        cases = [
            (DialogState("s"), []),
            (self._state(3), []),
            (self._state(1, productive=0), [("0", 1.0)]),
        ]
        for state, pool in cases:
            msg = clarify("other", state, pool, _SplitFacets(), None, cfg)
            self.assertIsInstance(msg, str)
            self.assertTrue(msg)

    def test_broad_on_the_opening_turn(self) -> None:
        from src.phrasing import BROAD_BANK, clarify

        cfg = AgentConfig(natural_questions=True)
        state = DialogState("s")
        state.observe(1, "I'm looking for a belt")  # turn 1 -> pool is the catalog prior
        pool = [(str(i), 1.0) for i in range(40)]
        self.assertIn(clarify("other", state, pool, _SplitFacets(), None, cfg), BROAD_BANK)

    def test_single_word_facet_is_now_a_productive_turn(self) -> None:
        # The structured slot ledger recognizes useful single-word facets even
        # when the legacy multi-word span extractor yields nothing.
        from src.phrasing import clarify

        cfg = AgentConfig(natural_questions=True)
        state = DialogState("s")
        state.observe(1, "I'm looking for a belt")
        state.observe(2, "leather")
        self.assertEqual(state.productive_turns, 1)
        pool = [(str(i), 1.0) for i in range(40)]
        msg = clarify("other", state, pool, _SplitFacets(), None, cfg)
        self.assertIn("leather", msg)
        self.assertIn("canvas", msg)

    def test_broad_fallback_varies_across_turns(self) -> None:
        from src.phrasing import clarify

        cfg = AgentConfig(natural_questions=True)
        seen = set()
        for turn in range(1, 5):
            state = DialogState("s")
            for t in range(1, turn + 1):
                state.observe(t, "hi")  # never productive -> always the broad path
            seen.add(clarify("other", state, [], _SplitFacets(), None, cfg))
        self.assertGreaterEqual(len(seen), 2)

    def test_leadin_rotates_and_is_sometimes_absent(self) -> None:
        from src.phrasing import clarify
        from src.router import BROWSING

        cfg = AgentConfig(natural_questions=True)
        pool = [(str(i), 1.0) for i in range(40)]
        seen = set()
        bare = 0
        for turn in range(2, 11):
            state = DialogState("s")
            state.observe(1, "I'm after a wallet")
            for t in range(2, turn + 1):
                state.observe(t, "leather")
            msg = clarify("other", state, pool, _SplitFacets(), BROWSING, cfg)
            seen.add(msg)
            bare += int(msg[0].isupper() and not msg.startswith("To "))
        self.assertGreaterEqual(len(seen), 4)   # varied wording turn to turn
        self.assertGreaterEqual(bare, 1)        # some turns carry no prefix

    def test_override_turn_is_acknowledged(self) -> None:
        from src.phrasing import LEADIN_OVERRIDE, clarify
        from src.router import BROWSING

        cfg = AgentConfig(natural_questions=True)
        pool = [(str(i), 1.0) for i in range(40)]
        state = DialogState("s")
        state.observe(1, "still exploring")
        state.observe(2, "leather")
        state.observe(3, "actually, forget that - something else")  # OVERRIDE_CUES
        self.assertEqual(state.override_turn, 3)
        msg = clarify("other", state, pool, _SplitFacets(), BROWSING, cfg)
        self.assertTrue(any(msg.startswith(p) for p in LEADIN_OVERRIDE), msg)

    def test_leading_pronoun_is_not_lowercased(self) -> None:
        from src.phrasing import _apply

        self.assertEqual(_apply("So, ", "I'm seeing a split."), "So, I'm seeing a split.")
        self.assertEqual(_apply("So, ", "The options vary."), "So, the options vary.")


if __name__ == "__main__":
    unittest.main()


class TurnOneLLMReachabilityTests(unittest.TestCase):
    """The LLM must be able to run on turn 1, and only when asked to.

    Turn 1 is the one turn where S6 is otherwise inert: rerank() returns its
    input untouched when query_spans() is empty, and query_spans() skips the
    opening deliberately (it is the simulator's framing, not quoted product
    copy). It is also the turn sniper sizing stakes a whole slate on.
    """

    class _StubLLM:
        available = True

        def __init__(self) -> None:
            self.stats = collections.Counter()
            self.queries: list[str] = []

        def rank(self, query_text, candidates):
            self.queries.append(query_text)
            return [item["asin"] for item in candidates][::-1]

    def _state(self):
        state = DialogState(session_id="t1")
        state.observe(1, "I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.")
        return state

    def _index(self):
        return _StubIndex({
            "A1": {"parent_asin": "A1", "text": "alloy necklace", "categories": ["Jewelry", "Necklaces"]},
            "A2": {"parent_asin": "A2", "text": "silver necklace", "categories": ["Jewelry", "Necklaces"]},
        })

    def test_turn_one_has_no_spans(self) -> None:
        """The premise. If this ever changes, the rest of this class is moot."""
        self.assertEqual(self._state().query_spans(), [])

    def test_llm_does_not_run_on_turn_one_by_default(self) -> None:
        """The lexical reranker now runs on turn 1; the LLM still must not.

        ``rerank_without_spans`` ships on, so turn 1 is no longer a no-op - but
        that is the offline stage. Reaching the model needs ``llm_weight > 0``,
        which is not a default, so the shipped agent stays offline and free.
        """
        llm = self._StubLLM()
        rerank(self._index(), self._state(), [("A1", 1.0), ("A2", 0.9)],
               RerankConfig(), llm=llm)
        self.assertEqual(llm.queries, [], "the model was consulted without being asked for")

    def test_lexical_rerank_does_run_on_turn_one_by_default(self) -> None:
        """The shipped default rescores turn 1 - worth +0.0017 public, free."""
        untouched = rerank(self._index(), self._state(), [("A1", 1.0), ("A2", 0.9)],
                           RerankConfig(rerank_without_spans=False))
        rescored = rerank(self._index(), self._state(), [("A1", 1.0), ("A2", 0.9)],
                          RerankConfig())
        self.assertEqual(untouched, [("A1", 1.0), ("A2", 0.9)])
        self.assertNotEqual(rescored, untouched, "turn 1 was left unscored")

    def test_a_later_span_less_turn_is_still_left_alone(self) -> None:
        """Scoped to turn 1 on purpose: a declined "no preference" reply also
        produces no spans, and rescoring it only re-sorts stale evidence.
        Measured, unscoped vs turn-1-only: 0.9555 vs 0.9567."""
        state = self._state()
        state.observe(2, "I don't have a preference for material; please use your judgment.")
        candidates = [("A1", 1.0), ("A2", 0.9)]
        self.assertEqual(state.query_spans(), [], "premise: this turn has no spans")
        self.assertEqual(rerank(self._index(), state, candidates, RerankConfig()),
                         candidates)

    def test_enabling_the_llm_reaches_turn_one(self) -> None:
        llm = self._StubLLM()
        candidates = [("A1", 1.0), ("A2", 0.9)]
        out = rerank(self._index(), self._state(), candidates,
                     RerankConfig(llm_weight=1.0), llm=llm)
        self.assertEqual(len(llm.queries), 1, "the model was never consulted")
        self.assertEqual([asin for asin, _ in out], ["A2", "A1"],
                         "the model's order did not reach the result")

    def test_the_gate_is_open_on_turn_one(self) -> None:
        """leader_margin is 0.0 before any pool is observed, so an unobserved
        pool reads as maximally ambiguous. Asserted rather than left to luck."""
        self.assertEqual(self._state().leader_margin, 0.0)
        self.assertTrue(_llm_gate_open(self._state(), RerankConfig(llm_gate_margin=0.05)))

    def test_llm_query_opening_sends_the_category_too(self) -> None:
        llm = self._StubLLM()
        rerank(self._index(), self._state(), [("A1", 1.0), ("A2", 0.9)],
               RerankConfig(llm_weight=1.0, llm_query="opening"),
               llm=llm)
        self.assertIn("Necklaces", llm.queries[0],
                      "the opening query should carry the category, not just the slot value")
