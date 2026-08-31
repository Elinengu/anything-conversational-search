"""Guards for structured state, runtime transitions, and distilled retrieval.

These tests use tiny in-memory stubs. They pin the state-machine behavior without
loading the 50,000-product catalog or changing the organizer-owned evaluator.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.context_programming import DialogPhase
from src.retrieval import RetrievalConfig, retrieve
from src.state import DialogState, SessionPhase
from starter.agent import Agent, AgentConfig


class SlotLedgerTests(unittest.TestCase):
    def test_incremental_slots_accumulate_with_provenance(self) -> None:
        state = DialogState("s")
        state.observe(1, "I need black leather boots")
        state.observe(2, "For that, what matters is: waterproof ankle support.")

        self.assertEqual(state.active_slot_values()["color"], ["black"])
        self.assertEqual(state.active_slot_values()["material"], ["leather"])
        self.assertIn("waterproof ankle support", state.active_slot_values()["feature"])
        self.assertEqual(state.active_slots["material"][0].source_turn, 1)

    def test_override_rewrites_authoritative_slots_but_keeps_audit_history(self) -> None:
        state = DialogState("s")
        state.observe(1, "I need a black leather wallet")
        state.observe(2, "Actually, ignore that. I need grey canvas instead.")

        active = state.active_slot_values()
        self.assertEqual(active["color"], ["grey"])
        self.assertEqual(active["material"], ["canvas"])
        self.assertTrue(all(slot.status == "superseded" for slot in state.superseded_slots))
        self.assertIn("black", state.full_text())
        self.assertNotIn("black", state.authoritative_text())

    def test_decline_does_not_create_a_product_constraint(self) -> None:
        state = DialogState("s")
        state.observe(1, "I am looking for socks")
        state.record_ask("color")
        state.observe(2, "I don't have an additional preference for color.")
        self.assertEqual(state.active_slot_values(), {})
        self.assertIn("color", state.dead_attributes)

    def test_browse_gated_stall_discloses_nothing_and_is_not_a_decline(self) -> None:
        """The browse-gated customer's stall must not become a product constraint.

        tools/stress_harness.py's browse-gated customer answers a broad question
        with free-form text rather than the official simulator's template.
        constraint_spans() chunks it on " - " into two fragments, so without the
        STALL_CUES guard both were recorded as "feature" slots and the turn
        counted as productive - resetting the unproductive streak that is the
        sole input to DialogPhase.STAGNATING.
        """
        state = DialogState("s")
        state.observe(1, "I am looking for socks")
        state.record_ask("color")
        state.observe(
            2,
            "I'm still just browsing - ask me about one particular thing "
            "and I'll tell you.",
        )
        self.assertEqual(state.active_slot_values(), {})
        self.assertTrue(state.utterances[-1].declined)
        self.assertFalse(state.last_turn_productive)
        self.assertEqual(state.unproductive_streak, 1)
        # A stall is not "no preference for color" - the attribute stays live.
        self.assertNotIn("color", state.dead_attributes)

    def test_browsing_opening_is_never_treated_as_a_stall(self) -> None:
        """turn 1 is exempt from STALL_CUES, and that exemption is load-bearing.

        evaluator/local_evaluator.py opens every browsing session with
        "I'm looking for {category}, but I'm still exploring." - which matches
        STALL_CUES. Gating on turn > 1 keeps the single most important
        utterance for retrieval in the query view for every browsing session.
        """
        state = DialogState("s")
        state.observe(1, "I'm looking for socks, but I'm still exploring.")
        self.assertFalse(state.utterances[0].declined)
        self.assertIn("socks", state.full_text())


class ProgressSignalTests(unittest.TestCase):
    def test_recent_failure_streak_resets_after_new_information(self) -> None:
        state = DialogState("s")
        state.observe(1, "I am looking for boots")
        state.observe(2, "I don't have a preference for brand")
        state.observe(3, "Either is fine")
        self.assertEqual(state.unproductive_streak, 2)
        state.observe(4, "black leather")
        self.assertEqual(state.unproductive_streak, 0)
        self.assertEqual(state.max_unproductive_streak, 2)

    def test_flat_repeated_pool_is_over_general_and_stable(self) -> None:
        state = DialogState("s")
        candidates = [(str(i), 1.0) for i in range(300)]
        state.observe_pool(candidates)
        self.assertTrue(state.over_general)
        state.observe_pool(candidates)
        self.assertEqual(state.stable_pool_turns, 1)
        self.assertEqual(state.pool_overlap, 1.0)


class IntentAndPhaseTests(unittest.TestCase):
    def test_browsing_promotes_to_buying_and_stays_there(self) -> None:
        agent = Agent.__new__(Agent)
        state = DialogState("s")
        state.observe(1, "I'm looking for boots, but I'm still exploring")
        self.assertEqual(agent._route_for(state).name, "browsing")
        state.observe(2, "black leather")
        state.observe(3, "waterproof ankle support")
        self.assertEqual(agent._route_for(state).name, "buying")
        state.observe(4, "I'm not sure about anything else")
        self.assertEqual(agent._route_for(state).name, "buying")
        self.assertEqual([x.current for x in state.intent_history], ["browsing", "buying"])

    def test_plan_transition_is_written_to_state(self) -> None:
        state = DialogState("s")
        state.observe(1, "show me boots")
        plan = SimpleNamespace(
            phase=DialogPhase.STAGNATING,
            guidance_action="diversify_and_fallback_probe",
        )
        Agent._apply_plan_to_state(state, plan)
        self.assertEqual(state.phase, SessionPhase.STAGNATING)
        self.assertEqual(state.transition_history[-1].current, "stagnating")


class PlanApplicationTests(unittest.TestCase):
    def test_stagnation_plan_changes_question_policy_but_respects_declines(self) -> None:
        default_policy = object()
        targeted_policy = object()
        agent = Agent.__new__(Agent)
        agent.config = AgentConfig(policy=default_policy, use_adaptive_orchestration=True)
        agent._targeted_policy = targeted_policy
        plan = SimpleNamespace(phase=DialogPhase.STAGNATING)
        state = DialogState("s")
        # intent_track defaults to "browsing" (src/state.py), which now has its
        # own routing ahead of the STAGNATING check this test targets - pin the
        # track to isolate the mechanism under test.
        state.intent_track = "buying"

        self.assertIs(agent._policy_for_state(state, plan), targeted_policy)
        state.dead_attributes.add("other")
        self.assertIs(agent._policy_for_state(state, plan), default_policy)

    def test_browsing_track_uses_the_browsing_policy_until_a_decline(self) -> None:
        default_policy = object()
        browsing_policy = object()
        agent = Agent.__new__(Agent)
        agent.config = AgentConfig(policy=default_policy, use_adaptive_orchestration=True)
        agent._browsing_policy = browsing_policy
        state = DialogState("s")
        state.intent_track = "browsing"

        # No plan (adaptive orchestration effectively off for this call) still
        # routes browsing-track turns to the browsing policy - it is not gated
        # on the plan/phase machinery the STAGNATING branch below it is.
        self.assertIs(agent._policy_for_state(state, None), browsing_policy)

        state.dead_attributes.add("other")
        # A decline falls through to the ordered FixedPolicy fallback exactly
        # like the buying track does - see the STAGNATING test above, and the
        # boundary-scenario regression this guards against.
        self.assertIs(agent._policy_for_state(state, None), default_policy)

    def test_plan_controls_recommendation_cutoff_and_slate_size(self) -> None:
        agent = Agent.__new__(Agent)
        agent.config = AgentConfig(elimination_scan=False)
        state = DialogState("s")
        candidates = [(str(i), 1.0) for i in range(10)]

        hold = SimpleNamespace(recommendation_cutoff=True, recommended_slate_size=2)
        self.assertEqual(agent._shortlist(state, candidates, 1, 10, plan=hold), [])

        show = SimpleNamespace(recommendation_cutoff=False, recommended_slate_size=2)
        result = agent._shortlist(state, candidates, 3, 10, plan=show)
        self.assertEqual([item["parent_asin"] for item in result], ["0", "1"])


class StructuredRetrievalTests(unittest.TestCase):
    class Index:
        def __init__(self):
            self.queries: list[str] = []

        def search_terms(self, text, limit):
            self.queries.append(text)
            return [("A", 1.0)]

    def test_active_slots_form_a_real_retrieval_route(self) -> None:
        state = DialogState("s")
        state.observe(1, "I need black leather boots")
        index = self.Index()
        config = RetrievalConfig(
            use_anchor=False,
            use_terms=False,
            use_focused=False,
            use_structured=True,
        )
        self.assertEqual(retrieve(index, state, config)[0][0], "A")
        self.assertEqual(index.queries, [state.authoritative_text()])

        # Normal turns keep this route off; the stagnation plan can turn it on
        # for one pass without mutating the global retrieval configuration.
        hinted_index = self.Index()
        hinted_config = RetrievalConfig(
            use_anchor=False,
            use_terms=False,
            use_focused=False,
            use_structured=False,
        )
        retrieve(hinted_index, state, hinted_config, route_hint="structured")
        self.assertEqual(hinted_index.queries, [state.authoritative_text()])


class SnapshotTests(unittest.TestCase):
    def test_snapshot_exposes_slots_progress_intent_and_transitions(self) -> None:
        state = DialogState("s")
        state.observe(1, "black leather boots")
        state.transition_to(SessionPhase.NARROWING, "targeted_disambiguation")
        state.update_intent(SimpleNamespace(name="buying", confidence=0.9), "constraints")
        snap = state.snapshot()

        self.assertEqual(snap["phase"], "narrowing")
        self.assertEqual(snap["intent"]["track"], "buying")
        self.assertEqual(snap["active_slots"]["material"], ["leather"])
        self.assertEqual(snap["active_slot_ledger"]["material"][0]["source_turn"], 1)
        self.assertIn("leather", snap["structured_query"])
        self.assertIn("pool", snap)
        self.assertEqual(snap["transitions"][-1]["current"], "narrowing")


if __name__ == "__main__":
    unittest.main()
