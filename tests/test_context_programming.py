"""Unit tests for Dynamic Context Programming (Runtime Adaptation & Adaptive Orchestration)."""

import unittest
from src.context_programming import (
    AdaptiveOrchestrator,
    ContextDistiller,
    DialogPhase,
    LongTermProfileStore,
    UserProfile,
)
from src.state import DialogState
from starter.agent import Agent, AgentConfig


class TestContextProgramming(unittest.TestCase):
    def setUp(self):
        self.store = LongTermProfileStore()

    def test_short_term_context_distillation(self):
        state = DialogState(session_id="s1")
        user = self.store.get_or_create("u1")

        state.observe(1, "I'm looking for cotton pants under $50.")
        ctx = ContextDistiller.distill(state, user, intent_track="buying")

        self.assertEqual(ctx.session_id, "s1")
        self.assertEqual(ctx.turn, 1)
        self.assertEqual(ctx.intent_track, "buying")
        self.assertEqual(ctx.query_facets.get("material"), "cotton")
        self.assertFalse(ctx.is_override)

        # Turn 2: Discloses specific constraint
        state.observe(2, "Must be black with pockets.")
        ctx2 = ContextDistiller.distill(state, user, intent_track="buying")
        self.assertIn("must be black with pockets", ctx2.hard_spans)

        # Turn 3: Customer declines brand preference
        state.record_ask("brand")
        state.observe(3, "I don't have an additional preference for brand.")
        ctx3 = ContextDistiller.distill(state, user, intent_track="buying")
        self.assertIn("brand", ctx3.dead_slots)

        # Turn 4: Override reversal transition
        state.observe(4, "Actually, let's look for a silk hoodie instead.")
        ctx4 = ContextDistiller.distill(state, user, intent_track="buying")
        self.assertTrue(ctx4.is_override)
        self.assertEqual(ctx4.override_turn, 4)

    def test_long_term_user_profile_evolution(self):
        user_id = "user_shopper_42"
        seed_meta = {"preference_tags": ["comfort", "vintage"]}
        profile = self.store.get_or_create(user_id, seed_meta)

        self.assertEqual(profile.session_count, 0)
        self.assertEqual(profile.total_turns, 0)
        self.assertIn("vintage", profile.preferred_styles)

        # Session 1: Shopper searches for black leather boots
        state1 = DialogState(session_id="sess_01")
        state1.observe(1, "I need black leather boots for winter.")
        ContextDistiller.distill(state1, profile, intent_track="browsing")

        self.assertEqual(profile.preferred_materials["leather"], 1)
        self.assertEqual(profile.preferred_colors["black"], 1)
        self.assertEqual(profile.total_turns, 1)

        # Session 2: Same user returns for a leather belt under $30
        profile.session_count += 1
        state2 = DialogState(session_id="sess_02")
        state2.observe(1, "Looking for a brown leather belt under 30.")
        ContextDistiller.distill(state2, profile, intent_track="buying")

        self.assertEqual(profile.session_count, 1)
        self.assertEqual(profile.total_turns, 2)
        self.assertEqual(profile.preferred_materials["leather"], 2)
        self.assertEqual(profile.preferred_colors["brown"], 1)

        # Verify top affinities distilled across sessions
        affinities = profile.top_affinities()
        self.assertEqual(affinities["material"], "leather")

    def test_adaptive_orchestration_phase_alignment(self):
        state = DialogState(session_id="s1")
        user = self.store.get_or_create("u1")
        config = AgentConfig()

        # Turn 1: Exploring phase (high entropy, broad pool)
        state.observe(1, "Show me running shoes.")
        ctx = ContextDistiller.distill(state, user, intent_track="browsing")
        candidates = [("asin_1", 10.0), ("asin_2", 9.8), ("asin_3", 9.5)]
        plan = AdaptiveOrchestrator.align_strategy(ctx, user, candidates, config)

        self.assertEqual(plan.phase, DialogPhase.EXPLORING)
        self.assertEqual(plan.guidance_action, "proactive_clarification")
        # The cutoff tracks first_recommend_turn, which the sniper ramp moved to
        # 1: turn 1 now emits its single best guess rather than withholding.
        self.assertFalse(plan.recommendation_cutoff)
        self.assertEqual(plan.recommended_slate_size, 1)
        held = AdaptiveOrchestrator.align_strategy(
            ctx, user, candidates, AgentConfig(first_recommend_turn=3))
        self.assertTrue(held.recommendation_cutoff, "cutoff no longer tracks the emit turn")

        # Turn 2: Buying track with clear leader -> Converging phase (fast path)
        state_buy = DialogState(session_id="s2")
        state_buy.observe(1, "I need running shoes.")
        state_buy.observe(2, "Breathable mesh lightweight.")
        ctx_buy = ContextDistiller.distill(state_buy, user, intent_track="buying")
        candidates_conv = [("asin_1", 10.0), ("asin_2", 8.0)]  # 20% margin
        plan_conv = AdaptiveOrchestrator.align_strategy(ctx_buy, user, candidates_conv, config)

        self.assertEqual(plan_conv.phase, DialogPhase.CONVERGING)
        self.assertFalse(plan_conv.recommendation_cutoff)
        self.assertEqual(plan_conv.guidance_action, "fast_path_recommendation")

        # Turn 3: Override Reversal -> Override phase
        state_buy.observe(3, "Actually I want sandals instead.")
        ctx_override = ContextDistiller.distill(state_buy, user, intent_track="buying")
        plan_override = AdaptiveOrchestrator.align_strategy(ctx_override, user, candidates, config)

        self.assertEqual(plan_override.phase, DialogPhase.OVERRIDE_REVERSAL)
        self.assertEqual(plan_override.retrieval_route, "focused")
        self.assertEqual(plan_override.guidance_action, "override_reversal_recovery")

        # Recovery lasts for the reversal turn only. New evidence after the
        # rewrite returns the session to an ordinary narrowing/converging phase.
        state_buy.observe(4, "black waterproof upper")
        ctx_after = ContextDistiller.distill(state_buy, user, intent_track="buying")
        plan_after = AdaptiveOrchestrator.align_strategy(
            ctx_after, user, [("asin_1", 10.0), ("asin_2", 9.8)], config
        )
        self.assertEqual(plan_after.phase, DialogPhase.NARROWING)

    def test_agent_integration_e2e(self):
        agent = Agent("data/catalog.jsonl")
        user_meta = {"user_id": "cust_999", "preference_tags": ["leather"]}

        agent.reset("session_a", user_meta)
        res1 = agent.respond("session_a", "I am looking for a vintage leather jacket.", turn=1, top_k=10)
        self.assertIn("message", res1)
        self.assertIn("ask_attribute", res1)

        # Check that user history is recorded in agent.profile_store
        user = agent.profile_store.get_or_create("cust_999")
        self.assertEqual(user.preferred_materials["leather"], 1)

        # Same user starts second session
        agent.reset("session_b", user_meta)
        agent.respond("session_b", "Also need a leather wallet.", turn=1, top_k=10)
        self.assertEqual(user.session_count, 2)
        self.assertEqual(user.preferred_materials["leather"], 2)


if __name__ == "__main__":
    unittest.main()
