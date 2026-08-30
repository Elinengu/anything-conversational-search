#!/usr/bin/env python3
"""Evaluation & Demonstration Tool for Dynamic Context Programming.

Verifies and displays:
1. Short-Term Session Context Distillation (spans, dead slots, override transitions).
2. Long-Term User Profile Evolution (multi-turn and multi-session preference updates).
3. Adaptive Orchestration (dialogue phase detection and dynamic strategy alignment).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starter.agent import Agent, AgentConfig
from src.context_programming import (
    AdaptiveOrchestrator,
    ContextDistiller,
    DialogPhase,
    LongTermProfileStore,
)
from src.state import DialogState


def evaluate_short_term_distillation():
    print("=" * 70)
    print("1. EVALUATING SHORT-TERM SESSION CONTEXT DISTILLATION")
    print("=" * 70)

    store = LongTermProfileStore()
    user = store.get_or_create("demo_shopper")
    state = DialogState(session_id="sess_demo_01")

    turns = [
        (1, "I'm looking for high-waisted black cotton pants under 50.", "browsing"),
        (2, "Must have deep pockets and relaxed fit.", "browsing"),
        (3, "I don't have an additional preference for brand.", "browsing"),
        (4, "Actually, I want a silk evening dress instead.", "browsing"),
    ]

    for turn_num, message, track in turns:
        if turn_num == 3:
            state.record_ask("brand")
        state.observe(turn_num, message)

        ctx = ContextDistiller.distill(state, user, intent_track=track)
        print(f"\n--- Turn {turn_num} ---")
        print(f"User Message:       {message}")
        print(f"Extracted Spans:    {ctx.hard_spans}")
        print(f"Extracted Facets:   {ctx.query_facets}")
        print(f"Dead Slots:         {list(ctx.dead_slots)}")
        print(f"Override Active:    {ctx.is_override} (Turn: {ctx.override_turn})")
        print(f"Productive Turns:   {ctx.productive_turns}")

    assert "cotton" in ctx.query_facets.values()
    assert "brand" in ctx.dead_slots
    assert ctx.is_override is True
    assert ctx.override_turn == 4
    print("\n[PASS] Short-term session distillation verified.")


def evaluate_long_term_profile_evolution():
    print("\n" + "=" * 70)
    print("2. EVALUATING LONG-TERM USER PROFILE EVOLUTION ACROSS SESSIONS")
    print("=" * 70)

    agent = Agent("data/catalog.jsonl")
    user_id = "customer_alex_88"
    user_metadata = {
        "user_id": user_id,
        "preference_tags": ["comfort", "vintage"],
        "purchase_frequency": "frequent",
    }

    # Session 1
    print("\n[Session 1] User searches for leather outerwear...")
    agent.reset("session_alex_01", user_metadata)
    agent.respond("session_alex_01", "Looking for a brown leather motorcycle jacket.", turn=1, top_k=10)
    agent.respond("session_alex_01", "Must have heavy duty zippers.", turn=2, top_k=10)

    user_profile = agent.profile_store.get_or_create(user_id)
    print(f"Session Count:       {user_profile.session_count}")
    print(f"Total Turns:         {user_profile.total_turns}")
    print(f"Preferred Materials: {dict(user_profile.preferred_materials)}")
    print(f"Preferred Colors:    {dict(user_profile.preferred_colors)}")

    # Session 2
    print("\n[Session 2] Same user returns to search for leather boots...")
    agent.reset("session_alex_02", user_metadata)
    agent.respond("session_alex_02", "I need matching brown leather Chelsea boots.", turn=1, top_k=10)

    print(f"Session Count:       {user_profile.session_count}")
    print(f"Total Turns:         {user_profile.total_turns}")
    print(f"Preferred Materials: {dict(user_profile.preferred_materials)}")
    print(f"Preferred Colors:    {dict(user_profile.preferred_colors)}")
    print(f"Top Affinities:      {user_profile.top_affinities()}")

    assert user_profile.session_count == 2
    assert user_profile.preferred_materials["leather"] == 2
    assert user_profile.preferred_colors["brown"] == 2
    assert user_profile.top_affinities()["material"] == "leather"
    print("\n[PASS] Long-term user profile evolution verified across sessions.")


def evaluate_adaptive_orchestration():
    print("\n" + "=" * 70)
    print("3. EVALUATING ADAPTIVE ORCHESTRATION & STRATEGY ALIGNMENT")
    print("=" * 70)

    store = LongTermProfileStore()
    user = store.get_or_create("demo_orchestrator_user")
    config = AgentConfig()

    scenarios = [
        ("Turn 1 Browsing (Over-General)", [ (1, "I want shoes.") ], "browsing", [("a1", 10.0), ("a2", 9.9), ("a3", 9.8)], DialogPhase.EXPLORING),
        ("Turn 2 Buying (Fast Converging)", [ (1, "I need running shoes."), (2, "Waterproof running shoes size 10.") ], "buying", [("a1", 10.0), ("a2", 8.0)], DialogPhase.CONVERGING),
        ("Turn 3 Override (Intent Reversal)", [ (1, "I need shoes."), (2, "Waterproof."), (3, "Actually I want sandals instead.") ], "buying", [("a1", 10.0), ("a2", 9.0)], DialogPhase.OVERRIDE_REVERSAL),
    ]

    for title, turns_list, track, cands, expected_phase in scenarios:
        state = DialogState(session_id="orch_test")
        for t_idx, t_msg in turns_list:
            state.observe(t_idx, t_msg)
        ctx = ContextDistiller.distill(state, user, intent_track=track)
        plan = AdaptiveOrchestrator.align_strategy(ctx, user, cands, config)

        print(f"\n--- {title} ---")
        print(f"Identified Phase:       {plan.phase.value}")
        print(f"Pool Entropy:           {plan.pool_entropy:.3f}")
        print(f"Confidence Margin:      {plan.gating_margin:.2f}")
        print(f"Recommendation Cutoff:  {plan.recommendation_cutoff}")
        print(f"Retrieval Route:        {plan.retrieval_route}")
        print(f"Guidance Action:        {plan.guidance_action}")
        assert plan.phase == expected_phase

    print("\n[PASS] Adaptive Orchestration & Phase Alignment verified.")


def main():
    print("======================================================================")
    print("DYNAMIC CONTEXT PROGRAMMING: VERIFICATION & EVALUATION SUITE")
    print("======================================================================\n")

    evaluate_short_term_distillation()
    evaluate_long_term_profile_evolution()
    evaluate_adaptive_orchestration()

    print("\n" + "=" * 70)
    print("ALL CONTEXT PROGRAMMING EVALUATIONS COMPLETED SUCCESSFULLY (100% PASS)")
    print("=" * 70)


if __name__ == "__main__":
    main()
