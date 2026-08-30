"""Dynamic Context Programming: Runtime Adaptation & Adaptive Orchestration.

This module fulfills Innovation Pillar III of the Conversational Search Specification:
1. Runtime Adaptation: Continuous Personalized Context Distillation updating
   short-term session state (constraints, dead slots, override transitions) and
   long-term user profiles (accumulated style, material, brand, price affinities).
2. Adaptive Orchestration: Dynamic Context Programming that computes dialogue
   phase and candidate pool dispersion to achieve runtime workflow re-orchestration
   and strategy alignment.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.facets import extract_query_facets
from src.state import DialogState


class DialogPhase(str, Enum):
    """Runtime interaction phase identified by the orchestrator."""
    EXPLORING = "exploring"         # Early turns / broad pool: active clarification
    CONVERGING = "converging"       # High confidence leader candidate: fast-path emission
    OVERRIDE_REVERSAL = "override"  # User reversed intent: context purging & focused routing
    STAGNATING = "stagnating"       # Information plateau: diversity / fallback probing


@dataclass
class UserProfile:
    """Long-term distilled user profile that evolves across dialogue turns and sessions."""
    user_id: str
    preferred_categories: Counter[str] = field(default_factory=Counter)
    preferred_materials: Counter[str] = field(default_factory=Counter)
    preferred_colors: Counter[str] = field(default_factory=Counter)
    preferred_styles: Counter[str] = field(default_factory=Counter)
    preferred_brands: Counter[str] = field(default_factory=Counter)
    price_sensitivities: Counter[str] = field(default_factory=Counter)
    session_count: int = 0
    total_turns: int = 0
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def record_turn(self, query_facets: dict[str, str], spans: list[str]) -> None:
        """Update long-term preference distributions from an observed dialogue turn."""
        self.total_turns += 1
        if "material" in query_facets:
            self.preferred_materials[query_facets["material"]] += 1
        if "color" in query_facets:
            self.preferred_colors[query_facets["color"]] += 1
        if "style" in query_facets:
            self.preferred_styles[query_facets["style"]] += 1
        if "brand" in query_facets:
            self.preferred_brands[query_facets["brand"]] += 1
        if "budget" in query_facets:
            self.price_sensitivities[query_facets["budget"]] += 1

    def top_affinities(self) -> dict[str, str]:
        """Return the user's primary long-term affinity in each attribute dimension."""
        affinities = {}
        for attr, counter in [
            ("material", self.preferred_materials),
            ("color", self.preferred_colors),
            ("style", self.preferred_styles),
            ("brand", self.preferred_brands),
            ("budget", self.price_sensitivities),
        ]:
            if counter:
                top_val, _ = counter.most_common(1)[0]
                affinities[attr] = top_val
        return affinities


class LongTermProfileStore:
    """In-memory persistent store managing long-term user profiles across sessions."""

    def __init__(self) -> None:
        self._profiles: dict[str, UserProfile] = {}

    def get_or_create(self, user_id: str, seed_profile: dict[str, Any] | None = None) -> UserProfile:
        if user_id not in self._profiles:
            profile = UserProfile(user_id=user_id, raw_metadata=dict(seed_profile or {}))
            # Seed initial tags from profile metadata if available
            if seed_profile and "preference_tags" in seed_profile:
                for tag in seed_profile["preference_tags"]:
                    tag_str = str(tag).lower()
                    profile.preferred_styles[tag_str] += 1
            self._profiles[user_id] = profile
        return self._profiles[user_id]

    def clear(self) -> None:
        self._profiles.clear()


@dataclass
class DistilledShortTermContext:
    """Distilled state of the active session synthesized on each turn."""
    session_id: str
    turn: int
    intent_track: str
    hard_spans: list[str]
    intact_pairs: list[str]
    dead_slots: set[str]
    is_override: bool
    override_turn: int | None
    query_facets: dict[str, str]
    productive_turns: int


class ContextDistiller:
    """Extracts and synthesizes short-term session state and long-term user context."""

    @staticmethod
    def distill(
        state: DialogState,
        user_profile: UserProfile,
        intent_track: str,
    ) -> DistilledShortTermContext:
        """Perform personalized context distillation over accumulated history."""
        full_text = state.full_text()
        query_facets = extract_query_facets(full_text)
        spans = state.query_spans()
        pairs = state.query_pair_spans()

        # Update long-term profile with current turn's newly disclosed facts
        latest_text = state.utterances[-1].text if state.utterances else ""
        turn_facets = extract_query_facets(latest_text)
        user_profile.record_turn(turn_facets, spans)

        return DistilledShortTermContext(
            session_id=state.session_id,
            turn=state.turn_count,
            intent_track=intent_track,
            hard_spans=spans,
            intact_pairs=pairs,
            dead_slots=set(state.dead_attributes),
            is_override=(state.override_turn is not None),
            override_turn=state.override_turn,
            query_facets=query_facets,
            productive_turns=state.productive_turns,
        )


@dataclass
class OrchestrationPlan:
    """Execution strategy aligned dynamically by the orchestrator for the current turn."""
    phase: DialogPhase
    pool_entropy: float
    confidence_lead: float
    gating_margin: float
    retrieval_route: str
    recommendation_cutoff: bool
    recommended_slate_size: int
    guidance_action: str


class AdaptiveOrchestrator:
    """Dynamic Context Programming: runtime workflow re-orchestration & strategy alignment."""

    @staticmethod
    def compute_pool_entropy(candidates: list[tuple[str, float]], depth: int = 30) -> float:
        """Calculate normalized entropy over candidate retrieval scores."""
        top_candidates = candidates[:depth]
        scores = [max(s, 0.0) for _, s in top_candidates if s > 0.0]
        total = sum(scores)
        if not scores or total <= 0.0 or len(scores) < 2:
            return 0.0
        entropy = 0.0
        for score in scores:
            p = score / total
            if p > 0.0:
                entropy -= p * math.log2(p)
        return entropy / math.log2(len(scores))

    @staticmethod
    def align_strategy(
        context: DistilledShortTermContext,
        user_profile: UserProfile,
        candidates: list[tuple[str, float]],
        config: Any,
    ) -> OrchestrationPlan:
        """Dynamically select retrieval, reranking, and gating strategies."""
        turn = context.turn
        entropy = AdaptiveOrchestrator.compute_pool_entropy(candidates)

        # Measure leader candidate score margin
        lead = 0.0
        if len(candidates) >= 2 and candidates[0][1] > 0:
            lead = (candidates[0][1] - candidates[1][1]) / candidates[0][1]

        # Phase 1: Intent Override Reversal Recovery
        if context.is_override and context.override_turn is not None and turn >= context.override_turn:
            return OrchestrationPlan(
                phase=DialogPhase.OVERRIDE_REVERSAL,
                pool_entropy=entropy,
                confidence_lead=lead,
                gating_margin=0.08,
                retrieval_route="focused",
                recommendation_cutoff=(turn < config.first_recommend_turn and lead < 0.08),
                recommended_slate_size=config.list_size_ramp[min(max(0, turn - config.first_recommend_turn), len(config.list_size_ramp) - 1)],
                guidance_action="override_reversal_recovery",
            )

        # Phase 2: Converging / High-Confidence Fast Path (Buying or strong leader)
        margin_threshold = (
            config.buying_confidence_margin
            if context.intent_track == "buying"
            else config.confidence_margin
        )
        is_confident = (lead >= margin_threshold and turn >= config.earliest_recommend_turn)

        if is_confident:
            return OrchestrationPlan(
                phase=DialogPhase.CONVERGING,
                pool_entropy=entropy,
                confidence_lead=lead,
                gating_margin=margin_threshold,
                retrieval_route="terms",
                recommendation_cutoff=False,
                recommended_slate_size=config.list_size_ramp[min(max(0, turn - config.first_recommend_turn), len(config.list_size_ramp) - 1)],
                guidance_action="fast_path_recommendation",
            )

        # Phase 3: Stagnation / Dead-End Exploration
        if turn >= 4 and context.productive_turns <= 1:
            return OrchestrationPlan(
                phase=DialogPhase.STAGNATING,
                pool_entropy=entropy,
                confidence_lead=lead,
                gating_margin=config.confidence_margin,
                retrieval_route="terms",
                recommendation_cutoff=(turn < config.first_recommend_turn),
                recommended_slate_size=10,
                guidance_action="diversify_and_fallback_probe",
            )

        # Phase 4: Standard Information Seeking (Exploration)
        return OrchestrationPlan(
            phase=DialogPhase.EXPLORING,
            pool_entropy=entropy,
            confidence_lead=lead,
            gating_margin=config.confidence_margin,
            retrieval_route="terms",
            recommendation_cutoff=(turn < config.first_recommend_turn),
            recommended_slate_size=config.list_size_ramp[min(max(0, turn - config.first_recommend_turn), len(config.list_size_ramp) - 1)],
            guidance_action="proactive_clarification",
        )
