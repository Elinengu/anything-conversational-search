"""Conversational shopping agent - the entry point the evaluator imports.

This file is deliberately thin. It owns the response contract
(``docs/agent_api_contract.json``) and wires together the pipeline stages that
live in ``src/``:

    S1 index      src/index.py       catalog -> FTS5 + trimmed product records
    S2 router     src/router.py      buying vs browsing track selection
    S3 state      src/state.py       turn accumulation, provenance, override
    S4 policy     src/policy.py      which attribute to ask for next
    S5 retrieval  src/retrieval.py   multi-route candidate generation
    S6 rerank     src/rerank.py      verbatim span coverage over the pool

The core path is fully offline: no network, no API key, standard library only.

Design note - why this beats the shipped BM25 baseline by ~7x: the simulated
customer only discloses constraints when the agent sets ``ask_attribute``, and the
baseline never set it and never remembered a previous turn. Asking a question and
accumulating the answers is worth far more than any retrieval tuning.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow ``python -m evaluator.local_evaluator`` to import src/ regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.index import load_index  # noqa: E402
from src.facets import FacetStore  # noqa: E402
from src.phrasing import clarify  # noqa: E402
from src.policy import ALLOWED_ATTRIBUTES, FixedPolicy, InfoGainPolicy  # noqa: E402
from src.rerank import RerankConfig, rerank  # noqa: E402
from src.retrieval import RetrievalConfig, retrieve  # noqa: E402
from src.router import BUYING, classify, detect_turn_intent  # noqa: E402
from src.context_programming import (  # noqa: E402
    AdaptiveOrchestrator,
    ContextDistiller,
    DialogPhase,
    LongTermProfileStore,
)
from src.llm import LLMClient, LLMConfig  # noqa: E402
from src.state import DialogState, SessionPhase  # noqa: E402


@dataclass
class AgentConfig:
    """Everything the sweep harness is allowed to vary."""

    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    #: Left unset so the Agent can build the default policy against its own index.
    policy: object | None = None
    # Recommending before the customer has disclosed anything locks in a poor
    # reciprocal rank, because the session ends on the first hit at any position.
    first_recommend_turn: int = 3
    # Shortlist size per turn; the last entry applies to all later turns.
    # The first slate is deliberately narrow. Emitting a list ends the session
    # the moment the target appears and freezes MRR at that position, while a
    # wrong list costs only a turn - so revealing ten candidates on turn 3 banks
    # whatever rank the target holds *then*, and showing four defers to turn 4,
    # when the next disclosed constraint has re-ranked it higher. The
    # elimination scan makes the deferral free of coverage risk: the candidates
    # held back are the top of the survivor list next turn (see _shortlist).
    # Measured, one process, four sets (dev / holdout / generated / hard):
    #   (10,)    0.9233 / 0.9048 / 0.9181 / 0.7944   flat, the pre-ramp floor
    #   (3,10)   0.9254 / 0.9146 / 0.9212 / 0.7968
    #   (4,10)   0.9268 / 0.9096 / 0.9197 / 0.7981   <- ships
    #   (5,10)   0.9295 / 0.9100 / 0.9210 / 0.8001
    # First-slate sizes 3, 4 and 5 all beat the flat ramp on all four sets, so
    # this sits mid-plateau rather than at any split's argmax - (5,10) has the
    # better mean, but choosing it after the fact is how you buy noise.
    # Narrowing a *second* turn is worse, not more of the same good thing:
    # (5,5,10) scores 0.9272 / 0.9044 / 0.9187 / 0.7934, regressing holdout and
    # hard below the floor. Each session holds four constraints disclosed at up
    # to two per turn, so by turn 4-5 no further evidence is coming and holding
    # narrow only spends turns. Cost of the ramp: generated-set Hit@10 0.995 ->
    # 0.990, one session that runs out of turns. See docs/team/agent_changes.md.
    list_size_ramp: tuple[int, ...] = (4, 10)
    # Optional confidence gate: emit earlier than first_recommend_turn when the top
    # candidate clearly leads the pool. 0.0 disables it.
    # 0.15-0.50 all beat 0.0 on dev and holdout alike; the curve is flat, so this
    # sits mid-plateau rather than at either split's argmax.
    confidence_margin: float = 0.20
    # Track-aware Turn-2 gating: buying sessions start with higher constraint density
    # (1 hard requirement on Turn 1 + 2 constraints on Turn 2). Fast-pathing confident
    # buying candidates on Turn 2 reduces MTTC across all splits without losing MRR.
    buying_confidence_margin: float = 0.08
    earliest_recommend_turn: int = 2
    # Dual-track routing. When True (the default), src/router.py classifies the
    # opening as "buying" or "browsing" and that track now drives *behaviour*, not
    # only phrasing: which clarification policy runs (S4), which RerankConfig and
    # whether the buying hard-filter applies (S6), and when recommendations start
    # and how wide the first slate is (S7). The track is re-checked every turn via
    # detect_turn_intent, so a browsing session that discloses enough - and any
    # post-override turn - is promoted to buying (never demoted). When False, none
    # of that fires and the agent is the flat, single-track pipeline: identical
    # scored output to the pre-routing agent. False is kept as the measurement
    # baseline and the guaranteed-safe fallback (the respond() exception path is
    # flat regardless). See docs/team/dual_track_routing.md.
    use_router: bool = True
    #: When use_router, pick the policy by track (buying -> FixedPolicy,
    #: browsing -> InfoGainPolicy). False keeps ``policy`` on both tracks. This is
    #: the one track lever left ON by default: it is what the realism harness
    #: (tools/stress_harness.py, --customer browse-gated) exists to justify. The other three levers
    #: below default to values identical to the single-track pipeline - measured
    #: net-negative on the fully-cooperative public simulator (it over-rewards the
    #: broad "other" question and early recommendations), they are switches for
    #: harness experiments, not the public path. See docs/team/dual_track_routing.md.
    route_policies: bool = True
    #: Per-track rerank weights (S6). None on a track -> reuse ``rerank``.
    buying_rerank: RerankConfig | None = None
    browsing_rerank: RerankConfig | None = None
    #: Per-track recommendation timing (S7), used only when use_router. A buying
    #: turn of None falls back to the router's suggested_first_recommend_turn;
    #: default 3 == the single-track value (suggested=2 was measured -0.11 buying
    #: MRR on the public sim - recommending before turn 3 freezes a worse rank).
    buying_first_recommend_turn: int | None = 3
    browsing_first_recommend_turn: int = 3
    buying_list_size_ramp: tuple[int, ...] = (4, 10)
    browsing_list_size_ramp: tuple[int, ...] = (4, 10)
    # Consume the plan produced by AdaptiveOrchestrator. False disables plan
    # effects for focused A/B measurements while retaining the new state ledger.
    use_adaptive_orchestration: bool = True
    # Elimination scan: any product shown and not hit on is a confirmed
    # non-target (the session ends on a hit), so each turn we drop everything
    # already shown and return the top 10 of the re-ranked survivors. This walks
    # up to 100 distinct candidates over 10 turns, re-ranking with all current
    # information every turn, with no frozen-pool bookkeeping. False = the plain
    # "same top 10 every turn" behaviour.
    elimination_scan: bool = True
    # Hold every recommendation until disclosure has stalled (a turn that adds no
    # new real constraint). Off by default; the scan already tolerates early,
    # under-informed lists because a wrong list only costs a turn.
    hold_until_stalled: bool = False
    # Pool-aware clarification wording (src/phrasing.py). ``ask_attribute`` is
    # unchanged and the simulator never reads ``message``, so the score is
    # identical on vs off - measured, one process:
    #   natural_off  dev 0.9418  holdout 0.9136   (per-scenario identical)
    #   natural_on   dev 0.9418  holdout 0.9136
    # (tools/sweep.py rows ``natural_off`` / ``natural_on``). The value is
    # product realism for the demo / Innovation / Presentation criteria. It can
    # name a useful live-pool split while keeping an ``other`` ask explicitly
    # open-ended. Specific ``ask_attribute`` values only produce messages about
    # that same attribute. ``False`` restores the fixed question strings exactly.
    natural_questions: bool = True
    #: Candidates inspected when choosing which facet to voice (see phrasing).
    phrasing_depth: int = 40
    #: Optional LLM integration for transparent reasoning and conversational generation.
    llm: LLMConfig = field(default_factory=LLMConfig)


class Agent:
    """Multi-turn shopping agent conforming to the official interface."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        config: AgentConfig | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.index = load_index(catalog_path)
        self.facets = FacetStore(self.index.products)
        self.llm = LLMClient(self.config.llm)
        if self.config.policy is None:
            # FixedPolicy is the default because it wins on the held-out split
            # (0.8349 vs 0.8119). InfoGainPolicy is the more interesting design and
            # ties on dev; see README "Clarification policy" for the measurements.
            self.config.policy = FixedPolicy()
        # Dual-track policies, built once, consulted only when use_router.
        # Buying keeps the broad question - a decided customer recites everything
        # on "anything else?". Browsing asks the highest-information-gain
        # attribute *once broad questions stop paying off*: InfoGainPolicy already
        # prefers broad while it yields (src/policy.py select()), so on the
        # cooperative public simulator - where broad always yields - it tracks
        # FixedPolicy closely, and on a realistic browser who volunteers nothing
        # it pivots to targeted questions on the first turn. expected_broad_answers
        # is raised so the pivot waits out the public sim's full 4-constraint
        # disclosure.
        self._buying_policy = FixedPolicy()
        self._browsing_policy = InfoGainPolicy(self.facets, expected_broad_answers=4.0)
        # Used only after the live state detects over-generality or stagnation,
        # overlaid on top of whichever dual-track policy above is in effect (see
        # _policy_for_state). Broad questions remain the measured default while
        # they are productive.
        self._targeted_policy = InfoGainPolicy(self.facets, allow_broad=False)
        self._states: dict[str, DialogState] = {}
        # Long-term user profile store across sessions (Dynamic Context Programming)
        self.profile_store = LongTermProfileStore()
        self._session_users: dict[str, str] = {}
        # parent_asins already shown this session - excluded from later turns
        # (see AgentConfig.elimination_scan).
        self._shown: dict[str, set[str]] = {}
        # override turn last accounted for, per session (see _shortlist).
        self._shown_override: dict[str, int] = {}
        # last-seen real-disclosure count per session (AgentConfig.hold_until_stalled).
        self._disclosed_count: dict[str, int] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._states[session_id] = DialogState(
            session_id=session_id,
            profile=user_profile if isinstance(user_profile, dict) else {},
        )
        user_id = (user_profile.get("user_id") if isinstance(user_profile, dict) else None) or session_id
        self._session_users[session_id] = user_id
        user = self.profile_store.get_or_create(user_id, user_profile)
        user.session_count += 1

        self._shown.pop(session_id, None)
        self._shown_override.pop(session_id, None)
        self._disclosed_count.pop(session_id, None)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            return self._respond(session_id, user_message, turn, top_k)
        except Exception:
            # A raised exception is scored as a miss for the whole session, so the
            # turn always degrades to a valid empty response instead.
            return self._envelope("Could you tell me a little more about what you need?", "other", [])

    # ---- internals ------------------------------------------------------------

    def _respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._states.get(session_id)
        if state is None:
            # Defensive: the contract guarantees reset() first, but a missing
            # session must not cost the run.
            state = self._states[session_id] = DialogState(session_id=session_id)

        state.observe(turn, user_message)

        # The promotable track (buying once enough is disclosed, or after an
        # override) drives S6 weights / hard-filter, S7 timing, and the
        # adaptive-orchestration plan below. ``_route_for`` (superseding the old
        # ``_track()``/``_track_cache`` dict) records this stickily on ``state``
        # itself (``state.intent_history``), which is also what makes it
        # observable in ``tools/observe.py``. The clarification *policy*
        # instead keys off how the session *opened* and stays there:
        # InfoGainPolicy already self-adapts (broad while broad pays off,
        # targeted once it stops), so a browser who turns decisive keeps a
        # policy that will still dig for the constraints they have left to
        # give - see _policy_for_state.
        route = self._route_for(state) if self.config.use_router else None
        track = route.name if route is not None else None
        opening_track = classify(state.opening).name if self.config.use_router else None

        candidates = retrieve(self.index, state, self.config.retrieval)
        rerank_config = self._rerank_config(track)
        # Deterministic-only pass: seeds the live pool-uncertainty signal
        # (state.observe_pool - entropy, leader margin, stagnation) that both
        # the LLM trigger-gate below and AdaptiveOrchestrator's plan consume,
        # and is itself the safe fallback ranking if the LLM never fires.
        candidates = rerank(self.index, state, candidates, rerank_config, track=track)
        state.observe_pool(candidates)

        llm_scores = {}
        llm_rerank_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if rerank_config.llm_weight > 0.0 and self.llm.is_available() and candidates:
            # Trigger-gated (change 18, rewired in change 19 onto the live pool
            # signal instead of a redundant second rerank pass): a real
            # DeepSeek call is only worth its latency/cost/regression-risk when
            # the deterministic signals haven't already separated #1
            # confidently - see RerankConfig.llm_trigger_margin and
            # DialogState.leader_margin.
            if state.leader_margin < rerank_config.llm_trigger_margin:
                cand_objs = [self.index.products.get(asin, {}) for asin, _ in candidates[:15]]
                llm_scores, llm_rerank_usage = self.llm.rerank_candidates(state.full_text(), cand_objs)
                if llm_scores:
                    candidates = rerank(
                        self.index, state, candidates, rerank_config, track=track, llm_scores=llm_scores
                    )
                    state.observe_pool(candidates, advance=False)

        # Runtime Adaptation: Personalized Context Distillation, after the live
        # pool has been observed. The long-term profile is keyed by user_id
        # (falls back to session_id) and accumulates style/material/brand/price
        # affinities across sessions; see src/context_programming.py.
        user_id = self._session_users.get(session_id, session_id)
        user_prof = self.profile_store.get_or_create(user_id)
        distilled_ctx = ContextDistiller.distill(
            state, user_prof, intent_track=track or "browsing"
        )

        # Adaptive Orchestration: dynamic strategy alignment, now actually
        # applied (change 15 on branch dynamic-state-slot wired the
        # previously-computed-and-discarded plan into real retrieval reroute,
        # phase transitions, policy, and recommendation gating below).
        plan = AdaptiveOrchestrator.align_strategy(distilled_ctx, user_prof, candidates, self.config)
        if self.config.use_adaptive_orchestration:
            # ``terms`` is the standard fused pass already performed above. The
            # other named routes are real strategy switches and receive one
            # re-retrieval with the plan's weighting hint. Deterministic only -
            # a reroute does not re-consult the LLM a second time this turn.
            if plan.retrieval_route != "terms":
                candidates = retrieve(
                    self.index,
                    state,
                    self.config.retrieval,
                    route_hint=plan.retrieval_route,
                )
                candidates = rerank(self.index, state, candidates, rerank_config, track=track)
                state.observe_pool(candidates, advance=False)
            self._apply_plan_to_state(state, plan)
        else:
            plan = None

        attribute = self._policy_for_state(opening_track, state, plan).select(state, candidates)
        if attribute not in ALLOWED_ATTRIBUTES:
            attribute = "other"
        state.record_ask(attribute)

        recommendations = self._shortlist(state, candidates, turn, top_k, track, route, plan=plan)
        # ``ask_attribute`` (above) is what the simulator reads and is unchanged;
        # ``clarify`` only builds the English ``message`` - see src/phrasing.py.
        message = clarify(attribute, state, candidates, self.facets, route, self.config)

        usage = {
            "prompt_tokens": llm_rerank_usage.get("prompt_tokens", 0),
            "completion_tokens": llm_rerank_usage.get("completion_tokens", 0),
        }
        if recommendations and self.llm.is_available():
            top_prods = [self.index.products.get(r["parent_asin"], {}) for r in recommendations[:3]]
            explanation, llm_usage = self.llm.explain_recommendations(state.query_spans(), top_prods)
            if explanation:
                message = explanation
            usage["prompt_tokens"] += llm_usage.get("prompt_tokens", 0)
            usage["completion_tokens"] += llm_usage.get("completion_tokens", 0)

        return self._envelope(message, attribute, recommendations, usage=usage)

    # ---- dual-track helpers --------------------------------------------------

    def _policy_for(self, opening_track: str | None):
        if opening_track is None or not self.config.route_policies:
            return self.config.policy
        return self._buying_policy if opening_track == "buying" else self._browsing_policy

    def _rerank_config(self, track: str | None) -> RerankConfig:
        if track == "buying" and self.config.buying_rerank is not None:
            return self.config.buying_rerank
        if track == "browsing" and self.config.browsing_rerank is not None:
            return self.config.browsing_rerank
        return self.config.rerank

    def _first_recommend_turn(self, track: str | None, route: object) -> int:
        if track == "buying":
            configured = self.config.buying_first_recommend_turn
            if configured is not None:
                return configured
            suggested = getattr(route, "suggested_first_recommend_turn", None)
            return suggested if isinstance(suggested, int) else self.config.first_recommend_turn
        if track == "browsing":
            return self.config.browsing_first_recommend_turn
        return self.config.first_recommend_turn

    def _list_size_ramp(self, track: str | None) -> tuple[int, ...]:
        if track == "buying":
            return self.config.buying_list_size_ramp
        if track == "browsing":
            return self.config.browsing_list_size_ramp
        return self.config.list_size_ramp

    def _route_for(self, state: DialogState):
        """Evolve browsing intent into buying as concrete evidence accumulates."""
        if not state.intent_history:
            route = classify(state.opening)
            reason = "opening classification"
        elif state.override_turn is not None and state.turn_count >= state.override_turn:
            route = BUYING
            reason = "intent override makes the new request authoritative"
        elif state.intent_track == "buying":
            # Buying is sticky: a later polite or vague sentence must not undo
            # explicit requirements already supplied.
            route = BUYING
            reason = "buying intent retained"
        else:
            route = detect_turn_intent(
                state.full_text(),
                state.turn_count,
                state.intent_track,
                state.productive_turns,
            )
            reason = (
                "concrete constraints promoted browsing to buying"
                if route.name == "buying"
                else "session remains exploratory"
            )
        state.update_intent(route, reason)
        return route

    def _policy_for_state(self, opening_track: str | None, state: DialogState, plan: object | None):
        """Switch to targeted clarification only when progress has genuinely stalled.

        Layered on top of the dual-track base policy (_policy_for), not instead
        of it: a decided buyer still gets FixedPolicy and a browser still gets
        InfoGainPolicy right up until the live state itself says progress has
        stalled, so this is an override of last resort, not a replacement for
        the track-aware default.
        """
        base = self._policy_for(opening_track)
        if not self.config.use_adaptive_orchestration or plan is None:
            return base
        phase = getattr(plan, "phase", None)
        # A direct "no preference" is already handled by FixedPolicy's ordered
        # fallback. Replacing that user-respecting fallback with a catalog-only
        # information-gain guess regresses boundary conversations. Info gain is
        # reserved for unexplained stagnation, where the user has not declined a
        # dimension and the broad question genuinely yielded no information.
        if phase == DialogPhase.STAGNATING and not state.dead_attributes:
            return self._targeted_policy
        return base

    @staticmethod
    def _apply_plan_to_state(state: DialogState, plan: object) -> None:
        mapping = {
            DialogPhase.EXPLORING: SessionPhase.EXPLORING,
            DialogPhase.NARROWING: SessionPhase.NARROWING,
            DialogPhase.CONVERGING: SessionPhase.CONVERGING,
            DialogPhase.OVERRIDE_REVERSAL: SessionPhase.OVERRIDE_RECOVERY,
            DialogPhase.STAGNATING: SessionPhase.STAGNATING,
        }
        phase = mapping.get(getattr(plan, "phase", None), SessionPhase.EXPLORING)
        reason = str(getattr(plan, "guidance_action", "standard pipeline"))
        state.transition_to(phase, reason)

    def _shortlist(
        self,
        state: DialogState,
        candidates: list[tuple[str, float]],
        turn: int,
        top_k: int,
        track: str | None = None,
        route: object = None,
        plan: object | None = None,
    ) -> list[dict]:
        top_limit = top_k if isinstance(top_k, int) and top_k > 0 else 10
        first_turn = self._first_recommend_turn(track, route)
        # Track-aware Turn-2 gating (buying_confidence_margin): a buying session
        # is held to a looser confidence margin than browsing, see _confident.
        is_buying = track == "buying"
        sid = state.session_id

        if plan is not None:
            if bool(getattr(plan, "recommendation_cutoff", False)):
                return []
        elif turn < first_turn and not self._confident(candidates, turn, is_buying=is_buying):
            return []
        if self.config.hold_until_stalled and turn < 10:
            # Hold every list until a turn adds no new real constraint (the "no
            # preference" replies parse into spans too, so count real ones only).
            count = self._real_disclosure_count(state)
            rising = count > self._disclosed_count.get(sid, -1)
            self._disclosed_count[sid] = count
            if rising:
                return []

        ramp = self._list_size_ramp(track)
        if plan is not None:
            size = int(getattr(plan, "recommended_slate_size", ramp[-1]))
        else:
            size = ramp[min(max(0, turn - first_turn), len(ramp) - 1)]
        limit = min(size, top_limit)

        if not self.config.elimination_scan:
            return [{"parent_asin": asin} for asin, _score in candidates[:limit]]

        # A product shown on an earlier turn and not hit on is a confirmed
        # non-target (the session would have ended). Drop everything shown so far
        # and return the top of the re-ranked survivors. rerank() runs every turn,
        # so this reflects new constraints; and because it is always the current
        # top of all survivors, a target that reorders downward is still a
        # survivor and surfaces a turn later - no gap.
        shown = self._shown.setdefault(sid, set())
        # Exception: a list shown *before* an intent override does not confirm
        # anything - the evaluator ignores hits until the override lands, so the
        # target may well have been in it. Un-exclude everything on override.
        if (state.override_turn or 0) != self._shown_override.get(sid, 0):
            shown.clear()
            self._shown_override[sid] = state.override_turn or 0
        picks = [asin for asin, _score in candidates if asin not in shown][:limit]
        shown.update(picks)
        return [{"parent_asin": asin} for asin in picks]

    def _confident(self, candidates: list[tuple[str, float]], turn: int, is_buying: bool = False) -> bool:
        """True when the leader is far enough ahead to be worth showing early.

        Emitting a list ends the session the moment the target appears anywhere in
        it, freezing that rank into MRR. Showing an uncertain list early therefore
        trades a good rank later for a poor one now; the margin test is what
        distinguishes the two cases.
        """
        margin = self.config.buying_confidence_margin if is_buying else self.config.confidence_margin
        if margin <= 0.0 or turn < self.config.earliest_recommend_turn or len(candidates) < 2:
            return False
        best, runner_up = candidates[0][1], candidates[1][1]
        if best <= 0.0:
            return False
        return (best - runner_up) / best >= margin

    @staticmethod
    def _real_disclosure_count(state: DialogState) -> int:
        """Count genuine constraint spans the customer has disclosed.

        The simulator's "I don't have a preference for X" replies also parse into
        a constraint span, so a plain productive-turn counter keeps rising every
        turn. Those spans always contain "preference for"; real product
        constraints do not. Used by ``hold_until_stalled`` to tell "new
        information arrived" from "the customer just declined".
        """
        return sum(1 for span in state.query_spans() if "preference for" not in span)

    @staticmethod
    def _envelope(
        message: str,
        attribute: str | None,
        recommendations: list[dict],
        usage: dict[str, int] | None = None,
    ) -> dict:
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": recommendations,
            "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0},
        }
