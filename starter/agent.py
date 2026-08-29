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
from src.policy import ALLOWED_ATTRIBUTES, FixedPolicy  # noqa: E402
from src.rerank import RerankConfig, rerank  # noqa: E402
from src.retrieval import RetrievalConfig, retrieve  # noqa: E402
from src.router import classify  # noqa: E402
from src.state import DialogState  # noqa: E402


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
    list_size_ramp: tuple[int, ...] = (10,)
    # Optional confidence gate: emit earlier than first_recommend_turn when the top
    # candidate clearly leads the pool. 0.0 disables it.
    # 0.15-0.50 all beat 0.0 on dev and holdout alike; the curve is flat, so this
    # sits mid-plateau rather than at either split's argmax.
    confidence_margin: float = 0.20
    earliest_recommend_turn: int = 2
    #: Route the customer-facing phrasing by detected intent.
    use_router: bool = True
    # Once information stops arriving (~turn 3) the ranking no longer changes, so
    # re-showing the same top 10 wastes the remaining turns. Instead, freeze the
    # ranking on the first list shown and walk it in windows: turn 3 -> ranks
    # 1-10, turn 4 -> 11-20, turn 5 -> 21-30, ... A miss only costs a turn, and
    # the session ends on the first hit, so scanning deeper is close to free.
    scan_windows: bool = True


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
        if self.config.policy is None:
            # FixedPolicy is the default because it wins on the held-out split
            # (0.8349 vs 0.8119). InfoGainPolicy is the more interesting design and
            # ties on dev; see README "Clarification policy" for the measurements.
            self.config.policy = FixedPolicy()
        self._states: dict[str, DialogState] = {}
        # Per-session state for the windowed shortlist (see AgentConfig.scan_windows).
        self._scan_pool: dict[str, list[str]] = {}
        self._scan_cursor: dict[str, int] = {}
        self._scan_frozen_sig: dict[str, tuple[int, int]] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._states[session_id] = DialogState(
            session_id=session_id,
            profile=user_profile if isinstance(user_profile, dict) else {},
        )
        self._scan_pool.pop(session_id, None)
        self._scan_cursor.pop(session_id, None)
        self._scan_frozen_sig.pop(session_id, None)

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

        route = classify(state.opening) if self.config.use_router else None
        candidates = retrieve(self.index, state, self.config.retrieval)
        candidates = rerank(self.index, state, candidates, self.config.rerank)

        attribute = self.config.policy.select(state, candidates)
        if attribute not in ALLOWED_ATTRIBUTES:
            attribute = "other"
        state.record_ask(attribute)

        recommendations = self._shortlist(state, candidates, turn, top_k)
        question = self.config.policy.question(attribute)
        if route is not None:
            question = route.tone + question[0].lower() + question[1:]
        return self._envelope(question, attribute, recommendations)

    def _shortlist(
        self,
        state: DialogState,
        candidates: list[tuple[str, float]],
        turn: int,
        top_k: int,
    ) -> list[dict]:
        top_limit = top_k if isinstance(top_k, int) and top_k > 0 else 10
        first_turn = self.config.first_recommend_turn

        if turn < first_turn:
            # Early, high-confidence emission always shows the current best slice.
            if self._confident(candidates, turn):
                return [{"parent_asin": asin} for asin, _score in candidates[:top_limit]]
            return []

        ramp = self.config.list_size_ramp
        size = ramp[min(max(0, turn - first_turn), len(ramp) - 1)]
        limit = min(size, top_limit)

        if not self.config.scan_windows:
            return [{"parent_asin": asin} for asin, _score in candidates[:limit]]

        # Serve the shortlist from a frozen ranking, walked in windows (1-10, then
        # 11-20, ...). Freezing avoids a gap: if the live pool reorders between
        # turns a target could fall between an already-passed window and one not
        # yet reached and never be shown. Re-snapshot (and restart the scan) each
        # time the information state genuinely changes - a new real constraint, or
        # an intent override taking effect (which the evaluator ignores hits
        # before). Once it settles, the scan proceeds undisturbed.
        sid = state.session_id
        pool = self._scan_pool.get(sid)
        signature = (self._real_disclosure_count(state), state.override_turn or 0)
        if pool is None or signature != self._scan_frozen_sig.get(sid):
            pool = self._scan_pool[sid] = [asin for asin, _score in candidates]
            self._scan_frozen_sig[sid] = signature
            self._scan_cursor[sid] = 0

        start = self._scan_cursor.get(sid, 0)
        window = pool[start:start + limit]
        if window:
            self._scan_cursor[sid] = start + limit
            return [{"parent_asin": asin} for asin in window]
        # Pool exhausted - fall back to re-showing the strongest slice.
        return [{"parent_asin": asin} for asin in pool[:limit]]

    def _confident(self, candidates: list[tuple[str, float]], turn: int) -> bool:
        """True when the leader is far enough ahead to be worth showing early.

        Emitting a list ends the session the moment the target appears anywhere in
        it, freezing that rank into MRR. Showing an uncertain list early therefore
        trades a good rank later for a poor one now; the margin test is what
        distinguishes the two cases.
        """
        margin = self.config.confidence_margin
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
        turn and would re-freeze the scan pool forever. Those spans always contain
        "preference for"; real product constraints do not.
        """
        return sum(1 for span in state.query_spans() if "preference for" not in span)

    @staticmethod
    def _envelope(message: str, attribute: str | None, recommendations: list[dict]) -> dict:
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": recommendations,
            # No model is used on the offline path, so the honest count is zero.
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
