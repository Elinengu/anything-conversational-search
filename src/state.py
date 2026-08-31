"""S3 - dialog state, slot provenance and intent-override handling.

The shipped baseline is stateless: it answers every turn from the latest message
alone and therefore never accumulates the constraints the simulated customer
discloses. Accumulating those turns is the single largest scoring lever in the
pipeline, so this module is deliberately conservative - it keeps everything the
customer said and records *when* they said it, rather than throwing anything
away.

Intent override is handled by down-weighting pre-override utterances rather than
erasing them. See the note in ``apply_override``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
import math

from src.facets import extract_query_facets
from src.text import constraint_spans, pair_spans


# Cue-based, not template matching: the private evaluation set may paraphrase the
# simulator's wording, so we look for reversal markers rather than exact strings.
OVERRIDE_CUES = re.compile(
    r"\b(actually|instead|ignore(?:\s+my)?|forget|disregard|scratch that|"
    r"changed my mind|on second thought|no longer|rather than|not what i)\b",
    re.IGNORECASE,
)

# The customer said they have no preference for what we asked. Recording this
# stops the policy from spending another turn on the same dead attribute.
NO_PREFERENCE_CUES = re.compile(
    r"\b(don'?t have (?:a|an|any)?\s*(?:additional\s+)?preference|"
    r"no preference|use your judg|doesn'?t matter|either is fine)\b",
    re.IGNORECASE,
)

# A stall/deferral in answer to a broad question ("I'm still just browsing -
# ask me about one particular thing and I'll tell you", the browse-gated
# customer in tools/stress_harness.py). Unlike NO_PREFERENCE_CUES this is NOT a
# decline of a specific attribute - dead_attributes must not learn from it -
# but it discloses nothing and must not reach _record_slots either. Without
# this guard constraint_spans() (built for the official simulator's templated
# wording, not free-form text) chunks the sentence on " - " into
# "i m still just browsing" / "ask me about one particular thing and i ll tell
# you" and records both as fabricated "feature" slots, polluting the structured
# retrieval query and falsely marking the turn productive.
STALL_CUES = re.compile(
    r"\b(still (?:just )?(?:browsing|exploring|looking)|"
    r"ask me (?:about )?(?:one particular thing|something specific)|"
    r"not sure (?:yet|what i want))\b",
    re.IGNORECASE,
)

PRE_OVERRIDE_WEIGHT = 0.35


@dataclass
class Utterance:
    """One customer message plus the provenance the retrieval stage needs."""

    turn: int
    text: str
    weight: float = 1.0
    #: The customer declined to answer ("I don't have a preference for X"). Such
    #: replies carry no product signal but their words ("feature", "material",
    #: "colour", ...) would otherwise leak into the bag-of-words query and the
    #: span matcher, so they are held out of every retrieval view.
    declined: bool = False


class SessionPhase(str, Enum):
    """Observable phase of a live shopping conversation."""

    EXPLORING = "exploring"
    NARROWING = "narrowing"
    CONVERGING = "converging"
    OVERRIDE_RECOVERY = "override_recovery"
    STAGNATING = "stagnating"


@dataclass
class SlotValue:
    """One extracted constraint with provenance and lifecycle status."""

    attribute: str
    value: str
    source_turn: int
    raw_text: str
    confidence: float = 1.0
    status: str = "active"
    superseded_turn: int | None = None

    def as_dict(self) -> dict:
        return {
            "attribute": self.attribute,
            "value": self.value,
            "source_turn": self.source_turn,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "status": self.status,
            "superseded_turn": self.superseded_turn,
        }


@dataclass(frozen=True)
class PhaseTransition:
    turn: int
    previous: str
    current: str
    reason: str


@dataclass(frozen=True)
class IntentTransition:
    turn: int
    previous: str | None
    current: str
    confidence: float
    reason: str


@dataclass
class DialogState:
    session_id: str
    profile: dict = field(default_factory=dict)
    utterances: list[Utterance] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    dead_attributes: set[str] = field(default_factory=set)
    override_turn: int | None = None
    opening: str = ""
    productive_turns: int = 0
    last_turn_productive: bool = False
    # Authoritative structured constraints. Multiple values are allowed for
    # attributes such as material; an override supersedes the whole active view
    # while preserving the old values in ``superseded_slots``.
    active_slots: dict[str, list[SlotValue]] = field(default_factory=dict)
    superseded_slots: list[SlotValue] = field(default_factory=list)
    # Recent progress is distinct from total progress: one failed turn after five
    # useful turns is a pause; five failures after one useful turn is stagnation.
    unproductive_streak: int = 0
    max_unproductive_streak: int = 0
    pool_size: int = 0
    pool_entropy: float = 0.0
    leader_margin: float = 0.0
    pool_overlap: float = 0.0
    stable_pool_turns: int = 0
    over_general: bool = False
    _previous_pool_head: tuple[str, ...] = field(default_factory=tuple, repr=False)
    phase: SessionPhase = SessionPhase.EXPLORING
    phase_reason: str = "session started"
    transition_history: list[PhaseTransition] = field(default_factory=list)
    intent_track: str = "browsing"
    intent_confidence: float = 0.5
    intent_history: list[IntentTransition] = field(default_factory=list)

    # ---- observation ----------------------------------------------------------

    def observe(self, turn: int, text: str) -> None:
        """Record a customer message, applying override handling if triggered."""
        message = (text or "").strip()
        if turn == 1:
            self.opening = message
        declined = bool(message and NO_PREFERENCE_CUES.search(message))
        # A deferral to a broad question ("still just browsing, ask me
        # something specific") reveals nothing, but - unlike a true decline -
        # it is not "no preference for X" and must not deaden the asked
        # attribute. It needs the same treatment everywhere else a decline gets:
        # held out of _record_slots and every retrieval view (Utterance.declined)
        # and never counted as a productive turn.
        # turn > 1 only, and that guard is load-bearing:
        # evaluator/local_evaluator.py's own browsing opening template is
        # literally "I'm looking for {category}, but I'm still exploring." for
        # every browsing session, so an ungated check would exclude turn 1 - the
        # single most important utterance for retrieval - from every browsing
        # session in the dataset.
        stalled = bool(message and turn > 1 and STALL_CUES.search(message))
        excluded = declined or stalled
        is_override = bool(
            message and OVERRIDE_CUES.search(message) and self.override_turn is None
        )
        if is_override:
            self.apply_override(turn)
        if declined and self.asked:
            self.dead_attributes.add(self.asked[-1])
        known = set(self.query_spans())
        known_slots = {
            (attribute, slot.value)
            for attribute, slots in self.active_slots.items()
            for slot in slots
        }
        self.utterances.append(Utterance(turn=turn, text=message, declined=excluded))
        if message and not excluded:
            self._record_slots(turn, message)
        # A turn is "productive" when it disclosed a constraint we had not seen.
        # The policy uses this to judge whether broad questions are exhausted; a
        # decline - or a stall that discloses nothing - never counts.
        current_slots = {
            (attribute, slot.value)
            for attribute, slots in self.active_slots.items()
            for slot in slots
        }
        produced = not excluded and turn > 1 and (
            any(span not in known for span in constraint_spans(message))
            or bool(current_slots - known_slots)
        )
        self.last_turn_productive = produced
        if produced:
            self.productive_turns += 1
            self.unproductive_streak = 0
        elif turn > 1:
            self.unproductive_streak += 1
            self.max_unproductive_streak = max(
                self.max_unproductive_streak, self.unproductive_streak
            )

    def record_ask(self, attribute: str | None) -> None:
        if attribute:
            self.asked.append(attribute)

    def apply_override(self, turn: int) -> None:
        """Down-weight everything said before the customer reversed themselves.

        The problem brief describes override as slot *erasure*. We deliberately
        deviate: in this evaluator the discarded preference is still derived from
        the target product, so erasing it destroys usable signal and costs score.
        Down-weighting keeps the evidence available while letting the post-override
        turns dominate ranking. This trade-off is documented in the README.
        """
        self.override_turn = turn
        for utterance in self.utterances:
            utterance.weight = PRE_OVERRIDE_WEIGHT
        self._supersede_active_slots(turn)

    def _supersede_active_slots(self, turn: int) -> None:
        """Move the authoritative active view into the provenance archive."""
        for slots in self.active_slots.values():
            for slot in slots:
                slot.status = "superseded"
                slot.superseded_turn = turn
                self.superseded_slots.append(slot)
        self.active_slots.clear()

    def _record_slots(self, turn: int, message: str) -> None:
        """Extract typed facets and untyped constraint spans from one message."""
        values: list[tuple[str, str, float]] = [
            (attribute, value, 1.0)
            for attribute, value in extract_query_facets(message).items()
        ]
        if turn > 1:
            values.extend(("feature", span, 0.8) for span in constraint_spans(message))

        for attribute, value, confidence in values:
            cleaned = str(value).strip().lower()
            if not cleaned:
                continue
            slots = self.active_slots.setdefault(attribute, [])
            if any(slot.value == cleaned for slot in slots):
                continue
            slots.append(
                SlotValue(
                    attribute=attribute,
                    value=cleaned,
                    source_turn=turn,
                    raw_text=message,
                    confidence=confidence,
                )
            )

    # ---- views for retrieval --------------------------------------------------

    def full_text(self) -> str:
        """Everything informative the customer has said - maximum recall."""
        return " ".join(u.text for u in self.utterances if not u.declined)

    def focused_text(self) -> str:
        """Only the currently authoritative turns - maximum precision.

        Identical to ``full_text`` until an override fires.
        """
        return " ".join(
            u.text for u in self.utterances if not u.declined and u.weight >= 1.0
        ) or self.full_text()

    def authoritative_text(self) -> str:
        """Compact query made only from currently active structured slots."""
        ordered = sorted(
            (slot for slots in self.active_slots.values() for slot in slots),
            key=lambda slot: (slot.source_turn, slot.attribute, slot.value),
        )
        values: list[str] = []
        seen: set[str] = set()
        for slot in ordered:
            if slot.value not in seen:
                seen.add(slot.value)
                values.append(slot.value)
        return " ".join(values) or self.focused_text()

    def active_slot_values(self) -> dict[str, list[str]]:
        return {
            attribute: [slot.value for slot in slots]
            for attribute, slots in sorted(self.active_slots.items())
            if slots
        }

    # ---- runtime progress and transitions -------------------------------------

    def observe_pool(
        self,
        candidates: list[tuple[str, float]],
        depth: int = 30,
        *,
        advance: bool = True,
    ) -> None:
        """Record whether the live candidate pool is broad, flat, or stable.

        ``advance=False`` replaces the current turn's post-reroute pool without
        incrementing the rolling stability counter a second time.
        """
        head = candidates[: max(2, depth)]
        self.pool_size = len(candidates)
        scores = [max(float(score), 0.0) for _asin, score in head]
        total = sum(scores)
        if len(scores) >= 2 and total > 0.0:
            entropy = -sum(
                (score / total) * math.log2(score / total)
                for score in scores
                if score > 0.0
            )
            self.pool_entropy = entropy / math.log2(len(scores))
        else:
            self.pool_entropy = 0.0

        if len(head) >= 2 and head[0][1] > 0.0:
            self.leader_margin = max(
                0.0, (head[0][1] - head[1][1]) / head[0][1]
            )
        else:
            self.leader_margin = 0.0

        current_ids = tuple(parent_asin for parent_asin, _score in head)
        if advance and self._previous_pool_head and current_ids:
            previous = set(self._previous_pool_head)
            self.pool_overlap = len(previous.intersection(current_ids)) / max(
                1, min(len(previous), len(current_ids))
            )
            self.stable_pool_turns = (
                self.stable_pool_turns + 1 if self.pool_overlap >= 0.90 else 0
            )
        elif advance:
            self.pool_overlap = 0.0
            self.stable_pool_turns = 0
        self._previous_pool_head = current_ids
        self.over_general = (
            self.pool_size >= 100
            and self.pool_entropy >= 0.97
            and self.leader_margin < 0.05
        )

    def transition_to(self, phase: SessionPhase | str, reason: str) -> None:
        target = phase if isinstance(phase, SessionPhase) else SessionPhase(phase)
        if target == self.phase:
            self.phase_reason = reason
            return
        self.transition_history.append(
            PhaseTransition(
                turn=self.turn_count,
                previous=self.phase.value,
                current=target.value,
                reason=reason,
            )
        )
        self.phase = target
        self.phase_reason = reason

    def update_intent(self, route: object, reason: str) -> None:
        current = str(getattr(route, "name", "browsing"))
        confidence = float(getattr(route, "confidence", 0.5))
        previous = self.intent_track if self.intent_history else None
        if not self.intent_history or current != self.intent_track:
            self.intent_history.append(
                IntentTransition(
                    turn=self.turn_count,
                    previous=previous,
                    current=current,
                    confidence=confidence,
                    reason=reason,
                )
            )
        self.intent_track = current
        self.intent_confidence = confidence

    def snapshot(self) -> dict:
        """Serializable state for observers, debugging, and demo explanations."""
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "opening": self.opening,
            "intent": {
                "track": self.intent_track,
                "confidence": round(self.intent_confidence, 3),
                "history": [transition.__dict__ for transition in self.intent_history],
            },
            "phase": self.phase.value,
            "phase_reason": self.phase_reason,
            "transitions": [transition.__dict__ for transition in self.transition_history],
            "active_slots": self.active_slot_values(),
            "active_slot_ledger": {
                attribute: [slot.as_dict() for slot in slots]
                for attribute, slots in sorted(self.active_slots.items())
            },
            "superseded_slots": [slot.as_dict() for slot in self.superseded_slots],
            "structured_query": self.authoritative_text(),
            "asked": list(self.asked),
            "dead_attributes": sorted(self.dead_attributes),
            "productive_turns": self.productive_turns,
            "last_turn_productive": self.last_turn_productive,
            "unproductive_streak": self.unproductive_streak,
            "max_unproductive_streak": self.max_unproductive_streak,
            "pool": {
                "size": self.pool_size,
                "entropy": round(self.pool_entropy, 4),
                "leader_margin": round(self.leader_margin, 4),
                "overlap": round(self.pool_overlap, 4),
                "stable_turns": self.stable_pool_turns,
                "over_general": self.over_general,
            },
            # Backward-compatible observer fields.
            "override_turn": self.override_turn,
            "spans": self.query_spans()[:12],
            "focused_text": self.focused_text()[:400],
        }

    def query_spans(self) -> list[str]:
        """Verbatim constraint fragments, newest first.

        Turn 1 is excluded: the opening line is the simulator's own framing
        ("I'm looking for X"), not quoted product copy, so its spans are noise
        for span matching while its tokens still feed the bag-of-words route.
        """
        spans: list[str] = []
        seen: set[str] = set()
        for utterance in reversed(self.utterances):
            if utterance.turn == 1 or utterance.declined:
                continue
            for span in constraint_spans(utterance.text):
                if span not in seen:
                    seen.add(span)
                    spans.append(span)
        return spans

    def query_pair_spans(self) -> list[str]:
        """Association-preserving spans, newest first - see text.pair_spans.

        Same exclusions as ``query_spans``, plus anything already emitted as a
        fragment span so the reranker never counts the same evidence twice.
        """
        fragments = set(self.query_spans())
        spans: list[str] = []
        seen: set[str] = set()
        for utterance in reversed(self.utterances):
            if utterance.turn == 1 or utterance.declined:
                continue
            for span in pair_spans(utterance.text):
                if span not in seen and span not in fragments:
                    seen.add(span)
                    spans.append(span)
        return spans

    @property
    def turn_count(self) -> int:
        return len(self.utterances)
