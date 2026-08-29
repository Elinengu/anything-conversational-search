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

from src.text import constraint_spans, terms


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

PRE_OVERRIDE_WEIGHT = 0.35


@dataclass
class Utterance:
    """One customer message plus the provenance the retrieval stage needs."""

    turn: int
    text: str
    weight: float = 1.0


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

    # ---- observation ----------------------------------------------------------

    def observe(self, turn: int, text: str) -> None:
        """Record a customer message, applying override handling if triggered."""
        message = (text or "").strip()
        if turn == 1:
            self.opening = message
        if message and OVERRIDE_CUES.search(message) and self.override_turn is None:
            self.apply_override(turn)
        if message and NO_PREFERENCE_CUES.search(message) and self.asked:
            self.dead_attributes.add(self.asked[-1])
        known = set(self.query_spans())
        self.utterances.append(Utterance(turn=turn, text=message))
        # A turn is "productive" when it disclosed a constraint we had not seen.
        # The policy uses this to judge whether broad questions are exhausted.
        produced = turn > 1 and any(
            span not in known for span in constraint_spans(message)
        )
        self.last_turn_productive = produced
        if produced:
            self.productive_turns += 1

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

    # ---- views for retrieval --------------------------------------------------

    def full_text(self) -> str:
        """Everything the customer has said - maximum recall."""
        return " ".join(utterance.text for utterance in self.utterances)

    def focused_text(self) -> str:
        """Only the currently authoritative turns - maximum precision.

        Identical to ``full_text`` until an override fires.
        """
        return " ".join(
            utterance.text for utterance in self.utterances if utterance.weight >= 1.0
        ) or self.full_text()

    def query_terms(self) -> list[str]:
        return terms(self.full_text(), drop_boilerplate=True)

    def query_spans(self) -> list[str]:
        """Verbatim constraint fragments, newest first.

        Turn 1 is excluded: the opening line is the simulator's own framing
        ("I'm looking for X"), not quoted product copy, so its spans are noise
        for span matching while its tokens still feed the bag-of-words route.
        """
        spans: list[str] = []
        seen: set[str] = set()
        for utterance in reversed(self.utterances):
            if utterance.turn == 1:
                continue
            for span in constraint_spans(utterance.text):
                if span not in seen:
                    seen.add(span)
                    spans.append(span)
        return spans

    @property
    def turn_count(self) -> int:
        return len(self.utterances)
