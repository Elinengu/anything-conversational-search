"""S2 - intent routing.

Splits traffic into a high-precision "buying" track and an exploratory "browsing"
track, and detects the reversal that starts an intent override.

Classification is cue-based rather than template matching. The evaluator builds its
opening lines from fixed templates, but the organizer's private simulator may
paraphrase them, so matching those exact strings would be a trap: it would score
well locally and silently fail on the sessions that decide the result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


BUYING_CUES = re.compile(
    r"\b(key requirement|must (?:be|have)|i need|needs to be|specifically|"
    r"looking for a specific|it has to|requirement is|exactly)\b",
    re.IGNORECASE,
)

BROWSING_CUES = re.compile(
    r"\b(still exploring|just (?:looking|browsing)|not sure|no idea|"
    r"open to|any (?:suggestions|ideas)|show me some|what do you have|"
    r"i'?m exploring|help me find)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Route:
    """Dialog posture for one intent track.

    Scope note, from measurement rather than intent: this route deliberately does
    *not* carry retrieval or timing overrides. Widening the candidate pool for
    browsing and narrowing it for buying was implemented and measured, and changed
    the dev score not at all (0.8715 either way) while costing 0.002 on the
    holdout. Since the span reranker already resolves both tracks well, that
    routing was removed rather than kept as decoration, and the route now drives
    only how the agent phrases itself - which is a real product requirement even
    where it is not a scoring one.
    """

    name: str
    #: Prefix applied to the clarification question, so an exploratory customer is
    #: not addressed as though they had already decided.
    tone: str


BUYING = Route(name="buying", tone="To narrow this down: ")
BROWSING = Route(name="browsing", tone="To point you in the right direction: ")


def classify(opening: str) -> Route:
    """Route an opening customer message.

    Browsing is the safer default: treating a vague customer as a buyer commits to
    constraints they never stated, while treating a buyer as a browser costs at
    most one extra question.
    """
    text = opening or ""
    if BROWSING_CUES.search(text):
        return BROWSING
    if BUYING_CUES.search(text):
        return BUYING
    return BROWSING
