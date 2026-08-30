"""S2 - highly sensitive intent detection and dual-track routing.

Splits traffic into:
  * "buying" track: high purchase intent, explicit constraint density, decisive
    dialogue posture, and precision candidate filtering.
  * "browsing" track: exploratory search, open discovery questions, and conservative
    recommendation timing to protect MRR against premature unconstrained recommendations.

Also detects mid-session intent signals:
  * intent override (preference reversal / slot modification)
  * boundary / indifference (customer declining a requested attribute)

Classification combines:
  1. Multi-pattern linguistic cues (modal verbs, specification markers, exploration cues).
  2. Concrete facet & entity density (materials, colors, sizes, prices, feature specs).
  3. Continuous score & confidence calibration with safe defaults.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.llm import get_llm_client


# -------------------------------------------------------------------------------
# Linguistic Cue Matchers
# -------------------------------------------------------------------------------

# Direct buying / high-intent cues
BUYING_CUES = re.compile(
    r"\b("
    r"key requirement|must (?:be|have|include)|i need|needs to be|specifically|"
    r"looking for a specific|it has to|requirement is|exactly|only want|"
    r"looking to buy|ready to purchase|trying to find a specific|"
    r"specifically looking for|must include|strictly|want a specific|"
    r"searching for a specific|targeted for|particular|essential requirement"
    r")\b",
    re.IGNORECASE,
)

# Exploratory / browsing cues
BROWSING_CUES = re.compile(
    r"\b("
    r"still exploring|just (?:looking|browsing)|not sure|no idea|"
    r"open to|any (?:suggestions|ideas)|show me some|what do you have|"
    r"i'?m exploring|help me find|looking around|gift ideas?|"
    r"recommend something|anything (?:good|nice|suitable)|haven'?t decided|"
    r"not really sure|just checking|looking for options|what are some"
    r")\b",
    re.IGNORECASE,
)

# Preference reversal cues (intent override)
OVERRIDE_CUES = re.compile(
    r"\b("
    r"actually|instead|ignore(?:\s+my)?|forget|disregard|scratch that|"
    r"changed my mind|on second thought|no longer|rather than|not what i"
    r")\b",
    re.IGNORECASE,
)

# Indifference / boundary cues
BOUNDARY_CUES = re.compile(
    r"\b("
    r"don'?t have (?:a|an|any)?\s*(?:additional\s+)?preference|"
    r"no preference|use your judgment?|doesn'?t matter|either is fine|"
    r"any is fine|up to you|whatever you think|no specific preference"
    r")\b",
    re.IGNORECASE,
)

# -------------------------------------------------------------------------------
# Domain Facet & Entity Patterns for Constraint Density Detection
# -------------------------------------------------------------------------------

FACET_PATTERNS: dict[str, re.Pattern] = {
    "material": re.compile(
        r"\b(leather|cotton|polyester|nylon|wool|spandex|silk|rayon|denim|linen|"
        r"cashmere|suede|velvet|satin|mesh|fleece|acrylic|bamboo|alloy|titanium|"
        r"rubber|canvas|gold plated|sterling silver|stainless steel)\b",
        re.IGNORECASE,
    ),
    "color": re.compile(
        r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange|"
        r"beige|navy|silver|gold|teal|burgundy|khaki|olive|charcoal|ivory)\b",
        re.IGNORECASE,
    ),
    "price": re.compile(
        r"(\$\s*\d+(?:\.\d+)?|under\s*\$?\d+|around\s*\$?\d+|budget\s*(?:around|of|under)?\s*\$?\d+|priced)",
        re.IGNORECASE,
    ),
    "spec": re.compile(
        r"\b(waterproof|water resistant|closure|buckle|zipper|battery|sleeve|"
        r"collar|pocket|sole|strap|fit|size|width|cuff|heel|pendant|karat|chronograph)\b",
        re.IGNORECASE,
    ),
    "use_case": re.compile(
        r"\b(hiking|running|gym|workout|yoga|travel|office|wedding|party|beach|"
        r"swimming|winter|summer|outdoor|athletic)\b",
        re.IGNORECASE,
    ),
}


# -------------------------------------------------------------------------------
# Route & Analysis Data Structures
# -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Route:
    """Dialog posture and execution parameters for one intent track."""

    name: str
    #: Conversational phrasing prefix applied to questions.
    tone: str
    #: Calibrated confidence in the detected track in [0.0, 1.0].
    confidence: float = 1.0
    #: Raw score component for buying intent.
    buying_score: float = 0.0
    #: Raw score component for browsing intent.
    browsing_score: float = 0.0
    #: Matched linguistic cues.
    detected_cues: tuple[str, ...] = ()
    #: Extracted domain facets / entity constraints.
    detected_facets: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Scenario hint if recognized ('buying', 'browsing', 'intent_override', 'boundary').
    scenario_hint: str | None = None
    #: Recommended earliest turn to begin emitting recommendations.
    suggested_first_recommend_turn: int = 3

    @property
    def is_buying(self) -> bool:
        return self.name == "buying"

    @property
    def is_browsing(self) -> bool:
        return self.name == "browsing"


BUYING = Route(
    name="buying",
    tone="To narrow this down: ",
    confidence=1.0,
    buying_score=2.0,
    browsing_score=0.0,
    suggested_first_recommend_turn=2,
)

BROWSING = Route(
    name="browsing",
    tone="To point you in the right direction: ",
    confidence=1.0,
    buying_score=0.0,
    browsing_score=2.0,
    suggested_first_recommend_turn=3,
)


# -------------------------------------------------------------------------------
# Intent Detection Engine
# -------------------------------------------------------------------------------

def extract_opening_facets(text: str) -> dict[str, tuple[str, ...]]:
    """Extract recognized domain attributes present in the utterance."""
    results: dict[str, tuple[str, ...]] = {}
    for facet, pattern in FACET_PATTERNS.items():
        matches = tuple(m.group(0).lower() for m in pattern.finditer(text))
        if matches:
            results[facet] = matches
    return results


def route_with_tie_breaker(
    buying_score: float,
    browsing_score: float,
    *,
    tie_breaker: callable | None = None,
    high_confidence_margin: float = 0.6,
    strong_signal_threshold: float = 1.5,
) -> str:
    """Choose a route with a confidence gate and explicit tie-breaker fallback.

    This matches the PR1 design: strong signals route immediately, ambiguous ones
    defer to the tie-breaker rather than forcing a brittle all-or-nothing decision.
    """
    if buying_score >= strong_signal_threshold and buying_score >= browsing_score + high_confidence_margin:
        return "buying"
    if browsing_score >= strong_signal_threshold and browsing_score >= buying_score + high_confidence_margin:
        return "browsing"

    if buying_score > browsing_score:
        return "buying"
    if browsing_score > buying_score:
        return "browsing"

    if tie_breaker is not None:
        return tie_breaker(buying_score, browsing_score)
    return "browsing"


def _llm_route_hint(text: str) -> Route | None:
    """Optional Gemini-backed routing hint. Returns None when the client is unavailable."""
    client = get_llm_client()
    if not client.is_configured:
        return None
    prompt = (
        "Classify the shopping intent of the following customer message. "
        "Return strict JSON with keys: route, confidence, reason, is_override, is_boundary. "
        "Routes must be 'buying' or 'browsing'.\n\n"
        f"Message: {text}"
    )
    payload = client.generate_json(prompt)
    if not isinstance(payload, dict):
        return None
    route = payload.get("route")
    if route not in {"buying", "browsing"}:
        return None
    return Route(
        name=str(route),
        tone="To narrow this down: " if route == "buying" else "To point you in the right direction: ",
        confidence=float(payload.get("confidence", 0.5) or 0.5),
        buying_score=2.0 if route == "buying" else 0.0,
        browsing_score=2.0 if route == "browsing" else 0.0,
        scenario_hint="intent_override" if payload.get("is_override") else "boundary" if payload.get("is_boundary") else route,
        suggested_first_recommend_turn=2 if route == "buying" else 3,
    )


def classify(opening: str) -> Route:
    """Highly sensitive intent classification of the customer's opening message.

    Combines:
      - Linguistic cues (modal verbs, specification markers, exploration cues)
      - Entity / constraint density (material, color, specs, price patterns)
      - Structural delimiters (e.g., ':', ';', 'requirement is:')

    Browsing remains the safe fallback for ambiguous queries: mistaking a browser
    for a buyer commits to constraints they never stated, whereas treating a buyer
    as a browser costs at most one extra turn.
    """
    text = (opening or "").strip()
    if not text:
        return BROWSING

    buying_matches = [m.group(0) for m in BUYING_CUES.finditer(text)]
    browsing_matches = [m.group(0) for m in BROWSING_CUES.finditer(text)]
    override_matches = [m.group(0) for m in OVERRIDE_CUES.finditer(text)]
    boundary_matches = [m.group(0) for m in BOUNDARY_CUES.finditer(text)]

    detected_facets = extract_opening_facets(text)
    total_facet_count = sum(len(values) for values in detected_facets.values())

    # Calculate multi-signal scores
    b_score = len(buying_matches) * 2.0 + total_facet_count * 0.8
    if ":" in text or ";" in text:
        b_score += 0.5

    br_score = len(browsing_matches) * 2.5
    if not detected_facets and not buying_matches:
        br_score += 0.5

    # Determine scenario hint
    scenario_hint: str | None = None
    if override_matches:
        scenario_hint = "intent_override"
    elif boundary_matches:
        scenario_hint = "boundary"
    elif b_score > br_score:
        scenario_hint = "buying"
    else:
        scenario_hint = "browsing"

    total = b_score + br_score + 1e-6
    buying_conf = b_score / total
    browsing_conf = br_score / total

    # Decision rule:
    # Strong evidence returns immediately; ambiguous scores intentionally defer to
    # the tie-breaker so that near-equal signals do not trigger a brittle route.
    tie_breaker = lambda b_score, br_score: "browsing" if br_score >= b_score else "buying"
    name = route_with_tie_breaker(b_score, br_score, tie_breaker=tie_breaker)

    # Gemini is only used for genuinely ambiguous routing decisions, not for every
    # message in the evaluator. This avoids burning quota on routine traffic while
    # still allowing a single optional LLM tie-break on uncertain cases.
    if abs(b_score - br_score) <= 0.75:
        llm_hint = _llm_route_hint(text)
        if llm_hint is not None:
            return llm_hint

    if name == "browsing":
        tone = "To point you in the right direction: "
        confidence = browsing_conf if br_score > 0.0 and browsing_matches else 0.5
        rec_turn = 3
    else:
        tone = "To narrow this down: "
        confidence = buying_conf
        rec_turn = 2 if buying_conf >= 0.75 else 3

    detected_cues = tuple(buying_matches + browsing_matches + override_matches + boundary_matches)
    return Route(
        name=name,
        tone=tone,
        confidence=round(confidence, 3),
        buying_score=round(b_score, 3),
        browsing_score=round(br_score, 3),
        detected_cues=detected_cues,
        detected_facets=detected_facets,
        scenario_hint=scenario_hint,
        suggested_first_recommend_turn=rec_turn,
    )


def detect_turn_intent(
    text: str,
    turn: int = 1,
    current_track: str = "browsing",
    productive_turns: int = 0,
) -> Route:
    """Dynamic multi-turn intent tracking.

    Evaluates whether an initially exploratory session has transitioned into
    focused buying after concrete constraints were provided.
    """
    route = classify(text)
    # If the session started as browsing but has accumulated strong evidence:
    if current_track == "browsing" and turn > 1 and (productive_turns >= 2 or route.buying_score >= 1.5):
        return Route(
            name="buying",
            tone="To narrow this down: ",
            confidence=max(route.confidence, 0.8),
            buying_score=route.buying_score,
            browsing_score=route.browsing_score,
            detected_cues=route.detected_cues,
            detected_facets=route.detected_facets,
            scenario_hint="buying",
            suggested_first_recommend_turn=min(route.suggested_first_recommend_turn, turn),
        )
    return route
