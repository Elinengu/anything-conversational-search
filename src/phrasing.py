"""S4b - clarification-question phrasing.

The clarification policy (S4, ``src/policy.py``) decides *what to ask about* -
the ``ask_attribute`` field the simulator reads. This module decides *how to say
it*.

``FixedPolicy`` keeps ``ask_attribute="other"`` because that is the score-optimal
extraction on the local evaluator: the simulator ignores the English sentence
entirely, and a specific attribute can whiff (0 constraints returned) and retire
itself, while ``other`` never does. So the sentence is pure product realism -
instead of repeating "Is there anything else that matters for this one?" every
turn, the agent names a facet the live candidate pool is genuinely split on:

    "For the material, I'm seeing both leather and canvas - do you have a
     preference?"

while ``ask_attribute`` is untouched.

Deterministic and template-based on purpose: the facet vocabularies
(``src/facets.py`` VOCABULARIES) are small enough that "which facet, top few
values" is enumerable, exactly like the rest of the pipeline. An LLM could later
replace ``_grounded`` for fluency with this as its fallback; ``ask_attribute``
and the evaluator score stay put either way.
"""

from __future__ import annotations

import math

from src.facets import FacetStore, weighted_value_counts
from src.policy import QUESTION_TEXT
from src.state import DialogState


#: Facets worth voicing: those extracted from a fixed keyword vocabulary
#: (``src/facets.py`` VOCABULARIES), so the values read as words. ``brand`` is
#: excluded (thousands of values - a 3-value summary covers almost none of the
#: pool); ``budget`` too (79% of the catalog has no price); ``category`` too
#: (its values are category-path fragments like "women" / "novelty", and the
#: opening already names the category).
VOICEABLE = ("material", "color", "style", "size", "use_case")

LEAD = {
    "material": "For the material",
    "color": "On colour",
    "size": "For sizing",
    "style": "Style-wise",
    "use_case": "For how you'll use it",
}

#: Rotated by ``state.turn_count`` so a session does not read identically each
#: turn. None start with "I" - the router tone-prefix lowercases the first char.
GROUNDED_TEMPLATES = (
    "{lead}, I'm seeing {vals} - do you have a preference?",
    "{lead}, the pool is split across {vals}. Does one matter more to you?",
    "There's a mix of {vals} here - {lead_lc}, does one stand out?",
)

BROAD_BANK = (
    "Is there anything else that matters for this one?",
    "Anything else you'd want me to factor in?",
    "What else is important for this?",
    "Is there another detail that would help narrow it down?",
)

# --- selection thresholds for _grounded ---
_MIN_COVERAGE = 0.35      # this share of the pool must resolve the facet
_MAX_TOP_SHARE = 0.85     # the top value must hold <= this of the resolved mass
_MIN_VALUE_SHARE = 0.05   # a value needs >= this of resolved mass to count as present


def _gain_ratio(distribution: dict[str, float]) -> float:
    """Entropy normalised to [0, 1] by the max entropy of the split.

    0 = the pool agrees on one value, 1 = perfectly even. Same measure as
    ``InfoGainPolicy._gain_ratio``.
    """
    mass = sum(distribution.values())
    if mass <= 0.0 or len(distribution) < 2:
        return 0.0
    entropy = 0.0
    for weight in distribution.values():
        probability = weight / mass
        if probability > 0.0:
            entropy -= probability * math.log2(probability)
    return entropy / math.log2(len(distribution))


def _join(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + " and " + values[-1]


def _tone(route: object, text: str) -> str:
    """Prepend the router tone exactly as ``starter/agent.py`` used to."""
    if route is None:
        return text
    return route.tone + text[0].lower() + text[1:]


def _broad(state: DialogState, attribute: str) -> str:
    if attribute in ("other", "feature"):
        return BROAD_BANK[state.turn_count % len(BROAD_BANK)]
    return QUESTION_TEXT.get(attribute, QUESTION_TEXT["other"])


def _grounded(
    state: DialogState,
    candidates: list[tuple[str, float]],
    store: FacetStore,
    depth: int,
) -> str | None:
    """A question about the facet the live pool is most split on, or None."""
    if not candidates:
        return None
    counts, total = weighted_value_counts(candidates, store, depth, VOICEABLE)
    if total <= 0.0:
        return None

    best_attr: str | None = None
    best_ratio = 0.0
    best_values: list[str] = []
    for attribute in VOICEABLE:
        if attribute in state.dead_attributes or attribute in state.asked:
            continue
        distribution = counts.get(attribute) or {}
        resolved = sum(distribution.values())
        if resolved <= 0.0 or resolved / total < _MIN_COVERAGE:
            continue
        ordered = sorted(distribution.items(), key=lambda item: -item[1])
        if ordered[0][1] / resolved > _MAX_TOP_SHARE:
            continue
        present = [value for value, mass in ordered if mass / resolved >= _MIN_VALUE_SHARE]
        if len(present) < 2:
            continue
        ratio = _gain_ratio(distribution)
        if ratio > best_ratio:
            best_attr, best_ratio, best_values = attribute, ratio, present[:3]

    if best_attr is None:
        return None
    lead = LEAD[best_attr]
    template = GROUNDED_TEMPLATES[state.turn_count % len(GROUNDED_TEMPLATES)]
    return template.format(
        lead=lead,
        lead_lc=lead[0].lower() + lead[1:],
        vals=_join(best_values),
    )


def clarify(
    attribute: str,
    state: DialogState,
    candidates: list[tuple[str, float]],
    store: FacetStore,
    route: object,
    config: object,
) -> str:
    """Build the customer-facing ``message`` for this turn.

    ``attribute`` (``ask_attribute``) is decided upstream and never changed here.
    With ``config.natural_questions`` off this reproduces the previous
    fixed-string behaviour byte for byte.
    """
    if not getattr(config, "natural_questions", False):
        return _tone(route, QUESTION_TEXT.get(attribute, QUESTION_TEXT["other"]))

    try:
        first = max(1, int(getattr(config, "first_recommend_turn", 3)))
        if state.productive_turns >= 1 and state.turn_count >= first:
            grounded = _grounded(
                state, candidates, store, int(getattr(config, "phrasing_depth", 40))
            )
            if grounded:
                return _tone(route, grounded)
        return _tone(route, _broad(state, attribute))
    except Exception:
        # A phrasing bug must never cost a session - degrade to a good question,
        # not to the empty response the outer handler would produce.
        return _tone(route, _broad(state, "other"))
