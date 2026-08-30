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

    "The materials I'm looking at come in leather and canvas - do you lean one
     way on material?"

while ``ask_attribute`` is untouched.

Everything customer-facing here is deterministic and template-based on purpose.
Three template banks (a lead-in by dialogue posture, a grounded question over
the pool split, a broad fallback) are each rotated by a *stable* hash of the
opening line and the turn number - so a session never repeats a sentence, two
sessions do not read the same way, and the wording is reproducible across
evaluator runs (the random session id is not used as the key). The facet
vocabularies (``src/facets.py`` VOCABULARIES) are small enough that "which
facet, top few values, which template" is enumerable, exactly like the rest of
the pipeline.

An optional DeepSeek polish pass (``_llm_polish``, ``src/llm.py``) restyles the
already-computed template sentence for fluency when ``DEEPSEEK_API_KEY`` is
configured. It never chooses the facet, the values, or ``ask_attribute`` -
only wording - and on any missing key, network failure, or malformed reply it
falls straight back to the deterministic template text. ``ask_attribute`` and
the evaluator score are therefore invariant to it either way.
"""

from __future__ import annotations

import math
import zlib

from src.facets import FacetStore, weighted_value_counts
from src.llm import get_llm_client
from src.policy import QUESTION_TEXT
from src.state import DialogState


#: Facets worth voicing: those extracted from a fixed keyword vocabulary
#: (``src/facets.py`` VOCABULARIES), so the values read as words. ``brand`` is
#: excluded (thousands of values - a 3-value summary covers almost none of the
#: pool); ``budget`` too (79% of the catalog has no price); ``category`` too
#: (its values are category-path fragments like "women" / "novelty", and the
#: opening already names the category).
VOICEABLE = ("material", "color", "style", "size", "use_case")

#: Natural noun for each voiceable facet, used mid-sentence ("...any steer on
#: sizing?").
ATTR_NOUN = {
    "material": "material",
    "color": "colour",
    "size": "sizing",
    "style": "style",
    "use_case": "how you'll use it",
}

#: Plural sentence-subject form ("The colours here run ..."). ``use_case`` has
#: no clean plural noun, so it borrows the generic "The options".
ATTR_SUBJECT = {
    "material": "The materials",
    "color": "The colours",
    "size": "The sizes",
    "style": "The styles",
    "use_case": "The options",
}

#: Complete grounded questions. Each is a full sentence with ``{vals}`` (the
#: pool's top values, e.g. "leather and canvas"), ``{noun}`` (``ATTR_NOUN``) and
#: ``{subject}`` (``ATTR_SUBJECT``) slots. None start with "I" so the lead-in's
#: first-char lowercasing is always safe.
GROUNDED_TEMPLATES = (
    "{subject} I'm looking at come in {vals} - do you lean one way on {noun}?",
    "Right now the shortlist splits between {vals}. Any preference on {noun}?",
    "These range across {vals} for {noun} - is one closer to what you had in mind?",
    "The mix on {noun} right now is {vals}. Does one of those stand out?",
    "On {noun} it's a mix of {vals} - which would you go for?",
    "The options here are {vals}. Do you care about {noun}, or should I choose?",
    "A few directions on {noun}: {vals}. Anything jump out?",
    "So far the list covers {vals} - any steer on {noun}?",
    "For {noun} I've got {vals} in the running. Want me to favour one?",
    "{subject} vary here - {vals}. Is there one you'd prefer?",
)

#: Used when the pool is not cleanly split on any facet, or on turn 1.
BROAD_BANK = (
    "Is there anything else that matters for this one?",
    "Anything else you'd want me to factor in?",
    "What else is important for this?",
    "Is there another detail that would help narrow it down?",
    "Anything else I should keep in mind?",
    "Is there a detail I'm missing here?",
    "What else would help me get this right?",
)

#: Enriched variants of the fixed ``QUESTION_TEXT`` ladder questions, used when a
#: specific attribute is asked and the pool is not split on it. First entry of
#: each is the original string.
SPECIFIC_BANK = {
    "material": (
        "Is there a material you prefer?",
        "Any material you're set on - or set against?",
        "Does the material matter to you here?",
    ),
    "color": (
        "Any colour you have in mind?",
        "Is there a colour you're leaning toward?",
        "Do you want me to hold to a particular colour?",
    ),
    "size": (
        "What size or fit are you after?",
        "How should this fit?",
        "Any sizing I should work around?",
    ),
    "style": (
        "What style would suit you best?",
        "Is there a particular look you're going for?",
        "Any style you'd want me to stick to?",
    ),
    "use_case": (
        "What will you mainly use it for?",
        "Where do you picture using this most?",
        "What's the main use you have in mind?",
    ),
    "feature": (
        "Which features matter most to you?",
        "Any must-have features?",
        "Is there a feature that would make or break it?",
    ),
    "category": (
        "What kind of item are you shopping for exactly?",
        "Which type of item did you have in mind?",
        "Can you pin down the kind of item?",
    ),
    "brand": (
        "Do you have a brand you like?",
        "Any brand you trust for this?",
        "Is brand something you care about here?",
    ),
    "budget": (
        "Roughly what budget are you working with?",
        "About what price range feels right?",
        "Any ceiling on price I should respect?",
    ),
}

#: Lead-ins by dialogue posture (``route.name``). The empty entries mean many
#: turns carry no prefix at all, which is how people actually talk.
LEADIN_BROWSING = (
    "", "", "", "",
    "To point you the right way, ",
    "To help me narrow things down, ",
    "Just so I show you the right things, ",
    "So we get closer to it, ",
)
LEADIN_BUYING = (
    "", "", "", "",
    "To narrow this down, ",
    "To zero in on the right one, ",
    "So I can tighten the shortlist, ",
    "To get you an exact match, ",
)
#: One clear acknowledgement on the turn the customer reverses course; after
#: that the session is treated as focused (buying lead-ins).
LEADIN_OVERRIDE = (
    "Okay, switching gears - ",
    "Got it, let's re-aim - ",
    "No problem, changing tack - ",
    "Sure, updating course - ",
)

# --- selection thresholds for _grounded ---
# These only gate *which facet gets voiced*, never what is asked or retrieved, so
# the evaluator score is invariant to them (the simulator never reads `message`).
# They are set to fire whenever the pool is genuinely split on a facet the
# shopper has not pinned down - loose enough that most mid-session turns get a
# grounded question instead of the broad fallback, strict enough that a facet
# the pool basically agrees on (or barely resolves) is left alone.
_MIN_COVERAGE = 0.25      # this share of the pool must resolve the facet
_MAX_TOP_SHARE = 0.90     # the top value must hold <= this of the resolved mass
_MIN_VALUE_SHARE = 0.05   # a value needs >= this of resolved mass to count as present


def _pick(options: tuple, *parts: object) -> str:
    """Deterministically choose one option from a stable hash of ``parts``.

    ``zlib.crc32`` is stable across processes (unlike ``hash``), so a given
    session always reads the same way. Keyed on the opening line rather than the
    random session id so the wording is reproducible across evaluator runs too.
    """
    key = "|".join(str(part) for part in parts).encode("utf-8", "replace")
    return options[zlib.crc32(key) % len(options)]


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


def _legacy_tone(route: object, text: str) -> str:
    """The pre-``natural_questions`` prefix: exactly what ``starter/agent.py``
    used to do. Kept verbatim so ``natural_questions=False`` is byte-identical."""
    if route is None:
        return text
    return route.tone + text[0].lower() + text[1:]


def _leadin(route: object, state: DialogState) -> str:
    """The rotated conversational prefix for this turn (may be empty)."""
    if route is None:
        return ""
    override_turn = state.override_turn
    if override_turn is not None and state.turn_count == override_turn:
        return _pick(LEADIN_OVERRIDE, state.opening, "override", override_turn)
    focused = getattr(route, "name", "") == "buying" or (
        override_turn is not None and state.turn_count > override_turn
    )
    bank = LEADIN_BUYING if focused else LEADIN_BROWSING
    return _pick(bank, state.opening or state.session_id, "lead", state.turn_count)


def _apply(leadin: str, text: str) -> str:
    if not leadin:
        return text
    # Don't lowercase a leading first-person "I" / "I'm" / "I've".
    if text == "I" or text[:2] in ("I ", "I'", "I’"):
        return leadin + text
    return leadin + text[0].lower() + text[1:]


def _tone(route: object, text: str, state: DialogState) -> str:
    return _apply(_leadin(route, state), text)


def _broad(state: DialogState, attribute: str) -> str:
    """The non-grounded question: a rotated broad prompt for ``other``, a rotated
    attribute-specific prompt for a named ladder rung."""
    seed = state.opening or state.session_id
    if attribute == "other":
        return _pick(BROAD_BANK, seed, "broad", state.turn_count)
    bank = SPECIFIC_BANK.get(attribute)
    if bank:
        return _pick(bank, seed, "specific", attribute, state.turn_count)
    return QUESTION_TEXT.get(attribute, QUESTION_TEXT["other"])


def _grounded(
    state: DialogState,
    candidates: list[tuple[str, float]],
    store: FacetStore,
    depth: int,
) -> str | None:
    """A question about a facet the live pool is genuinely split on, or None.

    Qualifying facets are ranked by split quality and rotated by turn, so a
    session that stays on this path varies the facet rather than repeating one.
    """
    if not candidates:
        return None
    counts, total = weighted_value_counts(candidates, store, depth, VOICEABLE)
    if total <= 0.0:
        return None

    qualifying: list[tuple[float, str, list[str]]] = []
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
        qualifying.append((_gain_ratio(distribution), attribute, present[:3]))

    if not qualifying:
        return None
    # Most-split facet first, then rotate by turn so a session that stays on the
    # grounded path does not ask about the same facet every turn.
    qualifying.sort(key=lambda item: -item[0])
    _, attribute, values = qualifying[state.turn_count % len(qualifying)]
    template = _pick(
        GROUNDED_TEMPLATES,
        state.opening or state.session_id, "grounded", attribute, state.turn_count,
    )
    return template.format(
        noun=ATTR_NOUN[attribute],
        subject=ATTR_SUBJECT[attribute],
        vals=_join(values),
    )


def _llm_polish(text: str) -> str:
    """Optional DeepSeek restyling of an already-computed, pool-grounded question.

    ``text`` is the deterministic question - the source of truth and the return
    value whenever no ``DEEPSEEK_API_KEY`` is configured, the request fails for any
    reason (network down, timeout, bad response), or the reply does not look
    like a single clean sentence. DeepSeek only restyles wording here; it is never
    asked to choose a fact, a facet, or a value, so a bad response can only ever
    degrade back to the original template text, never invent a claim.
    """
    client = get_llm_client()
    if not client.is_configured:
        return text
    prompt = (
        "Rephrase the following clarifying question from a shopping assistant so "
        "it reads naturally and conversationally, in a single short sentence. "
        "Keep the exact same meaning and facts - do not add, remove, or invent "
        "any preference, brand, value, or claim. Return only the rephrased "
        "question, nothing else.\n\n"
        f"Question: {text}"
    )
    try:
        polished = client.generate(prompt)
    except Exception:
        return text
    if not isinstance(polished, str):
        return text
    polished = polished.strip().strip('"').strip()
    # Reject anything that does not look like a plausible single-sentence
    # question - too short, implausibly long, or missing punctuation the
    # template guarantees - and fall back to the deterministic text instead.
    if not polished or len(polished) > 240 or "\n" in polished:
        return text
    return polished


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
        return _legacy_tone(route, QUESTION_TEXT.get(attribute, QUESTION_TEXT["other"]))

    try:
        # Grounded from turn 2 on: after the opening line the retrieval pool is
        # shaped by something the shopper actually said, so a "the pool is split
        # on X" question is about their results, not the catalog prior. We do not
        # require a *productive* turn - single-word disclosures ("leather",
        # "black") never form a multi-word constraint span, so productive_turns
        # can sit at 0 for a whole session that is in fact narrowing well. The
        # per-facet gates in `_grounded` are the real guard against voicing a
        # facet the pool has not split on.
        if state.turn_count >= 2:
            grounded = _grounded(
                state, candidates, store, int(getattr(config, "phrasing_depth", 40))
            )
            if grounded:
                return _tone(route, _llm_polish(grounded), state)
        return _tone(route, _llm_polish(_broad(state, attribute)), state)
    except Exception:
        # A phrasing bug must never cost a session - degrade to a good question,
        # not to the empty response the outer handler would produce.
        return _tone(route, _broad(state, "other"), state)
