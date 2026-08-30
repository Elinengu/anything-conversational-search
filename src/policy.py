"""S4 - clarification policy: which attribute to ask for next.

Two implementations share one interface so the sweep harness can compare them
directly and so the cheap one stays available as a guaranteed-safe fallback:

  * ``FixedPolicy``     - the measured floor. Always asks the broadest question.
  * ``InfoGainPolicy``  - selects the attribute that most reduces uncertainty
                          over the live candidate pool (built in a later stage).

Only these values are legal in the response contract
(``docs/agent_api_contract.json``).
"""

from __future__ import annotations

import math

from src.facets import TAG_HINTS, weighted_value_counts
from src.state import DialogState


ALLOWED_ATTRIBUTES = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other",
)

QUESTION_TEXT = {
    "category": "What kind of item are you shopping for exactly?",
    "material": "Is there a material you prefer?",
    "color": "Any colour you have in mind?",
    "size": "What size or fit are you after?",
    "style": "What style would suit you best?",
    "brand": "Do you have a brand you like?",
    "budget": "Roughly what budget are you working with?",
    "feature": "Which features matter most to you?",
    "use_case": "What will you mainly use it for?",
    "other": "Is there anything else that matters for this one?",
}


class FixedPolicy:
    """Ask the broadest available question every turn.

    This is the measured scoring floor (0.7811 on the public set). It is
    deliberately simple: it exists so that every later policy has a committed
    baseline to beat, and so there is always a working fallback.
    """

    name = "fixed"

    #: Tried in order once every attribute in ``sequence`` has been declined.
    #: Re-asking a question the customer has already refused wastes a turn and
    #: reads as not listening, so the policy always has somewhere else to go.
    FALLBACK = ("feature", "use_case", "style", "material", "color", "size", "category")

    def __init__(self, sequence: tuple[str, ...] = ("other",)) -> None:
        self.sequence = sequence

    def select(self, state: DialogState, candidates: list[tuple[str, float]]) -> str:
        for offset in range(len(self.sequence)):
            attribute = self.sequence[(state.turn_count - 1 + offset) % len(self.sequence)]
            if attribute not in state.dead_attributes:
                return attribute
        for attribute in self.FALLBACK:
            if attribute not in state.dead_attributes:
                return attribute
        return "other"

    def question(self, attribute: str) -> str:
        return QUESTION_TEXT.get(attribute, QUESTION_TEXT["other"])


class InfoGainPolicy:
    """Ask the question that most reduces uncertainty over the candidate pool.

    For a specific attribute ``a`` the expected value of asking is

        gain(a) = coverage(a) x H(a) / log2(distinct values of a)

    where ``H(a)`` is the entropy of ``a``'s value distribution across the live
    candidates (how much the answer would split the pool) and ``coverage(a)`` is
    the share of candidates for which ``a`` is even resolvable (how likely the
    question can be answered at all). The coverage term is what stops the agent
    asking about budget, which is unresolvable for 78.9% of this catalog.

    Dividing by ``log2(distinct values)`` makes this a *gain ratio* rather than raw
    information gain. Without it the measure is dominated by high-cardinality
    attributes - brand scored 6.1 bits against colour's 2.3 purely because the
    catalog holds thousands of distinct stores - and the agent would open every
    conversation by asking which brand the customer wants, which is both a poor
    question and one they can rarely answer.

    Coverage alone is still not enough, because it measures whether the *catalog*
    resolves an attribute, not whether a *shopper* can answer a question about it.
    Brand is the clearest case: every product has a store, so brand looks perfectly
    informative, yet few people can name the brand they want while still browsing.
    Each attribute therefore carries an answerability prior - a property of
    shoppers, not of this evaluator - which the customer's own ``preference_tags``
    adjust upward when the profile says they care about that dimension.

    Broad questions ("anything else that matters", "which features") are scored on
    the same scale rather than hard-coded. They can be answered as long as the
    customer still has something undisclosed, and they return whichever slot the
    customer considers most important - so their value is the mean gain across
    resolvable attributes, scaled by how many constraints a broad answer tends to
    surface, and decayed as broad answers stop being productive.

    The practical consequence is that broad questions win early, when little is
    known and most slots are open, and specific ones win later - which is also
    how a competent human shop assistant conducts the conversation.

    ``broad_yield`` is calibrated, not fitted: a broad answer surfaces roughly two
    constraints here against a specific attribute's fraction of one. Dev-split
    behaviour saturates for any value above about 5 (0.8530 at 5.0, 0.8651 at 8.0,
    0.8634 at 12.0 - all the same policy, differing only in noise), so the default
    sits mid-plateau rather than at the dev argmax.
    """

    name = "info_gain"

    #: Attributes with a value vocabulary we can actually partition the pool by.
    PARTITIONABLE = ("category", "material", "color", "style", "size", "brand", "budget", "use_case")

    #: Open-ended questions, scored against the mean gain of the specific ones.
    BROAD = ("feature", "other")

    #: P(a shopper can state a preference for this attribute). Domain priors, not
    #: fitted to the evaluator: people readily describe what an item is for or what
    #: it is made of, and rarely name an exact brand while still comparing options.
    ANSWERABILITY = {
        "feature": 0.90, "other": 0.85, "material": 0.70, "use_case": 0.65,
        "color": 0.60, "style": 0.60, "size": 0.50, "category": 0.40,
        "budget": 0.40, "brand": 0.20,
    }

    #: preference_tags naming these concepts raise the matching attribute's prior.
    #: Shared with the reranker's profile signal - the vocabulary lives in facets.
    TAG_HINTS = TAG_HINTS

    def __init__(
        self,
        facets,
        depth: int = 150,
        broad_yield: float = 6.0,
        expected_broad_answers: float = 2.0,
        tag_boost: float = 0.15,
        min_evidence: int = 1,
        allow_broad: bool = True,
    ) -> None:
        self.facets = facets
        self.depth = depth
        self.broad_yield = broad_yield
        self.expected_broad_answers = expected_broad_answers
        self.tag_boost = tag_boost
        self.min_evidence = min_evidence
        self.allow_broad = allow_broad

    # ---- scoring --------------------------------------------------------------

    def _distributions(self, candidates: list[tuple[str, float]]) -> tuple[dict[str, dict[str, float]], float]:
        """Score-weighted value counts per attribute over the candidate pool."""
        return weighted_value_counts(candidates, self.facets, self.depth, self.PARTITIONABLE)

    @staticmethod
    def _gain_ratio(distribution: dict[str, float]) -> float:
        """Entropy normalised to [0, 1] by the maximum entropy of that split."""
        mass = sum(distribution.values())
        if mass <= 0.0 or len(distribution) < 2:
            return 0.0
        entropy = 0.0
        for weight in distribution.values():
            probability = weight / mass
            if probability > 0.0:
                entropy -= probability * math.log2(probability)
        return entropy / math.log2(len(distribution))

    def _answerability(self, attribute: str, state: DialogState) -> float:
        """Prior probability the customer can answer, nudged by their profile."""
        prior = self.ANSWERABILITY.get(attribute, 0.5)
        tags = state.profile.get("preference_tags") or []
        for tag in tags:
            hinted = self.TAG_HINTS.get(str(tag).strip().lower())
            if hinted == attribute:
                prior = min(1.0, prior + self.tag_boost)
        return prior

    def scores(self, state: DialogState, candidates: list[tuple[str, float]]) -> dict[str, float]:
        """Expected information gain per legal question. Exposed for inspection."""
        if not candidates:
            return {"other": 1.0}
        counts, total = self._distributions(candidates)
        gains: dict[str, float] = {}
        for attribute in self.PARTITIONABLE:
            distribution = counts[attribute]
            coverage = sum(distribution.values()) / total if total else 0.0
            gains[attribute] = (
                coverage
                * self._gain_ratio(distribution)
                * self._answerability(attribute, state)
            )

        if self.allow_broad:
            covered = [value for value in gains.values() if value > 0.0]
            mean_gain = sum(covered) / len(covered) if covered else 0.0
            remaining = max(
                0.0,
                1.0 - state.productive_turns / max(self.expected_broad_answers, 1e-6),
            )
            for attribute in self.BROAD:
                gains[attribute] = (
                    remaining
                    * mean_gain
                    * self.broad_yield
                    * self._answerability(attribute, state)
                )

        # Never spend a turn on a question already asked or explicitly declined.
        for attribute in list(gains):
            if attribute in state.dead_attributes or attribute in state.asked:
                gains[attribute] = 0.0
        return gains

    def _last_ask_was_broad(self, state: DialogState) -> bool:
        return bool(state.asked) and state.asked[-1] in self.BROAD

    # ---- interface ------------------------------------------------------------
    #
    # ``select`` only ever returns an ``ALLOWED_ATTRIBUTES`` value - it is what
    # the simulator reads as ``ask_attribute`` and the response contract requires
    # it. Rewording *how* the question is asked (an optional DeepSeek polish layer)
    # happens downstream in ``src/phrasing.py::clarify``, never here - that keeps
    # this policy deterministic and keeps a flaky LLM call from ever being able
    # to turn a valid attribute into an invalid one.
    #
    # ``feature/gemini-infrastructure`` proposed a second, LLM-backed wording
    # layer inside this class (``_llm_question_hint`` / an LLM-aware
    # ``question()``). It duplicates the polish layer that already lives in
    # ``src/phrasing.py::clarify`` (grounded clarification generation, one of
    # ``LLMClient``'s three purpose-built features) and nothing in the live
    # code path ever calls ``InfoGainPolicy.question()`` - so it was dropped
    # here rather than merged, to keep one polish layer instead of two.

    def select(self, state: DialogState, candidates: list[tuple[str, float]]) -> str:
        # Two reasons to ask broadly, both evidence-driven rather than prior-driven:
        #
        #  1. Before anything is disclosed the candidate pool is still essentially
        #     the catalog prior. Its attribute distributions describe the inventory,
        #     not this shopper, so the gain estimate is not yet meaningful.
        #  2. While broad questions keep surfacing new constraints they remain the
        #     best available question, because they return whichever slot the
        #     customer thinks matters rather than one we guessed at. Only when a
        #     broad answer stops yielding is it worth spending turns on specific
        #     attributes.
        #
        # Condition 2 is observed, not assumed: it reads what the last answer
        # actually produced.
        if self.allow_broad and (
            state.productive_turns < self.min_evidence
            or (state.last_turn_productive and self._last_ask_was_broad(state))
        ):
            for attribute in self.BROAD:
                if attribute not in state.dead_attributes:
                    return attribute

        gains = self.scores(state, candidates)
        attribute, best = max(gains.items(), key=lambda item: (item[1], item[0]))
        if best <= 0.0:
            # Everything scored has been asked or is unanswerable; fall back to the
            # most open-ended question rather than repeating a dead one.
            for candidate in ("feature", "other", "style"):
                if candidate not in state.dead_attributes:
                    return candidate
            return "other"
        return attribute

    def question(self, attribute: str) -> str:
        return QUESTION_TEXT.get(attribute, QUESTION_TEXT["other"])
