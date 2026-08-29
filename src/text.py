"""Shared text normalisation used by the index, the state tracker and the policy."""

from __future__ import annotations

import re

# Amazon's structural metadata: "Package Dimensions", "Date First Available",
# "Item model number", "Department". Learned from the catalog rather than written
# by hand - tools/build_stoplist.py carries the rule and the evidence. Deriving it
# found every month name and year 2014-2022, which come from
# "Date First Available: August 15, 2019" and which nobody thought to write down.
#
# Note the wrong way to do this: dropping the most frequent catalog terms would
# delete polyester (frequency rank 72), cotton (83), black (97), leather (111) and
# spandex (141) - precisely the constraints the simulated customer discloses.
from src.stoplist import SCAFFOLDING


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Conversational filler: the simulator's own framing, which carries no product
# signal at all.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "have", "has", "not", "no", "do", "dont", "am", "your", "our", "we",
    "prefer", "preference", "additional", "matters", "judgment", "use",
    "key", "requirement", "actually", "ignore", "earlier", "need", "what",
    "quite", "right", "yet", "ask", "about", "one", "specific", "attribute",
    "options", "those", "still", "exploring", "there",
}

# The remainder of the boilerplate, kept by hand on purpose.
#
# These are care-and-origin phrases in features/description, not metadata, and no
# catalog statistic separates them from real attribute words. Two were measured
# and both fail:
#
#   * by document frequency, "closure" (38.6% of products) outranks
#     "polyester" (21.8%), so any frequency cut that catches one deletes the other;
#   * by spread across the 12 largest category buckets, "polyester" (CV 0.51)
#     falls between "imported" (0.49) and "wash" (0.60), and "black"/"white"
#     (0.34/0.31) land inside the scaffolding band.
#
# What actually separates them is that a shopper does not choose between
# "imported" and "not imported" - semantics, not frequency. Hand-written is the
# honest answer here; please do not retry the experiment without new evidence.
CARE_PHRASES = {
    "imported", "closure", "machine", "wash", "care", "instructions",
    "made", "usa", "product", "asin",
}

BOILERPLATE = SCAFFOLDING | CARE_PHRASES


def flatten(value: object) -> str:
    """Render any catalog field (string, list or dict) as searchable text."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def terms(text: str, drop_boilerplate: bool = False) -> list[str]:
    """Lowercase content tokens, in first-seen order, duplicates removed."""
    dropped = STOPWORDS | BOILERPLATE if drop_boilerplate else STOPWORDS
    out: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(text):
        lowered = token.lower()
        if len(lowered) < 2 or lowered in dropped or lowered in seen:
            continue
        seen.add(lowered)
        out.append(lowered)
    return out


def constraint_spans(text: str, min_words: int = 2) -> list[str]:
    """Normalised multi-word fragments from one customer message.

    The simulated customer discloses constraints copied verbatim from the target
    product's metadata, so a fragment like "stainless steel band" is very nearly
    exact catalog text. These are matched as substrings against product text in
    the reranker rather than as FTS5 phrases: stripping stopwords for an FTS
    phrase query breaks token adjacency and the phrase then matches nothing.
    """
    out: list[str] = []
    seen: set[str] = set()
    # Fragment separators mirror how constraints are joined in a single message.
    for chunk in re.split(r"[.;:,\n]| - ", text):
        span = " ".join(TOKEN_RE.findall(chunk)).lower().strip()
        if not span or span in seen:
            continue
        words = span.split()
        if len(words) < min_words or len(words) > 25:
            continue
        # A fragment made only of filler carries no signal.
        if all(word in STOPWORDS for word in words):
            continue
        seen.add(span)
        out.append(span)
    return out
