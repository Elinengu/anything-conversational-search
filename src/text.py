"""Shared text normalisation used by the index, the state tracker and the policy."""

from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Conversational filler plus the boilerplate that saturates Amazon feature bullets.
# Removing the latter matters because the simulated customer quotes product copy
# verbatim, so phrases like "imported" or "machine wash" would otherwise dominate
# the query while carrying almost no discriminative signal.
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

BOILERPLATE = {
    "imported", "closure", "machine", "wash", "care", "instructions",
    "made", "usa", "discontinued", "manufacturer", "date", "first",
    "available", "product", "dimensions", "inches", "ounces", "pounds",
    "department", "asin", "item", "model", "number", "package",
}


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
