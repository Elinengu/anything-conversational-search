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
    # tools/stress_harness.py's paraphrase carrier vocabulary ("One more
    # thing - X.", "X matters to me as well.", "I'd also want it to be X.",
    # "honestly I just want X.", ...). The official evaluator's fixed
    # template ("For that, what matters is: X") never needs these - the
    # colon already isolates its carrier framing - so this addition is a
    # no-op there. See constraint_spans()'s bidirectional strip and
    # docs/team/agent_changes.md (the change after "browse-gated stall text
    # corrupting the active-slot ledger").
    "d", "also", "important", "should", "more", "thing", "well", "ideally",
    "s", "honestly", "just", "main", "oh", "if", "possible", "gotta",
    "leaning", "towards", "something", "great", "m", "after",
    "really", "gist", "plus", "bit", "kind",
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

    Fragment separators (colon/comma/etc.) cleanly isolate carrier framing from
    the value on the official evaluator's fixed template ("For that, what
    matters is: X"), but tools/stress_harness.py's paraphrased carrier
    sentences ("One more thing - X.", "X matters to me as well.") have no such
    separator between the framing and the value. Stripping a stopword run off
    *both* ends of each chunk (mirroring pair_spans' leading-only strip) drops
    the carrier and keeps the value; a chunk that is nothing but filler strips
    down to empty and is dropped by the blank-span check below.
    """
    out: list[str] = []
    seen: set[str] = set()
    # Fragment separators mirror how constraints are joined in a single message.
    for chunk in re.split(r"[.;:,\n]| - ", text):
        tokens = [token.lower() for token in TOKEN_RE.findall(chunk)]
        while tokens and tokens[0] in STOPWORDS:
            tokens.pop(0)
        while tokens and tokens[-1] in STOPWORDS:
            tokens.pop()
        span = " ".join(tokens)
        if not span or span in seen:
            continue
        if len(tokens) < min_words or len(tokens) > 25:
            continue
        seen.add(span)
        out.append(span)
    return out


def pair_spans(text: str, min_words: int = 3) -> list[str]:
    """Association-preserving fragments from one customer message.

    ``constraint_spans`` splits on colons and commas, which severs key:value
    structure: "Heather Grey: 90% Cotton, 10% Polyester" becomes three
    fragments that any product mentioning the parts in any arrangement can
    match. Splitting only on sentence separators keeps the association intact
    ("heather grey 90 cotton 10 polyester"), so a candidate must state that
    composition *about that colour* - catalog copy repeats these blocks
    verbatim, so the joined form is still an exact substring of the target.

    A leading run of stopwords is stripped: the simulator's framing sits in
    front of the first colon, and without the split it would glue itself to
    the first real pair ("for that what matters is color grey" -> "color
    grey").
    """
    out: list[str] = []
    seen: set[str] = set()
    for chunk in re.split(r"[.;\n]| - ", text):
        tokens = [token.lower() for token in TOKEN_RE.findall(chunk)]
        while tokens and tokens[0] in STOPWORDS:
            tokens.pop(0)
        span = " ".join(tokens)
        if not span or span in seen:
            continue
        if len(tokens) < min_words or len(tokens) > 25:
            continue
        seen.add(span)
        out.append(span)
    return out
