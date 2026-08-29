"""Metadata-scaffolding stoplist learned from the frozen catalog.

GENERATED FILE - do not edit by hand.
Regenerate with: python3 tools/build_stoplist.py

Source        data/catalog.jsonl
Products      50,000
Distinct      101,064 tokens
Rule          document frequency >= 5% of products
              and >= 90% of occurrences inside the `details` field

See tools/build_stoplist.py for why frequency alone is the wrong signal, and
src/text.py for the hand-written remainder that this rule cannot reach.
"""

from __future__ import annotations

SCAFFOLDING = frozenset({
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
    "april",
    "august",
    "available",
    "date",
    "december",
    "department",
    "dimensions",
    "discontinued",
    "february",
    "first",
    "inches",
    "item",
    "january",
    "july",
    "june",
    "manufacturer",
    "march",
    "mens",
    "model",
    "november",
    "number",
    "october",
    "ounces",
    "package",
    "pounds",
    "september",
    "womens",
})
