"""Typed attribute extraction over catalog products.

Shared by the clarification policy (S4), which needs to know how a candidate pool
partitions along each attribute, and by the reranker (S6) for agreement scoring.

Extraction is regex/keyword based and imperfect by design: the goal is a usable
partition of the candidate pool, not a correct product taxonomy. Coverage is
reported per attribute so the policy can discount attributes it cannot resolve.
"""

from __future__ import annotations

import re

# Attribute vocabularies. Ordered - the first match wins, so put the more
# specific term first where two could both match.
VOCABULARIES: dict[str, tuple[str, ...]] = {
    "material": (
        "leather", "cotton", "polyester", "nylon", "wool", "spandex", "silk",
        "rayon", "denim", "linen", "cashmere", "suede", "velvet", "satin",
        "mesh", "fleece", "acrylic", "bamboo", "stainless steel", "sterling silver",
        "gold plated", "alloy", "titanium", "rubber", "canvas",
    ),
    "color": (
        "black", "white", "blue", "red", "pink", "green", "brown", "gray",
        "grey", "purple", "yellow", "orange", "beige", "navy", "silver",
        "gold", "ivory", "burgundy", "teal", "khaki",
    ),
    "style": (
        "casual", "formal", "vintage", "classic", "modern", "bohemian",
        "athletic", "sporty", "elegant", "minimalist", "long sleeve",
        "short sleeve", "sleeveless", "crew neck", "v neck", "slim fit",
        "relaxed fit", "regular fit", "high waisted", "oversized",
    ),
    "use_case": (
        "hiking", "running", "gym", "workout", "yoga", "travel", "work",
        "office", "wedding", "party", "beach", "swimming", "winter",
        "summer", "outdoor", "everyday", "sleep", "maternity",
    ),
    "size": (
        "small", "medium", "large", "x large", "xx large", "petite", "plus size",
        "wide", "narrow", "one size", "adjustable",
    ),
}

_PATTERNS = {
    attribute: re.compile(r"\b(" + "|".join(re.escape(term) for term in terms) + r")\b")
    for attribute, terms in VOCABULARIES.items()
}

# What a profile's preference_tags mean in attribute terms. Shared by the
# clarification policy (answerability priors) and the reranker (profile-weighted
# facet agreement). Identity entries let a tag name an attribute directly.
TAG_HINTS = {
    "fit": "size", "comfort": "material", "durability": "material",
    "quality": "material", "style": "style", "design": "style",
    "color": "color", "price": "budget", "value": "budget",
    "brand": "brand", "performance": "use_case",
    "material": "material", "size": "size", "use_case": "use_case",
    "category": "category", "budget": "budget",
}

# Buckets rather than raw prices: the exact figure is never what a customer states.
PRICE_BUCKETS = ((15.0, "under 15"), (30.0, "15 to 30"), (60.0, "30 to 60"),
                 (120.0, "60 to 120"), (float("inf"), "over 120"))

EXCLUDED_CATEGORIES = {
    "clothing", "shoes", "jewelry", "clothing shoes & jewelry",
    "clothing, shoes & jewelry",
}


def _price_bucket(price: object) -> str | None:
    try:
        value = float(price)
    except (TypeError, ValueError):
        return None
    for ceiling, label in PRICE_BUCKETS:
        if value < ceiling:
            return label
    return None


def _category_leaf(categories: list[str]) -> str | None:
    for value in reversed(categories or []):
        cleaned = str(value).strip().lower()
        if cleaned and cleaned not in EXCLUDED_CATEGORIES:
            return cleaned
    return None


def extract(product: dict) -> dict[str, str]:
    """Return the attribute values resolvable for one product.

    Missing attributes are simply absent, which is what lets the policy measure
    coverage and avoid asking questions the catalog cannot answer.
    """
    text = product.get("text", "")
    values: dict[str, str] = {}
    for attribute, pattern in _PATTERNS.items():
        match = pattern.search(text)
        if match:
            values[attribute] = match.group(1)

    brand = (product.get("store") or "").strip().lower()
    if brand:
        values["brand"] = brand

    bucket = _price_bucket(product.get("price"))
    if bucket:
        values["budget"] = bucket

    leaf = _category_leaf(product.get("categories") or [])
    if leaf:
        values["category"] = leaf

    return values


def extract_query_facets(text: str) -> dict[str, str]:
    """
    Extract shopper constraints from a query.

    Unlike extract(), this is designed for
    customer utterances rather than catalog products.

    Example:

    "black leather belt"

    ->
    {
        "color": "black",
        "material": "leather"
    }
    """

    text = (text or "").lower()

    values: dict[str, str] = {}

    for attribute, pattern in _PATTERNS.items():
        match = pattern.search(text)

        if match:
            values[attribute] = match.group(1)

    return values


class FacetStore:
    """Lazily extracted, memoised facets keyed by ``parent_asin``."""

    def __init__(self, products: dict[str, dict]) -> None:
        self._products = products
        self._cache: dict[str, dict[str, str]] = {}

    def get(self, parent_asin: str) -> dict[str, str]:
        values = self._cache.get(parent_asin)
        if values is None:
            product = self._products.get(parent_asin)
            values = extract(product) if product else {}
            self._cache[parent_asin] = values
        return values
