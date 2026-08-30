"""S6 - reranking the candidate pool.

Retrieval already places the target in the pool almost always; what costs score
is its *position*, because MRR carries 30% of the technical score and the session
ends at the first hit, freezing whatever rank was achieved.

The dominant signal is verbatim span coverage. Constraints the customer discloses
are copied from the target product's own metadata, so a candidate whose text
literally contains "stainless steel band" is far more likely to be the target than
one that merely shares those tokens. Popularity is a tie-break only: the target is
one specific purchase, not a bestseller.

Weighting spans by pool-local rarity was implemented and measured on the theory
that "buckle closure" should count for less than "two row stitch" among belts. It
changed the dev score by 0.0002 and the holdout not at all, because a pool
retrieved by those same terms has little rarity spread left to exploit, so it was
removed rather than kept as a dead option.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.index import CatalogIndex
from src.state import DialogState
from src.facets import (
    extract,
    extract_query_facets,
)
from src.text import terms


@dataclass
class RerankConfig:
    enabled: bool = True
    span_weight: float = 1.0
    # Longer spans are rarer and therefore stronger evidence.
    length_bonus: float = 0.12
    retrieval_weight: float = 1.0
    popularity_weight: float = 0.02
    # Candidate facets matching the customer's stated facets (material, colour, ...).
    facet_weight: float = 0.3
    category_weight: float = 0.4
    # Title keyword overlap and store/brand exact match to maximize MRR.
    title_weight: float = 0.25
    store_weight: float = 0.05

    # Rescore the whole retrieval pool (RetrievalConfig.pool_size), not a prefix -
    # ~12% of cluster-target sessions had the target in the pool but past rank 200,
    # where it was left in bm25 order and the span signal never applied.
    depth: int = 300


def _popularity(product: dict) -> float:
    """Small, bounded prior in [0, 1]. Tie-break only - see module docstring."""
    rating = product.get("average_rating") or 0.0
    count = product.get("rating_number") or 0
    try:
        return (float(rating) / 5.0) * min(1.0, math.log10(float(count) + 1.0) / 4.0)
    except (TypeError, ValueError):
        return 0.0


def _facet_agreement(
    customer_text: str,
    product: dict,
) -> float:
    """
    Count matching facet values between
    customer constraints and product facets.
    """
    '''
    customer_facets = extract(
        {
            "text": customer_text,
            "categories": [],
            "store": "",
            "price": None,
        }
    )
    '''
    customer_facets = extract_query_facets(
        customer_text
    )
    product_facets = extract(product)

    score = 0.0

    for key, value in customer_facets.items():

        if product_facets.get(key) == value:
            score += 1.0

    return score


def _category_match(
    state: DialogState,
    product: dict,
) -> float:
    """
    Category agreement between
    opening query and candidate product.
    """

    opening_terms = set(
        terms(state.opening, drop_boilerplate=True)
    )

    categories = {
        str(value).lower()
        for value in product.get("categories", [])
    }

    if not opening_terms:
        return 0.0

    score = 0.0

    for category in categories:

        category_tokens = set(
            terms(category)
        )

        if opening_terms.intersection(category_tokens):
            score += 1.0

    return score


def rerank(
    index: CatalogIndex,
    state: DialogState,
    candidates: list[tuple[str, float]],
    config: RerankConfig | None = None,
) -> list[tuple[str, float]]:
    config = config or RerankConfig()
    if not config.enabled or not candidates:
        return candidates

    spans = state.query_spans()
    head = candidates[: config.depth]
    tail = candidates[config.depth :]
    if not spans:
        return candidates

    # Normalise retrieval scores so the two signals combine on one scale.
    top_score = max(score for _asin, score in head) or 1.0
    opening_terms = set(terms(state.opening, drop_boilerplate=True)) if config.title_weight > 0 else set()

    scored: list[tuple[str, float]] = []
    for parent_asin, retrieval_score in head:
        product = index.products.get(parent_asin)
        if product is None:
            scored.append((parent_asin, 0.0))
            continue
        text = product["text"]
        title_str = (product.get("title") or "").lower()
        store_str = (product.get("store") or "").lower()

        coverage = 0.0
        for span in spans:
            if span in text:
                coverage += 1.0 + config.length_bonus * len(span.split())
                if config.title_weight > 0 and span in title_str:
                    coverage += config.title_weight

        title_overlap_score = 0.0
        if config.title_weight > 0 and opening_terms:
            title_toks = set(terms(title_str, drop_boilerplate=True))
            overlap = opening_terms.intersection(title_toks)
            if overlap:
                title_overlap_score = len(overlap) / (len(opening_terms) + 1.0)

        store_score = 0.0
        if config.store_weight > 0 and store_str and store_str in state.full_text().lower():
            store_score = config.store_weight

        facet_score = _facet_agreement(
            state.full_text(),
            product,
        )
        category_score = _category_match(
            state,
            product,
        )
        total = (
            config.span_weight * coverage
            + config.retrieval_weight * (retrieval_score / top_score)
            + config.popularity_weight * _popularity(product)
            + config.facet_weight * facet_score
            + config.category_weight * category_score
            + config.title_weight * title_overlap_score
            + store_score
        )
        scored.append((parent_asin, total))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored + tail
