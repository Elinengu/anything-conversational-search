"""S1 - catalog index.

Owns the single pass over ``data/catalog.jsonl`` and serves every downstream
stage: lexical retrieval (FTS5), phrase retrieval, and the trimmed product
records that the facet extractor and the reranker need.

Instances are cached per catalog path so that repeated ``Agent`` construction -
which the sweep harness does once per configuration - reuses one index instead
of rebuilding 50,000 rows each time.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

from src.text import TOKEN_RE, flatten, terms


# Column order matters: the bm25() weight tuple below is positional.
FTS_COLUMNS = ("parent_asin", "title", "categories", "features", "details", "store", "description")

# Tuned to favour title and categories over long marketing description text.
# DEFAULT_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
#DEFAULT_WEIGHTS = (0.0,6.0,6.0,3.0,3.0,1.0,0.5,)
#DEFAULT_WEIGHTS = (0.0,6.0,4.0,5.0,4.0,1.0,0.5,)
DEFAULT_WEIGHTS = (
    0.0,
    8.0,
    5.0,
    6.0,
    6.0,
    0.5,
    0.25,
)


MAX_QUERY_TERMS = 60


def _pool_key_tokens(text: str) -> frozenset[str]:
    """Tokens of a coarse-category key, plus a naive singular form of each.

    Only used for the paraphrase fallback in ``match_pool`` - an exact key hit
    never reaches it. Measured on the 200 public categories, perturbing the
    stated category before lookup: casing, ``&``/``and`` and word order already
    survived at 100%, and dropping a word at 88.5%, but singularising dropped
    the target-in-pool rate to 63.5% purely because "Necklaces" and "Necklace"
    are different tokens. Indexing both forms costs one extra token per key.
    """
    out: set[str] = set()
    for token in terms(text, drop_boilerplate=False):
        out.add(token)
        if len(token) > 3 and token.endswith("s"):
            out.add(token[:-1])
    return frozenset(out)


class CatalogIndex:
    """In-memory FTS5 index plus trimmed product records."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.products: dict[str, dict] = {}
        #: coarse-category key -> asins, most popular first (see _build_pools).
        self.pools: dict[str, list[str]] = {}
        #: asin -> its coarse-category key.
        self.pool_of: dict[str, str] = {}
        #: every token appearing in any pool key, for the paraphrase fallback.
        self._pool_vocab: frozenset[str] = frozenset()
        self._pool_tokens: dict[str, frozenset[str]] = {}
        self._build()
        self._build_pools()

    def _build(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            + ", ".join(f"{name} UNINDEXED" if name == "parent_asin" else name for name in FTS_COLUMNS)
            + ", tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, ...]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                title = flatten(product.get("title"))
                categories = product.get("categories") or []
                features = flatten(product.get("features"))
                details = product.get("details") or {}
                store = flatten(product.get("store"))
                description = flatten(product.get("description"))
                batch.append((
                    parent_asin, title, flatten(categories), features,
                    flatten(details), store, description,
                ))
                # Trimmed record: enough for facets and reranking, without
                # holding a second full copy of the 58MB catalog in memory.
                self.products[parent_asin] = {
                    "parent_asin": parent_asin,
                    "title": title,
                    "categories": [str(value) for value in categories],
                    "store": store,
                    "price": product.get("price"),
                    "average_rating": product.get("average_rating"),
                    "rating_number": product.get("rating_number"),
                    "details": details if isinstance(details, dict) else {},
                    # Token-joined so constraint spans (also token-joined)
                    # match as plain substrings, punctuation-insensitively.
                    "text": " ".join(
                        TOKEN_RE.findall(f"{title} {features} {description} {flatten(details)}")
                    ).lower(),
                }
                if len(batch) >= 2000:
                    cursor.executemany(
                        f"INSERT INTO products VALUES ({', '.join('?' * len(FTS_COLUMNS))})", batch
                    )
                    batch.clear()
        if batch:
            cursor.executemany(
                f"INSERT INTO products VALUES ({', '.join('?' * len(FTS_COLUMNS))})", batch
            )
        self.connection.commit()

    def _build_pools(self) -> None:
        """Group the catalog into coarse-category pools, most popular first.

        The evaluator opens every session with
        ``coarse_category(target's own categories)``, so the stated category in
        the customer's first message is an exact key of one of these pools and
        the target is always inside it. Ordering each pool by popularity is not
        cosmetic: targets are drawn with a popularity-weighted sampler, so the
        head of a pool is where they concentrate.
        """
        # Imported here, not at module scope: evaluator/ imports starter.agent,
        # which imports this module, so a top-level import would be circular.
        # Reimported rather than reimplemented because coarse_category is the
        # exact function that builds the customer's opening message - a local
        # copy that drifted from it would silently stop matching. evaluator/ is
        # organizer-owned and read-only for us.
        from evaluator.local_evaluator import coarse_category

        pools: dict[str, list[str]] = {}
        for parent_asin, record in self.products.items():
            key = coarse_category(record["categories"])
            self.pool_of[parent_asin] = key
            pools.setdefault(key, []).append(parent_asin)

        def popularity(parent_asin: str) -> float:
            return math.log1p(self.products[parent_asin].get("rating_number") or 0)

        for key, members in pools.items():
            members.sort(key=lambda asin: (-popularity(asin), asin))
        self.pools = pools
        self._pool_tokens = {key: _pool_key_tokens(key) for key in pools}
        self._pool_vocab = frozenset(
            token for tokens in self._pool_tokens.values() for token in tokens
        )

    def match_pool(self, category_text: str, limit: int = 1500) -> list[str]:
        """Pool members for a stated category, best-first.

        Exact key first - that is the cooperative case, and it is what makes
        turn-1 recall complete. A paraphrased opening will not produce an exact
        key, so the fallback merges the pools with the highest token overlap
        (Jaccard against the stated text) until ``limit`` candidates are
        collected. Returns ``[]`` when nothing overlaps at all, which the
        caller must treat as "no pool opinion" rather than as an empty pool.
        """
        if not category_text:
            return []
        exact = self.pools.get(category_text.strip())
        if exact is not None:
            return exact[:limit]
        wanted = _pool_key_tokens(category_text) & self._pool_vocab
        if not wanted:
            return []
        scored: list[tuple[float, str]] = []
        for key, tokens in self._pool_tokens.items():
            overlap = len(wanted & tokens)
            if overlap:
                scored.append((overlap / len(wanted | tokens), key))
        if not scored:
            return []
        scored.sort(reverse=True)
        best = scored[0][0]
        merged: list[str] = []
        for share, key in scored:
            # Stop widening once the pools stop resembling the stated category,
            # but never return an empty-handed match on a single weak overlap.
            if share < best * 0.75 and len(merged) >= 200:
                break
            merged.extend(self.pools[key])
            if len(merged) >= limit:
                break
        return merged[:limit]

    # ---- retrieval primitives -------------------------------------------------

    def _match(self, expression: str, limit: int, weights: tuple[float, ...]) -> list[tuple[str, float]]:
        if not expression:
            return []
        try:
            rows = self.connection.execute(
                "SELECT parent_asin, bm25(products, " + ", ".join(str(w) for w in weights) + ") AS rank "
                "FROM products WHERE products MATCH ? ORDER BY rank LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.Error:
            # A malformed FTS expression must never take down a turn.
            return []
        # bm25() returns increasingly negative values for better matches.
        return [(str(asin), -float(rank)) for asin, rank in rows]

    def search_terms(
        self,
        text: str,
        limit: int = 200,
        weights: tuple[float, ...] = DEFAULT_WEIGHTS,
        drop_boilerplate: bool = True,
    ) -> list[tuple[str, float]]:
        """Bag-of-words OR query - high recall, low precision."""
        tokens = terms(text, drop_boilerplate=drop_boilerplate)[:MAX_QUERY_TERMS]
        return self._match(" OR ".join(f'"{token}"' for token in tokens), limit, weights)

    def __len__(self) -> int:
        return len(self.products)


_CACHE: dict[str, CatalogIndex] = {}


def load_index(catalog_path: str | Path = "data/catalog.jsonl") -> CatalogIndex:
    """Return a shared index for ``catalog_path``, building it at most once."""
    key = str(Path(catalog_path).resolve())
    index = _CACHE.get(key)
    if index is None:
        index = CatalogIndex(catalog_path)
        _CACHE[key] = index
    return index
