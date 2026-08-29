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


class CatalogIndex:
    """In-memory FTS5 index plus trimmed product records."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.products: dict[str, dict] = {}
        self._build()

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

    def search_phrases(
        self,
        spans: list[str],
        limit: int = 200,
        weights: tuple[float, ...] = DEFAULT_WEIGHTS,
    ) -> list[tuple[str, float]]:
        """Quoted multi-word phrase query - low recall, high precision."""
        quoted = [f'"{span}"' for span in spans if span.strip()]
        return self._match(" OR ".join(quoted), limit, weights)

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
