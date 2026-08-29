"""Learn the metadata-scaffolding stoplist from the frozen catalog.

Why this exists
---------------
``BOILERPLATE`` in ``src/text.py`` was hand-written. This generator replaces the
half of it that can be derived from evidence, and records why the other half
cannot.

The obvious approach - drop the most frequent catalog terms - is wrong, and
measurably so. Ranked by document frequency over ``data/catalog.jsonl``:
``polyester`` is 72nd, ``cotton`` 83rd, ``black`` 97th, ``leather`` 111th,
``spandex`` 141st, ``white`` 190th. Those are exactly the constraints the
simulated customer discloses - ``intent_card()`` inserts a material at position 0
and a colour at position 1 of every card. Meanwhile ``asin``, a genuine member of
the hand-written list, sits at rank 10,379 with 0.0% document frequency.
Frequency alone fails in both directions.

What separates scaffolding from signal is *where* a token occurs. Amazon's
structural metadata lives in the ``details`` dict, and those tokens appear almost
nowhere else:

    department 100.0%   dimensions 99.6%   manufacturer 99.6%   inches 97.7%

while attribute values are spread across title, features and description:

    spandex 1.2%        cotton 2.5%        polyester 3.3%       black 13.4%

The gap between 16% and 96% is empty of real attribute words, so the threshold is
read off the catalog's own distribution rather than tuned against a score. That
matters: tuning it against dev/holdout would fit the public sessions through the
back door. This generator never reads ``data/public_set.jsonl``.

Usage
-----
    python3 tools/build_stoplist.py --dry-run    # show the diff, write nothing
    python3 tools/build_stoplist.py              # regenerate src/stoplist.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.index import FTS_COLUMNS  # noqa: E402
from src.text import TOKEN_RE, flatten  # noqa: E402


# The fields the index actually searches, so the stoplist describes the same
# corpus the queries are matched against.
FIELDS = tuple(name for name in FTS_COLUMNS if name != "parent_asin")
STRUCTURAL_FIELD = "details"

# A token must be common enough to be scaffolding at all. Below this it is a
# product-specific term, whatever field it happens to live in.
MIN_DOCUMENT_FREQUENCY = 0.05
# ...and it must occur almost exclusively inside the structural metadata dict.
# Set inside the measured 16%-96% gap, not tuned against any score.
MIN_STRUCTURAL_CONCENTRATION = 0.90
MIN_TOKEN_LENGTH = 2

OUTPUT_PATH = _REPO_ROOT / "src" / "stoplist.py"

HEADER = '''"""Metadata-scaffolding stoplist learned from the frozen catalog.

GENERATED FILE - do not edit by hand.
Regenerate with: python3 tools/build_stoplist.py

Source        {catalog}
Products      {products:,}
Distinct      {tokens:,} tokens
Rule          document frequency >= {min_df:.0%} of products
              and >= {min_conc:.0%} of occurrences inside the `{field}` field

See tools/build_stoplist.py for why frequency alone is the wrong signal, and
src/text.py for the hand-written remainder that this rule cannot reach.
"""

from __future__ import annotations

SCAFFOLDING = frozenset({{
{terms}}})
'''


def measure(catalog_path: Path) -> tuple[int, dict[str, int], dict[str, int]]:
    """Return (product count, document frequency, structural-field frequency)."""
    document_frequency: collections.Counter[str] = collections.Counter()
    structural_frequency: collections.Counter[str] = collections.Counter()
    products = 0
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            products += 1
            per_field = {
                field: set(TOKEN_RE.findall(flatten(product.get(field)).lower()))
                for field in FIELDS
            }
            for token in per_field[STRUCTURAL_FIELD]:
                structural_frequency[token] += 1
            for token in set().union(*per_field.values()):
                document_frequency[token] += 1
    return products, document_frequency, structural_frequency


def select(
    products: int,
    document_frequency: dict[str, int],
    structural_frequency: dict[str, int],
) -> set[str]:
    return {
        token
        for token, count in document_frequency.items()
        if len(token) >= MIN_TOKEN_LENGTH
        and count / products >= MIN_DOCUMENT_FREQUENCY
        and structural_frequency.get(token, 0) / count >= MIN_STRUCTURAL_CONCENTRATION
    }


def render(catalog_path: Path, products: int, tokens: int, selected: set[str]) -> str:
    body = "".join(f'    "{token}",\n' for token in sorted(selected))
    return HEADER.format(
        catalog=catalog_path.as_posix(),
        products=products,
        tokens=tokens,
        min_df=MIN_DOCUMENT_FREQUENCY,
        min_conc=MIN_STRUCTURAL_CONCENTRATION,
        field=STRUCTURAL_FIELD,
        terms=body,
    )


def report_threshold_evidence(
    products: int,
    document_frequency: dict[str, int],
    structural_frequency: dict[str, int],
) -> None:
    """Print how populated the decision boundary is, so the cut stays auditable.

    A threshold is only defensible if it sits in a sparse region. If this count
    grows on some future catalog, the rule needs revisiting rather than nudging.
    """
    band = [
        token
        for token, count in document_frequency.items()
        if count / products >= MIN_DOCUMENT_FREQUENCY
        and 0.20 <= structural_frequency.get(token, 0) / count < 0.96
    ]
    print(
        f"  {len(band)} tokens sit between 20% and 96% structural concentration "
        f"(the cut is at {MIN_STRUCTURAL_CONCENTRATION:.0%})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument(
        "--dry-run", action="store_true", help="print the diff against the committed list"
    )
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    print(f"reading {catalog_path} ...", flush=True)
    products, document_frequency, structural_frequency = measure(catalog_path)
    selected = select(products, document_frequency, structural_frequency)

    print(f"  {products:,} products, {len(document_frequency):,} distinct tokens")
    print(f"  {len(selected)} scaffolding terms selected")
    report_threshold_evidence(products, document_frequency, structural_frequency)

    try:
        from src.stoplist import SCAFFOLDING as current
    except ImportError:
        current = frozenset()
    added, removed = sorted(selected - current), sorted(current - selected)
    if added:
        print(f"  + {' '.join(added)}")
    if removed:
        print(f"  - {' '.join(removed)}")
    if not added and not removed:
        print("  no change against the committed list")

    rendered = render(catalog_path, products, len(document_frequency), selected)
    if args.dry_run:
        print("\n--dry-run: nothing written")
        return
    Path(args.output).write_text(rendered, encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
