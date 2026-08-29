"""Generate a deterministic, public-schema-compatible evaluation set.

The catalog and existing session sets are read-only inputs.  The generated file
contains only public session fields; hidden intent cards and customer behavior
continue to be derived by ``evaluator/local_evaluator`` at evaluation time.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC_SET = ROOT / "data" / "public_set.jsonl"
HARD_SET = ROOT / "data" / "hard_set.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "generated_test_set.jsonl"
PROTECTED_INPUTS = {CATALOG.resolve(), PUBLIC_SET.resolve()}

SCENARIO_COUNTS = {
    "buying": 80,
    "browsing": 80,
    "intent_override": 30,
    "boundary": 10,
}
DIFFICULTY_BY_SCENARIO = {
    "buying": "easy",
    "browsing": "medium",
    "intent_override": "hard",
    "boundary": "medium",
}

TAG_PATTERNS = (
    ("fit", r"\b(?:fit|fitted|size|sizing|width|wide|narrow|adjustable)\b"),
    ("comfort", r"\b(?:comfort|comfortable|cushion|padded|soft|lightweight)\b"),
    ("material", r"\b(?:cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric|linen|denim|rubber|steel|silver|gold)\b"),
    ("style", r"\b(?:style|stylish|fashion|casual|formal|dress|jewelry|costume|classic)\b"),
    ("durability", r"\b(?:durable|durability|reinforced|sturdy|resistant|stainless|leather|rubber)\b"),
    ("performance", r"\b(?:performance|athletic|running|sport|workout|wicking|compression|support)\b"),
    ("warmth", r"\b(?:warm|warmth|winter|thermal|fleece|insulated|wool)\b"),
    ("weather", r"\b(?:weather|waterproof|water-resistant|rain|snow|sun|outdoor|uv|upf)\b"),
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def product_text(product: dict) -> str:
    values: list[str] = []
    for field in ("title", "features", "description", "categories", "details", "store"):
        value = product.get(field)
        if isinstance(value, dict):
            values.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value not in (None, ""):
            values.append(str(value))
    return " ".join(values).lower()


def preference_tags(product: dict) -> list[str]:
    text = product_text(product)
    tags = [name for name, pattern in TAG_PATTERNS if re.search(pattern, text)]
    return tags[:4] or ["general shopping"]


def profile(product: dict, rating: float, rating_style: str) -> dict:
    tags = preference_tags(product)
    return {
        "average_prior_rating": rating,
        "preference_tags": tags,
        "purchase_frequency": "3-4 prior purchases",
        "rating_style": rating_style,
        "summary": f"Prior purchases emphasize {', '.join(tags)}; ratings are {rating_style}.",
    }


def existing_targets() -> set[str]:
    targets: set[str] = set()
    for path in (PUBLIC_SET, HARD_SET):
        if not path.exists():
            continue
        for sample in load_jsonl(path):
            targets.add(str(sample["ground_truth"]["parent_asin"]))
    return targets


def build(seed: int) -> list[dict]:
    rng = random.Random(seed)
    excluded = existing_targets()
    products = [
        product
        for product in load_jsonl(CATALOG)
        if str(product.get("parent_asin", "")) not in excluded
        and str(product.get("title") or "").strip()
        and product.get("categories")
    ]

    required = sum(SCENARIO_COUNTS.values())
    if len(products) < required:
        raise ValueError(f"need {required} eligible catalog products, found {len(products)}")
    selected = rng.sample(products, required)

    scenarios = [
        scenario
        for scenario, count in SCENARIO_COUNTS.items()
        for _ in range(count)
    ]
    rng.shuffle(scenarios)

    # This mirrors the public set's aggregate prior-rating distribution without
    # copying any public profile or tying a synthetic history to product ratings.
    ratings = (
        [(5.0, "usually positive")] * 134
        + [(4.0, "mixed")] * 21
        + [(3.0, "critical")] * 22
        + [(2.0, "critical")] * 9
        + [(1.0, "critical")] * 14
    )
    rng.shuffle(ratings)

    sessions: list[dict] = []
    for index, (product, scenario, rating) in enumerate(zip(selected, scenarios, ratings), 1):
        sessions.append({
            "category_bucket": "clothing",
            "difficulty_bucket": DIFFICULTY_BY_SCENARIO[scenario],
            "ground_truth": {"parent_asin": str(product["parent_asin"])},
            "sample_id": f"generated_{index:04d}",
            "scenario_type": scenario,
            "user_profile": profile(product, *rating),
        })
    return sessions


def validate(sessions: list[dict]) -> None:
    expected_keys = {
        "category_bucket", "difficulty_bucket", "ground_truth", "sample_id",
        "scenario_type", "user_profile",
    }
    catalog_ids = {str(product["parent_asin"]) for product in load_jsonl(CATALOG)}
    public_ids = {
        str(sample["ground_truth"]["parent_asin"])
        for sample in load_jsonl(PUBLIC_SET)
    }
    sample_ids = [sample["sample_id"] for sample in sessions]
    target_ids = [sample["ground_truth"]["parent_asin"] for sample in sessions]

    if any(set(sample) != expected_keys for sample in sessions):
        raise ValueError("generated session keys do not match the public schema")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique")
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("ground-truth targets must be unique")
    if not set(target_ids) <= catalog_ids:
        raise ValueError("a ground-truth target is missing from the catalog")
    if set(target_ids) & public_ids:
        raise ValueError("generated targets overlap the public set")
    if Counter(sample["scenario_type"] for sample in sessions) != Counter(SCENARIO_COUNTS):
        raise ValueError("scenario counts do not match the competition mix")
    for sample in sessions:
        if sample["difficulty_bucket"] != DIFFICULTY_BY_SCENARIO[sample["scenario_type"]]:
            raise ValueError("difficulty bucket does not match the public-set convention")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output in PROTECTED_INPUTS:
        raise ValueError("refusing to overwrite catalog.jsonl or public_set.jsonl")

    sessions = build(args.seed)
    validate(sessions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(session, ensure_ascii=False) + "\n" for session in sessions),
        encoding="utf-8",
    )
    print(f"wrote {len(sessions)} validated sessions to {args.output}")
    print(dict(Counter(session["scenario_type"] for session in sessions)))


if __name__ == "__main__":
    main()
