"""Generate a competition-compliant adversarial session set.

Only participant-visible catalog metadata is used to select difficult targets.
The output deliberately omits intent cards, simulator behavior, messages, and
other private state; ``evaluator.local_evaluator`` derives them exactly as it
does for the public set.

The 200 sessions retain the official 40/40/15/5 scenario mix.  This is a local
stress set, not a reconstruction of the organizer's private evaluation data.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import coarse_category, intent_card
from tools.hard_cases import (
    COLORS,
    MATERIALS,
    classify_product,
    load_catalog,
    primary,
    synth_profile,
)


CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC_SET = ROOT / "data" / "public_set.jsonl"
HARD_SET = ROOT / "data" / "hard_set.jsonl"
GENERATED_TEST_SET = ROOT / "data" / "generated_test_set.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "generated_adversarial_set.jsonl"
PROTECTED_INPUTS = {CATALOG.resolve(), PUBLIC_SET.resolve(), HARD_SET.resolve()}

# Each tuple is (scenario, adversarial target-selection bucket, session count).
# Totals: Buying 80, Browsing 80, Intent Override 30, Boundary 10.
PLAN = (
    ("buying", "homogeneous_cluster", 27),
    ("buying", "budget_only_signal", 27),
    ("buying", "cross_category_collision", 26),
    ("browsing", "boilerplate_soft", 40),
    ("browsing", "degenerate_card", 40),
    ("intent_override", "generic_override", 30),
    ("boundary", "degenerate_card", 5),
    ("boundary", "homogeneous_cluster", 5),
)
EXPECTED_SCENARIOS = {
    "buying": 80,
    "browsing": 80,
    "intent_override": 30,
    "boundary": 10,
}
SESSION_KEYS = {
    "category_bucket",
    "difficulty_bucket",
    "ground_truth",
    "sample_id",
    "scenario_type",
    "user_profile",
}
PROFILE_KEYS = {
    "average_prior_rating",
    "preference_tags",
    "purchase_frequency",
    "rating_style",
    "summary",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def excluded_targets() -> set[str]:
    """Keep the new stress set independent of all shipped/local session sets."""
    targets: set[str] = set()
    for path in (PUBLIC_SET, HARD_SET, GENERATED_TEST_SET):
        for sample in load_jsonl(path):
            targets.add(str(sample["ground_truth"]["parent_asin"]))
    return targets


def bucket_products(catalog: list[dict]) -> dict[str, list[dict]]:
    """Classify products using only fields exposed in the frozen catalog."""
    cluster: collections.Counter[tuple] = collections.Counter()
    material_color: collections.Counter[tuple] = collections.Counter()
    enriched: list[tuple[dict, tuple]] = []

    for product in catalog:
        corpus = " ".join(
            str(product.get(field))
            for field in ("title", "features", "details", "description")
        )
        material = primary(corpus, MATERIALS)
        color = primary(corpus, COLORS)
        key = (coarse_category(product.get("categories") or []).lower(), material, color)
        cluster[key] += 1
        if material and color:
            material_color[(material, color)] += 1
        enriched.append((product, key))

    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for product, key in enriched:
        pair_count = material_color[(key[1], key[2])] if key[1] and key[2] else 0
        for bucket in classify_product(product, cluster[key], pair_count):
            buckets[bucket].append(product)
    return buckets


def build(seed: int) -> list[dict]:
    rng = random.Random(seed)
    catalog = load_catalog()
    buckets = bucket_products(catalog)
    excluded = excluded_targets()
    selected: set[str] = set()
    sessions: list[dict] = []

    for scenario, bucket, required in PLAN:
        candidates = list(buckets[bucket])
        rng.shuffle(candidates)
        picked = 0
        for product in candidates:
            parent_asin = str(product["parent_asin"])
            if parent_asin in excluded or parent_asin in selected:
                continue
            # The evaluator must be able to materialize a usable customer intent.
            if not intent_card(product).get("hard_constraints"):
                continue

            selected.add(parent_asin)
            picked += 1
            sessions.append({
                "category_bucket": "clothing",
                "difficulty_bucket": "hard",
                "ground_truth": {"parent_asin": parent_asin},
                "sample_id": f"adversarial_{scenario}_{bucket}_{picked:03d}",
                "scenario_type": scenario,
                # Safe aggregate fields only; no identifiers or purchase records.
                "user_profile": synth_profile(product, rng, misleading=False),
            })
            if picked == required:
                break
        if picked != required:
            raise ValueError(
                f"bucket {bucket!r} supplied {picked} unique targets; need {required}"
            )

    rng.shuffle(sessions)
    return sessions


def validate(sessions: list[dict]) -> None:
    catalog_ids = {str(product["parent_asin"]) for product in load_catalog()}
    target_ids = [sample["ground_truth"]["parent_asin"] for sample in sessions]
    sample_ids = [sample["sample_id"] for sample in sessions]

    if len(sessions) != 200:
        raise ValueError(f"expected 200 sessions, found {len(sessions)}")
    if any(set(sample) != SESSION_KEYS for sample in sessions):
        raise ValueError("session fields do not match the public-set structure")
    if any(set(sample["user_profile"]) != PROFILE_KEYS for sample in sessions):
        raise ValueError("user-profile fields do not match the public-set structure")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique")
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("ground-truth targets must be unique")
    if not set(target_ids) <= catalog_ids:
        raise ValueError("a ground-truth target is not in the frozen catalog")
    if set(target_ids) & excluded_targets():
        raise ValueError("adversarial targets overlap an existing session set")
    counts = collections.Counter(sample["scenario_type"] for sample in sessions)
    if counts != collections.Counter(EXPECTED_SCENARIOS):
        raise ValueError(f"invalid scenario mix: {dict(counts)}")
    forbidden = {"intent_card", "behavior", "session_id", "user_message"}
    if any(forbidden & set(sample) for sample in sessions):
        raise ValueError("output contains evaluator-private or runtime state")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output in PROTECTED_INPUTS:
        raise ValueError("refusing to overwrite a protected source dataset")

    sessions = build(args.seed)
    validate(sessions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(session, ensure_ascii=False) + "\n" for session in sessions),
        encoding="utf-8",
    )
    counts = collections.Counter(session["scenario_type"] for session in sessions)
    print(f"wrote {len(sessions)} validated sessions to {args.output}")
    print(dict(counts))


if __name__ == "__main__":
    main()
