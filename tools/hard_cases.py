"""Adversarial session generator for the conversational-search agent.

The public set (200 sessions) and the private set (800) share a fixed scenario
mix and are sampled *uniformly* from the frozen catalog. The shipped agent scores
0.859 on the public set, but that number is inflated by how often the simulated
customer's disclosed constraints happen to be a near-unique fingerprint of the
target - which is exactly what the verbatim-span reranker is built to exploit.

This tool builds a *stress* set instead: sessions whose targets sit in regions of
the catalog where the disclosed constraints are **not** discriminative, so the
reranker has little to work with and the weaker stages (category anchoring, facet
agreement, profile use, override handling) are forced to carry the session.

Everything stays inside the competition scope (docs/competition_specification.md,
Track 4 of the problem-statement PDF):

  * targets are real `parent_asin` values from the frozen catalog - no mock ASINs
  * the catalog is read-only
  * sessions use the published schema, so `evaluator/local_evaluator.py` scores
    them unmodified: `python3 -m evaluator.local_evaluator --dataset data/hard_set.jsonl`
  * the hidden intent card is still built by the evaluator's own `intent_card()`
    from the target's metadata - we only choose *which* targets and *which*
    scenario, never what the customer says

Buckets
-------
homogeneous_cluster  target shares (category, material, colour) with >= 40 other
                     products; span coverage ties the whole cluster.  (~10% of catalog)
budget_only_signal   target has a price but <= 1 distinctive non-material span;
                     price never enters the FTS text or the rerank text blob and
                     FixedPolicy never asks `budget`, so ~1/4 of the disclosed
                     constraints is dead weight.  (~13% of catalog)
boilerplate_soft     every soft preference is Amazon boilerplate ("Imported",
                     "Machine Wash", "Package Dimensions ..."); boilerplate
                     stripping removes most of the query signal.  (~4% of catalog)
degenerate_card      features+details collapse to a single repeated phrase or the
                     bare title; the customer can disclose almost nothing.  (~0.5%)
generic_override     intent_override whose "new" intent (hard_constraints[0]) is a
                     bare material/short phrase AND the target sits in a large
                     (category, material, colour) cluster, so the post-override
                     turns steer the agent into a crowd.
cross_category_collision   the target's own (category, material, colour) cluster
                     is small, but its (material, colour) pair is shared by many
                     hundreds of catalog products in other categories - the
                     bag-of-words route pulls that crowd into the pool.

Usage
-----
    python3 tools/hard_cases.py                 # write data/hard_set.jsonl
    python3 tools/hard_cases.py --run           # write, then score the shipped agent
    python3 tools/hard_cases.py --per-bucket 20 # sessions per bucket (default 16)
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import coarse_category, intent_card  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"
OUT = ROOT / "data" / "hard_set.jsonl"

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon",
    "denim", "linen", "suede", "canvas", "satin", "mesh", "fleece", "rubber",
    "stainless steel", "sterling silver",
)
COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "navy", "silver", "gold", "beige", "ivory",
)
# Words that carry no discriminative signal after the agent's boilerplate filter.
NOISE = {
    "imported", "closure", "machine", "wash", "care", "instructions", "made",
    "usa", "manufacturer", "dimensions", "package", "department", "asin", "model",
    "number", "discontinued", "polyester", "cotton", "nylon", "the", "and", "for",
    "with", "your", "our", "you", "this", "that", "solid", "colors", "color",
}
BOILERPLATE_MARKERS = (
    "imported", "machine wash", "hand wash", "package dimensions", "is discontinued",
    "discontinued by manufacturer", "item model number", "date first available",
)
# Bare head nouns that name products in more than one department.
AMBIGUOUS_HEADS = {
    "tank", "boot", "boots", "charm", "charms", "shell", "slip", "slips", "pump",
    "pumps", "band", "bands", "clip", "clips", "cuff", "cuffs", "wrap", "wraps",
    "mule", "mules", "flat", "flats", "tee", "cross", "link",
}

PROFILE_TAG_POOL = ("fit", "comfort", "durability", "style", "design", "color", "value", "brand", "performance")


def load_catalog() -> list[dict]:
    return [json.loads(line) for line in CATALOG.open(encoding="utf-8") if line.strip()]


def primary(text: str, options) -> str | None:
    lowered = text.lower()
    for option in options:
        if option in lowered:
            return option
    return None


def content_tokens(span: str) -> list[str]:
    return [
        tok.lower() for tok in TOKEN_RE.findall(span)
        if len(tok) > 2 and not tok.isdigit() and tok.lower() not in NOISE
    ]


def distinctive_spans(card: dict) -> list[str]:
    spans = card["hard_constraints"] + card["soft_preferences"]
    return [span for span in spans if len(content_tokens(span)) >= 3]


def specificity(span: str) -> int:
    """Rough count of the rare, matchable words in a constraint phrase."""
    return len(content_tokens(span))


def synth_profile(product: dict, rng: random.Random, misleading: bool = False) -> dict:
    """A schema-valid aggregate profile.

    When ``misleading`` we deliberately seed a tag the target cannot satisfy
    (e.g. "brand" for a no-name product), to probe safe-personalisation: the
    agent must not over-index on the profile.
    """
    corpus = " ".join(str(product.get(key)) for key in ("title", "features", "details"))
    tags: list[str] = []
    if primary(corpus, MATERIALS):
        tags.append("durability")
    if primary(corpus, COLORS):
        tags.append("color")
    tags.append("fit")
    if misleading:
        tags.append("brand")
    tags = list(dict.fromkeys(tags))[:3] or ["fit"]
    rating = rng.choice([None, 3.0, 4.0, 4.5, 5.0, 2.0])
    style = "critical" if (rating or 5) < 3.5 else "usually positive"
    return {
        "purchase_frequency": rng.choice(["1-2 prior purchases", "3-4 prior purchases", "5+ prior purchases"]),
        "average_prior_rating": rating,
        "rating_style": style,
        "preference_tags": tags,
        "summary": f"Prior purchases emphasize {', '.join(tags)}; ratings are {style}.",
    }


def classify_product(product: dict, cluster_size: int, matcol_count: int) -> list[str]:
    card = intent_card(product)
    corpus = " ".join(str(product.get(key)) for key in ("title", "features", "details", "description"))
    material = primary(corpus, MATERIALS)
    color = primary(corpus, COLORS)
    hard, soft = card["hard_constraints"], card["soft_preferences"]
    tags: list[str] = []

    if cluster_size >= 40 and material and color:
        tags.append("homogeneous_cluster")
    if product.get("price") not in (None, "") and len(distinctive_spans(card)) <= 1:
        tags.append("budget_only_signal")
    if soft and all(
        any(marker in span.lower() for marker in BOILERPLATE_MARKERS) or span.lower() in MATERIALS
        for span in soft
    ):
        tags.append("boilerplate_soft")
    if hard == soft or len({*hard, *soft}) <= 1:
        tags.append("degenerate_card")

    # override where the "new" intent the evaluator sends is a bare material and
    # the target is buried in a same-material crowd
    new_value = (hard or [""])[0]
    if material and cluster_size >= 40 and specificity(new_value) <= 2:
        tags.append("generic_override")

    # own cluster is small, but the (material, colour) pair is catalog-wide common
    if material and color and cluster_size <= 6 and matcol_count >= 700:
        tags.append("cross_category_collision")

    return tags


BUCKET_SCENARIO = {
    "homogeneous_cluster": "buying",
    "budget_only_signal": "buying",
    "boilerplate_soft": "browsing",
    "degenerate_card": "browsing",
    "generic_override": "intent_override",
    "cross_category_collision": "buying",
}


def build(per_bucket: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    catalog = load_catalog()

    # (coarse category, primary material, primary colour) cluster sizes, plus
    # the catalog-wide (material, colour) count regardless of category
    cluster = collections.Counter()
    matcol = collections.Counter()
    enriched: list[tuple[dict, tuple]] = []
    for product in catalog:
        corpus = " ".join(str(product.get(k)) for k in ("title", "features", "details", "description"))
        material = primary(corpus, MATERIALS)
        color = primary(corpus, COLORS)
        key = (coarse_category(product.get("categories") or []).lower(), material, color)
        cluster[key] += 1
        if material and color:
            matcol[(material, color)] += 1
        enriched.append((product, key))

    by_bucket: dict[str, list[dict]] = collections.defaultdict(list)
    for product, key in enriched:
        matcol_count = matcol[(key[1], key[2])] if key[1] and key[2] else 0
        for tag in classify_product(product, cluster[key], matcol_count):
            by_bucket[tag].append(product)

    sessions: list[dict] = []
    for bucket, scenario in BUCKET_SCENARIO.items():
        pool = by_bucket.get(bucket, [])
        rng.shuffle(pool)
        picked = 0
        for product in pool:
            if picked >= per_bucket:
                break
            card = intent_card(product)
            # skip targets the evaluator can't turn into a usable session
            if not card["hard_constraints"]:
                continue
            misleading = bucket in ("degenerate_card", "boilerplate_soft") and picked % 2 == 0
            sessions.append({
                "sample_id": f"hard_{bucket}_{picked:02d}",
                "scenario_type": scenario,
                "hard_bucket": bucket,
                "category_bucket": "clothing",
                "difficulty_bucket": "hard",
                "ground_truth": {"parent_asin": str(product["parent_asin"])},
                "user_profile": synth_profile(product, rng, misleading),
            })
            picked += 1

    rng.shuffle(sessions)
    return sessions


def run_eval(sessions: list[dict]) -> None:
    from evaluator.local_evaluator import catalog_index, evaluate
    from starter.agent import Agent

    catalog_ids, categories, products = catalog_index(str(CATALOG))
    bucket_of = {s["sample_id"]: s["hard_bucket"] for s in sessions}
    result = evaluate(Agent(str(CATALOG)), sessions, catalog_ids, categories, products)

    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for session in result["sessions"]:
        groups[bucket_of[session["sample_id"]]].append(session)

    def line(name: str, rows: list[dict]) -> str:
        n = len(rows)
        hit = sum(r["hit"] for r in rows) / n
        mrr = sum(r["reciprocal_rank"] for r in rows) / n
        mttc = sum((r["first_hit_turn"] or 11) for r in rows) / n
        eff = max(0.0, min(1.0, (11 - mttc) / 10))
        score = 0.5 * hit + 0.3 * mrr + 0.2 * eff
        return f"  {name:34s} n={n:3d}  hit@10={hit:5.3f}  mrr={mrr:5.3f}  mttc={mttc:5.2f}  score={score:5.3f}"

    print("\nShipped agent on the adversarial set")
    print("=" * 78)
    for bucket in BUCKET_SCENARIO:
        if groups.get(bucket):
            print(line(bucket, groups[bucket]))
    print("-" * 78)
    print(line("ALL", result["sessions"]))
    print(f"\n  (public-set reference: hit@10=0.940  mrr=0.791  mttc=3.41  score=0.859)")
    print(f"  reported tokens: {result['reported_token_usage']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-bucket", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run", action="store_true", help="score the shipped agent after writing")
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()

    sessions = build(args.per_bucket, args.seed)
    Path(args.output).write_text(
        "".join(json.dumps(s) + "\n" for s in sessions), encoding="utf-8"
    )
    counts = collections.Counter(s["hard_bucket"] for s in sessions)
    print(f"wrote {len(sessions)} sessions -> {args.output}")
    for bucket in BUCKET_SCENARIO:
        print(f"  {bucket:34s} {counts.get(bucket, 0)}")

    if args.run:
        run_eval(sessions)


if __name__ == "__main__":
    main()
