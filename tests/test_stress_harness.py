"""Guards for tools/stress_harness.py - the composable customer stressors.

Pinned without loading the 50,000-row catalog.
"""

from __future__ import annotations

import random
import unittest

from src.text import constraint_spans
from tools.stress_harness import (
    StressCustomer,
    parse_spec,
    paraphrase_disclosure,
    _synonym_sub,
    _LEADINS,
    _LEADINS_HEAVY,
)

CARD = {
    "hard_constraints": ["100% Leather", "color: black"],
    "soft_preferences": ["Buckle closure", "Wide width"],
}
CATS = {"T": ["Clothing, Shoes & Jewelry", "Men", "Belts"]}


def _customer(scenario: str, **spec) -> StressCustomer:
    full = {"paraphrase": "", "browse_gated": False, "decoy": False}
    full.update(spec)
    return StressCustomer(
        sample={"scenario_type": scenario, "user_profile": {}},
        card=CARD, behavior={}, categories=CATS, target="T",
        rng=random.Random(0), index_products=None, **full,
    )


class SpecParsingTests(unittest.TestCase):
    def test_official_is_all_off(self) -> None:
        self.assertEqual(parse_spec("official"),
                         {"paraphrase": "", "browse_gated": False, "decoy": False})

    def test_compound_spec(self) -> None:
        self.assertEqual(
            parse_spec("paraphrase:medium+browse-gated"),
            {"paraphrase": "medium", "browse_gated": True, "decoy": False},
        )

    def test_unknown_stressor_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_spec("teleport")

    def test_bad_paraphrase_level_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_spec("paraphrase:extreme")

    def test_heavy_level_parses(self) -> None:
        self.assertEqual(parse_spec("paraphrase:heavy+browse-gated"),
                         {"paraphrase": "heavy", "browse_gated": True, "decoy": False})


class BaseBehaviourTests(unittest.TestCase):
    def test_no_stressor_matches_the_official_customer(self) -> None:
        c = _customer("buying")
        self.assertIn("A key requirement is: 100% Leather.", c.opening())
        msg = c.reply(2, "other")
        self.assertEqual(msg, "For that, what matters is: color: black; Buckle closure.")

    def test_boundary_decline_is_unchanged(self) -> None:
        c = _customer("browsing")  # boundary shares the browsing opening
        c.scenario = "boundary"
        self.assertIn("judgment", c.reply(2, "color"))


class BrowseGatedTests(unittest.TestCase):
    def test_broad_ask_reveals_nothing(self) -> None:
        c = _customer("browsing", browse_gated=True)
        msg = c.reply(2, "other")
        self.assertIn("still just browsing", msg)
        self.assertEqual(c.disclosed, set())

    def test_pointed_ask_reveals_one_constraint(self) -> None:
        c = _customer("browsing", browse_gated=True)
        msg = c.reply(2, "material")
        self.assertEqual(c.disclosed, {"100% Leather"})
        self.assertIn("100% Leather", msg)

    def test_buyer_is_untouched_by_browse_gating(self) -> None:
        c = _customer("buying", browse_gated=True)
        c.opening()  # discloses the first hard constraint, as the official sim does
        c.reply(2, "other")  # ... and "other" still drains two more
        self.assertEqual(c.disclosed, {"100% Leather", "color: black", "Buckle closure"})


class ParaphraseTests(unittest.TestCase):
    def test_medium_rewords_the_constraint(self) -> None:
        out = paraphrase_disclosure(["color: black"], "medium", random.Random(1))
        self.assertNotIn("color: black", out)
        self.assertIn("black", out)

    def test_light_keeps_tokens_changes_frame(self) -> None:
        out = paraphrase_disclosure(["100% Leather"], "light", random.Random(1))
        self.assertIn("100% Leather", out)
        self.assertNotIn("For that, what matters is", out)

    def test_paraphrase_composes_with_browse_gating(self) -> None:
        c = _customer("browsing", browse_gated=True, paraphrase="medium")
        msg = c.reply(2, "material")
        self.assertEqual(c.disclosed, {"100% Leather"})
        self.assertNotIn("For that, what matters is", msg)

    def test_heavy_substitutes_a_synonym(self) -> None:
        # across many seeds the phrase is rewritten most of the time.
        changed = sum(
            _synonym_sub("a waterproof leather strap", random.Random(s)) != "a waterproof leather strap"
            for s in range(40)
        )
        self.assertGreater(changed, 30)

    def test_heavy_rewrite_is_deterministic_per_seed(self) -> None:
        a = paraphrase_disclosure(["100% Leather", "Buckle closure"], "heavy", random.Random(7))
        b = paraphrase_disclosure(["100% Leather", "Buckle closure"], "heavy", random.Random(7))
        self.assertEqual(a, b)

    def test_heavy_drops_the_verbatim_token(self) -> None:
        out = paraphrase_disclosure(["100% Cotton"], "heavy", random.Random(3))
        self.assertNotIn("100% Cotton", out)


class DecoyTests(unittest.TestCase):
    def test_decoy_needs_a_real_override_and_a_product(self) -> None:
        # No index_products -> nothing to derive a decoy from; must not crash.
        c = _customer("intent_override", decoy=True)
        self.assertIsInstance(c, StressCustomer)


class ConstraintSpanCarrierTests(unittest.TestCase):
    """constraint_spans() must isolate the value from every paraphrased carrier
    sentence this harness can produce - not just the official template's
    colon-delimited wording."""

    #: A fixed 2-word value satisfies constraint_spans' min_words=2 default.
    VALUE = "synthetic sole"

    def test_every_carrier_template_isolates_the_value(self) -> None:
        for template in list(_LEADINS) + list(_LEADINS_HEAVY):
            with self.subTest(template=template):
                spans = constraint_spans(template.format(self.VALUE))
                self.assertIn(self.VALUE, spans, f"value lost for template: {template!r}")
                self.assertEqual(
                    len(spans), 1, f"carrier leaked into a span for: {template!r}"
                )

    def test_heavy_fused_joiners_isolate_each_value(self) -> None:
        joiners = [", and ", ", plus ", " - also ", ", and honestly "]
        wrappers = [
            "I'm after something {}.", "Ideally {}.", "What I care about: {}.",
            "Looking for {} really.", "So, {} - that's the gist.",
        ]
        for joiner in joiners:
            body = joiner.join(["synthetic sole", "breathable mesh"])
            for wrapper in wrappers:
                text = wrapper.format(body)
                spans = constraint_spans(text)
                with self.subTest(joiner=joiner, wrapper=wrapper):
                    self.assertIn("synthetic sole", spans)
                    self.assertIn("breathable mesh", spans)

    def test_bug_report_examples(self) -> None:
        self.assertEqual(
            constraint_spans("One more thing - a breathable net weave."),
            ["breathable net weave"],
        )
        # "imported" alone strips to a 1-word span, dropped by min_words=2 -
        # the same fate a single-word official-template constraint already
        # has today; not a new behaviour introduced by this fix.
        self.assertEqual(constraint_spans("imported matters to me as well."), [])
        self.assertEqual(
            constraint_spans("I'd also want it to be synthetic sole."),
            ["synthetic sole"],
        )

    def test_reword_heavy_filler_prefixes_still_isolate_the_value(self) -> None:
        # _FILLERS in tools/stress_harness.py can prepend "a bit "/"kind of "
        # directly onto the disclosed value before it ever reaches a leadin
        # template - a related edge case surfaced while auditing _LEADINS.
        for prefix in ("a bit ", "kind of ", "something ", "ideally ", "really "):
            with self.subTest(prefix=prefix):
                spans = constraint_spans(f"One more thing - {prefix}synthetic sole.")
                self.assertIn("synthetic sole", spans)


if __name__ == "__main__":
    unittest.main()
