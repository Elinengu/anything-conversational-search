"""Guards for tools/dual_track_harness.py.

The harness wraps the organizer's ``customer_reply`` to make the browsing
customer disclose only on a pointed question. These tests pin that wrapper's
logic without loading the 50,000-row catalog, and check that its disclosure
wording still parses under tools/observe.py's annotator.
"""

from __future__ import annotations

import unittest

from tools.dual_track_harness import DISCLOSURE, _make_reply
from tools.observe import annotate_message


CARD = {
    "hard_constraints": ["100% Leather", "color: black"],
    "soft_preferences": ["Buckle closure", "Wide width"],
}


def _sample(scenario: str) -> dict:
    return {"scenario_type": scenario, "intent_card": CARD}


class BrowsingDisclosureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reply = _make_reply(DISCLOSURE)

    def test_broad_question_reveals_nothing_to_a_browser(self) -> None:
        disclosed: set[str] = set()
        message, _ = self.reply(_sample("browsing"), "other", disclosed, False)
        self.assertEqual(disclosed, set())
        self.assertIn("browsing", message.lower())

    def test_none_attribute_reveals_nothing_to_a_browser(self) -> None:
        disclosed: set[str] = set()
        _, _ = self.reply(_sample("browsing"), None, disclosed, False)
        self.assertEqual(disclosed, set())

    def test_pointed_question_reveals_exactly_one_constraint(self) -> None:
        disclosed: set[str] = set()
        message, _ = self.reply(_sample("browsing"), "material", disclosed, False)
        self.assertEqual(disclosed, {"100% Leather"})
        self.assertEqual(annotate_message(message)["kind"], "disclosed")

    def test_pointed_question_with_no_match_declines(self) -> None:
        disclosed: set[str] = set()
        message, _ = self.reply(_sample("browsing"), "size", {"Wide width"}, False)
        self.assertEqual(annotate_message(message)["kind"], "no_preference")


class OtherScenariosAreUntouchedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reply = _make_reply(DISCLOSURE)

    def test_buyer_still_dumps_on_the_broad_question(self) -> None:
        disclosed: set[str] = set()
        self.reply(_sample("buying"), "other", disclosed, False)
        self.assertEqual(disclosed, {"100% Leather", "color: black"})

    def test_boundary_decline_is_the_organizers(self) -> None:
        message, used = self.reply(_sample("boundary"), "color", set(), False)
        self.assertTrue(used)
        self.assertEqual(annotate_message(message)["kind"], "boundary_decline")


if __name__ == "__main__":
    unittest.main()
