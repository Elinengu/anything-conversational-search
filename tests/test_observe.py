"""Guards for tools/observe.py.

The observer annotates each customer turn by recognising the simulator's reply
templates. Those templates live in the organizer-owned evaluator, so if they ever
change these tests fail loudly rather than letting the transcripts silently
degrade to unlabelled text.
"""

from __future__ import annotations

import unittest

from evaluator.local_evaluator import customer_reply, initial_message
from tools.observe import DIAGNOSES, annotate_message, diagnose


SAMPLE = {
    "scenario_type": "buying",
    "intent_card": {
        "hard_constraints": ["100% Leather", "color: black"],
        "soft_preferences": ["Buckle closure", "Imported"],
    },
}


class AnnotationTests(unittest.TestCase):
    def test_disclosure_template_is_recognised(self) -> None:
        message, _ = customer_reply(SAMPLE, "material", set(), False)
        annotation = annotate_message(message)
        self.assertEqual(annotation["kind"], "disclosed")
        self.assertIn("100% Leather", annotation["revealed"])

    def test_no_preference_template_is_recognised(self) -> None:
        message, _ = customer_reply(SAMPLE, "size", set(), False)
        self.assertEqual(annotate_message(message)["kind"], "no_preference")

    def test_boundary_decline_template_is_recognised(self) -> None:
        message, used = customer_reply(
            {**SAMPLE, "scenario_type": "boundary"}, "color", set(), False
        )
        self.assertTrue(used)
        self.assertEqual(annotate_message(message)["kind"], "boundary_decline")

    def test_stall_template_is_recognised(self) -> None:
        message, _ = customer_reply(SAMPLE, None, set(), False)
        self.assertEqual(annotate_message(message)["kind"], "stalled")

    def test_opening_and_override_templates_are_recognised(self) -> None:
        opening = initial_message(SAMPLE, "Belts", set())
        self.assertEqual(annotate_message(opening)["kind"], "opening")
        override = "Actually, ignore my earlier preference. What I need is: 100% Leather."
        self.assertEqual(annotate_message(override)["kind"], "override")


def _turn(turn: int, pool_rank, ranked_rank, withheld=True, shown_rank=None) -> dict:
    return {
        "turn": turn,
        "in": {"kind": "disclosed", "revealed": []},
        "retrieval": {"target_pool_rank": pool_rank},
        "rerank": {"target_rank": ranked_rank},
        "out": {"withheld": withheld, "target_shown_rank": shown_rank},
    }


class DiagnosisTests(unittest.TestCase):
    def test_missing_target_never_retrieved(self) -> None:
        record = {"turns": [_turn(1, None, None)], "behavior": {}}
        outcome = {"hit": False, "first_hit_turn": None}
        self.assertEqual(diagnose(record, outcome)["label"], "never_retrieved")

    def test_retrieved_but_out_of_reach_is_a_ranking_problem(self) -> None:
        record = {"turns": [_turn(t, 40, 40) for t in (1, 2, 3)], "behavior": {}}
        self.assertEqual(diagnose(record, {"hit": False, "first_hit_turn": None})["label"], "ranked_out")

    def test_wasted_turns_are_counted_from_the_first_convertible_top_ten(self) -> None:
        record = {"turns": [_turn(1, 40, 40), _turn(2, 1, 1), _turn(3, 1, 1, False, 1)], "behavior": {}}
        diagnosis = diagnose(record, {"hit": True, "first_hit_turn": 3})
        self.assertEqual(diagnosis["earliest_top10_turn"], 2)
        self.assertEqual(diagnosis["turns_left_on_table"], 1)

    def test_pre_override_placements_are_not_counted_as_wasted(self) -> None:
        # local_evaluator.py:252 ignores any hit before the override fires, so a
        # top-10 placement on turn 1 was never convertible.
        record = {
            "turns": [_turn(1, 1, 1), _turn(2, 1, 1), _turn(3, 1, 1, False, 1)],
            "behavior": {"override": {"turn": 3}},
        }
        diagnosis = diagnose(record, {"hit": True, "first_hit_turn": 3})
        self.assertEqual(diagnosis["earliest_top10_turn"], 3)
        self.assertEqual(diagnosis["turns_left_on_table"], 0)

    def test_every_diagnosis_label_has_an_explanation(self) -> None:
        record = {"turns": [_turn(1, 5, 5)], "behavior": {}}
        for outcome in ({"hit": True, "first_hit_turn": 1}, {"hit": False, "first_hit_turn": None}):
            self.assertIn(diagnose(record, outcome)["label"], DIAGNOSES)


if __name__ == "__main__":
    unittest.main()
