
import re
import unittest
from collections import Counter

from relational_lab.experiments.cue_interactions import (
    generate_trials,
)


class TestCueInteractions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.trials = list(generate_trials())

    def test_design_size_and_uniqueness(self):
        self.assertEqual(len(self.trials), 192)

        ids = [trial.case_id for trial in self.trials]
        self.assertEqual(len(ids), len(set(ids)))

        counts = Counter(t.family_id for t in self.trials)
        self.assertEqual(len(counts), 6)
        self.assertTrue(all(count == 32 for count in counts.values()))

    def test_prompt_preserves_passage_exactly(self):
        for trial in self.trials:
            with self.subTest(case=trial.case_id):
                expected = (
                    f"Passage: {trial.passage}\n"
                    f"Question: {trial.question}\n"
                )
                self.assertIn(expected, trial.prompt)

    def test_gold_from_independent_event_parse(self):
        for trial in self.trials:
            with self.subTest(case=trial.case_id):

                dawn = re.search(
                    r"At dawn, ([^.]+)\.",
                    trial.passage,
                )
                self.assertIsNotNone(dawn)

                event_text = dawn.group(1)

                active = re.fullmatch(
                    r"(\w+) praised (\w+)",
                    event_text,
                )

                passive = re.fullmatch(
                    r"(\w+) was praised by (\w+)",
                    event_text,
                )

                self.assertTrue(active or passive)

                if active:
                    agent, patient = active.groups()
                else:
                    patient, agent = passive.groups()

                expected_gold = (
                    agent if trial.role == "agent"
                    else patient
                )

                self.assertEqual(trial.gold, expected_gold)

                if trial.role == "agent":
                    expected_question = (
                        f"At dawn, who praised {patient}?"
                        if trial.question_voice == "active"
                        else f"At dawn, who was {patient} praised by?"
                    )
                else:
                    expected_question = (
                        f"At dawn, whom did {agent} praise?"
                        if trial.question_voice == "active"
                        else f"At dawn, who was praised by {agent}?"
                    )

                self.assertEqual(
                    trial.question,
                    expected_question,
                )

    def test_event_order(self):
        for trial in self.trials:
            with self.subTest(case=trial.case_id):
                prefix = (
                    "At dawn,"
                    if trial.order == "target_first"
                    else "At dusk,"
                )
                self.assertTrue(trial.passage.startswith(prefix))
                self.assertIn("At dawn,", trial.passage)
                self.assertIn("At dusk,", trial.passage)


if __name__ == "__main__":
    unittest.main()
