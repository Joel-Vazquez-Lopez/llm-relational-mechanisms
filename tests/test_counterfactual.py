import unittest
from relational_lab.analysis.counterfactual import QueryResult, diagnose_pair

class CounterfactualTests(unittest.TestCase):
    def test_local_right_global_wrong(self):
        a = QueryResult("Noor", {"Noor":7., "Maya":1., "Erin":9.}, "Erin")
        b = QueryResult("Maya", {"Noor":1., "Maya":8., "Erin":10.}, "Erin")
        d = diagnose_pair(a, b)
        self.assertTrue(d.local_pair_correct_both)
        self.assertEqual(d.change_in_new_vs_old, 13)
        self.assertEqual(d.global_gold_margin_before, -2)
        self.assertEqual(d.global_gold_margin_after, -2)
        self.assertFalse(d.generated_correct_before)
        self.assertFalse(d.generated_correct_after)

    def test_correct_candidate(self):
        a = QueryResult("Noor", {"Noor":9., "Maya":1., "Erin":2.})
        b = QueryResult("Maya", {"Noor":1., "Maya":10., "Erin":2.})
        d = diagnose_pair(a, b)
        self.assertTrue(d.local_pair_correct_both)
        self.assertIsNone(d.generated_correct_before)
        self.assertEqual(d.candidate_winner_after, "Maya")

    def test_different_candidates_rejected(self):
        with self.assertRaisesRegex(ValueError, "candidate sets"):
            diagnose_pair(QueryResult("A", {"A":1., "B":2.}),
                          QueryResult("B", {"A":2., "B":1., "C":0.}))

    def test_unchanged_gold_rejected(self):
        with self.assertRaisesRegex(ValueError, "change"):
            diagnose_pair(QueryResult("A", {"A":1., "B":2.}),
                          QueryResult("A", {"A":2., "B":1.}))

    def test_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            QueryResult("A", {"A":float("nan"), "B":1.}).global_gold_margin()

    def test_tie_has_no_unique_winner(self):
        self.assertIsNone(QueryResult("A", {"A":2., "B":2.}).unique_candidate_winner())

    def test_not_infer_generation(self):
        d = diagnose_pair(QueryResult("A", {"A":2., "B":1.}),
                          QueryResult("B", {"A":1., "B":3.}))
        self.assertIsNone(d.generated_answer_before)
        self.assertIsNone(d.generated_correct_before)

if __name__ == "__main__":
    unittest.main()
