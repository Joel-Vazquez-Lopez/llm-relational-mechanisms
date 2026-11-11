"""Independent regression tests for the ML-090 cross-example indexing defect.

Run from repo root: python3 -m unittest discover -s tests -p 'test_ml090_indexing.py'
No checkpoint, historical report, grammar generator or protected example is used.
"""
import unittest
import torch


class TestMl090Indexing(unittest.TestCase):
    def setUp(self):
        self.logits = torch.zeros(4, 8)
        self.agent = torch.tensor([1, 1, 3, 3])
        self.patient = torch.tensor([2, 2, 4, 4])
        self.logits[0, [1, 2, 3]] = torch.tensor([6., 1., 50.])
        self.logits[1, [1, 2]] = torch.tensor([1., 6.])
        self.logits[2, [1, 3, 4]] = torch.tensor([100., 7., 0.])
        self.logits[3, [3, 4]] = torch.tensor([0., 7.])

    @staticmethod
    def rowwise(logits, a, p):
        row0 = torch.arange(0, logits.shape[0], 2)
        row1 = row0 + 1
        return .5 * ((logits[row0, a[row0]] - logits[row0, p[row0]])
                     - (logits[row1, a[row1]] - logits[row1, p[row1]]))

    def test_legacy_is_cartesian(self):
        got = self.logits[::2, self.agent[::2]]
        self.assertEqual(tuple(got.shape), (2, 2))
        self.assertEqual(got.tolist(), [[6., 50.], [100., 7.]])

    def test_corrected_returns_one_value_per_pair(self):
        got = self.rowwise(self.logits, self.agent, self.patient)
        self.assertEqual(tuple(got.shape), (2,))
        self.assertEqual(got.tolist(), [5., 7.])

    def test_permute_pairs_preserves_mean_and_permuted_vector(self):
        perm = torch.tensor([2, 3, 0, 1])
        reordered = self.rowwise(self.logits[perm], self.agent[perm], self.patient[perm])
        self.assertEqual(reordered.tolist(), [7., 5.])

    def test_identical_paired_logits_give_zero(self):
        x = self.logits.clone()
        x[1] = x[0]
        x[3] = x[2]
        got = self.rowwise(x, self.agent, self.patient)
        self.assertTrue(torch.equal(got, torch.zeros(2)))


if __name__ == "__main__":
    unittest.main()
