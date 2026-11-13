import unittest

from relational_lab.experiments import order_role_behavior as beh
from relational_lab.experiments import order_role_stimuli as ors


def scorer_from_rule(pick):
    """Fake scorer: ``pick(item) -> name`` chosen; margin +1 if correct else -1."""
    def scorer(items):
        out = []
        for it in items:
            correct = pick(it) == it.answer
            out.append(beh.Score(it.item_id, 0.0 if correct else -1.0, -1.0 if correct else 0.0, correct))
        return out
    return scorer


def first_name(it):
    return it.names[0].name


def subject_name(it):
    return next(t.name for t in it.names if t.is_subject)


class TestSummaries(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.items = ors.generate_items(names=ors.NAMES[:8])

    def test_perfect_model_passes_gate(self):
        s = beh.summarize(self.items, scorer_from_rule(lambda it: it.answer)(self.items), n_boot=200)
        self.assertEqual(s["overall_choice_accuracy"], 1.0)
        self.assertTrue(s["gate"]["passed"])
        self.assertTrue(s["gate"]["complete"])
        self.assertEqual(s["cross_construction_consistency"], 1.0)
        self.assertTrue(all(v == 1.0 for v in s["both_roles_correct_by_construction"].values()))

    def test_first_mention_heuristic_is_exposed(self):
        s = beh.summarize(self.items, scorer_from_rule(first_name)(self.items), n_boot=200)
        cr = s["by_construction_role"]
        self.assertEqual(cr["active/agent"]["choice_accuracy"], 1.0)
        self.assertEqual(cr["passive/agent"]["choice_accuracy"], 0.0)
        self.assertEqual(cr["object_cleft/agent"]["choice_accuracy"], 0.0)
        self.assertEqual(cr["by_cleft/agent"]["choice_accuracy"], 1.0)
        self.assertFalse(s["gate"]["passed"])
        # The same name is never picked for both questions, so both-correct is 0 everywhere.
        self.assertTrue(all(v == 0.0 for v in s["both_roles_correct_by_construction"].values()))
        self.assertEqual(s["cross_construction_consistency"], 0.0)

    def test_subject_heuristic_fails_exactly_where_subject_is_not_agent(self):
        s = beh.summarize(self.items, scorer_from_rule(subject_name)(self.items), n_boot=200)
        cr = s["by_construction_role"]
        self.assertEqual(cr["active/agent"]["choice_accuracy"], 1.0)
        self.assertEqual(cr["object_cleft/agent"]["choice_accuracy"], 1.0)
        self.assertEqual(cr["passive/agent"]["choice_accuracy"], 0.0)
        self.assertEqual(cr["by_cleft/agent"]["choice_accuracy"], 0.0)
        self.assertFalse(s["gate"]["passed"])

    def test_missing_scores_raise(self):
        scores = scorer_from_rule(lambda it: it.answer)(self.items)[:-1]
        with self.assertRaises(ValueError):
            beh.summarize(self.items, scores, n_boot=50)

    def test_gate_incomplete_without_all_constructions(self):
        actives = [i for i in self.items if i.construction == "active"]
        s = beh.summarize(actives, scorer_from_rule(lambda it: it.answer)(actives), n_boot=50)
        self.assertFalse(s["gate"]["complete"])
        self.assertFalse(s["gate"]["passed"])

    def test_question_wording_effect(self):
        def rule(it):
            return it.answer if it.question_voice == "active" else it.distractor
        s = beh.summarize(self.items, scorer_from_rule(rule)(self.items), n_boot=50)
        self.assertTrue(all(v == 1.0 for v in s["question_wording_effect"].values()))


class TestBootstrap(unittest.TestCase):

    def test_interval_brackets_mean_and_is_deterministic(self):
        data = {f"f{i}": [float(i % 3 == 0)] * 4 for i in range(30)}
        a = beh.bootstrap_ci(data, n_boot=500, seed=1)
        b = beh.bootstrap_ci(data, n_boot=500, seed=1)
        self.assertEqual(a, b)
        mean = sum(sum(v) for v in data.values()) / sum(len(v) for v in data.values())
        self.assertLessEqual(a[0], mean)
        self.assertGreaterEqual(a[1], mean)

    def test_constant_data_has_zero_width(self):
        self.assertEqual(beh.bootstrap_ci({"a": [1.0], "b": [1.0]}, n_boot=100), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
