import json
import math
import re
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from relational_lab.experiments import order_role_stimuli as ors


class TestTemplates(unittest.TestCase):
    """Exact strings for one event, so any template change is visible."""

    def test_sentences(self):
        expected = {
            "active": "Leon praised Mira.",
            "passive": "Mira was praised by Leon.",
            "object_cleft": "It was Mira that Leon praised.",
            "by_cleft": "It was by Leon that Mira was praised.",
            "subject_cleft": "It was Leon who praised Mira.",
        }
        for construction, sentence in expected.items():
            with self.subTest(construction=construction):
                self.assertEqual(ors.render_sentence(construction, "Leon", "Mira", "praised"), sentence)

    def test_questions_never_name_a_participant(self):
        expected = {
            ("agent", "active"): "Who praised someone?",
            ("agent", "passive"): "Who was someone praised by?",
            ("patient", "active"): "Who did someone praise?",
            ("patient", "passive"): "Who was praised by someone?",
        }
        for (role, voice), question in expected.items():
            with self.subTest(role=role, voice=voice):
                self.assertEqual(ors.render_question(role, voice, "praise", "praised"), question)

    def test_unknown_values_raise(self):
        with self.assertRaises(ValueError):
            ors.render_sentence("dative", "A", "B", "praised")
        with self.assertRaises(ValueError):
            ors.render_question("theme", "active", "praise", "praised")


class TestDesign(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.items = ors.generate_items()
        cls.main = [i for i in cls.items if not i.is_control]

    def test_size_and_unique_ids(self):
        n_names = len(ors.NAMES)
        n_families = n_names * (n_names - 1) // 2
        per_family = 2 * 2 * len(ors.CONSTRUCTIONS) * 2 * 2  # orientations x verbs x constructions x roles x qvoices
        self.assertEqual(len(self.items), n_families * per_family)
        ids = [i.item_id for i in self.items]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_event_is_fully_crossed(self):
        cells = defaultdict(set)
        for it in self.items:
            cells[it.event_id].add((it.construction, it.role_asked, it.question_voice))
        full = {(c, r, q) for c in ors.CONSTRUCTIONS for r in ors.ROLES for q in ors.QUESTION_VOICES}
        for event, seen in cells.items():
            self.assertEqual(seen, full, event)

    def test_each_family_has_both_role_assignments(self):
        orientations = defaultdict(set)
        for it in self.items:
            orientations[it.family_id].add((it.agent, it.patient))
        for fam, pairs in orientations.items():
            a, b = fam.split("|")
            self.assertEqual(pairs, {(a, b), (b, a)}, fam)

    def test_answers_match_roles(self):
        for it in self.items:
            if it.role_asked == "agent":
                self.assertEqual((it.answer, it.distractor), (it.agent, it.patient))
            else:
                self.assertEqual((it.answer, it.distractor), (it.patient, it.agent))
            self.assertNotIn(it.agent, it.question)
            self.assertNotIn(it.patient, it.question)

    def test_labels_are_exactly_uncorrelated_in_main_constructions(self):
        for key, r in ors.label_correlations(self.main).items():
            with self.subTest(pair=key):
                self.assertAlmostEqual(r, 0.0, places=12)

    def test_labels_are_correlated_if_only_canonical_forms_were_used(self):
        # Sanity check of the check: actives alone confound all three codes.
        actives = [i for i in self.items if i.construction == "active"]
        for r in ors.label_correlations(actives).values():
            self.assertAlmostEqual(r, 1.0, places=12)

    def test_answer_position_is_balanced_in_main_constructions(self):
        for role in ors.ROLES:
            subset = [i for i in self.main if i.role_asked == role]
            share = sum(i.answer_is_first for i in subset) / len(subset)
            self.assertEqual(share, 0.5, role)

    def test_name_spans_and_labels(self):
        for it in self.items:
            first, second = it.names
            self.assertLess(first.char_end, second.char_start)
            for tok in it.names:
                self.assertEqual(it.sentence[tok.char_start:tok.char_end], tok.name)
                self.assertEqual(len(re.findall(rf"\b{tok.name}\b", it.sentence)), 1)
            self.assertTrue(first.is_first)
            self.assertFalse(second.is_first)
            self.assertEqual(sum(t.is_agent for t in it.names), 1)
            self.assertEqual(sum(t.is_subject for t in it.names), 1)
            agent_tok = next(t for t in it.names if t.is_agent)
            self.assertEqual(agent_tok.name, it.agent)

    def test_construction_label_table(self):
        expected = {  # (agent is first, agent is subject)
            "active": (True, True),
            "passive": (False, False),
            "object_cleft": (False, True),
            "by_cleft": (True, False),
            "subject_cleft": (True, True),
        }
        for it in self.items:
            agent_tok = next(t for t in it.names if t.is_agent)
            self.assertEqual((agent_tok.is_first, agent_tok.is_subject), expected[it.construction], it.item_id)

    def test_prompt_contains_sentence_and_question_verbatim(self):
        for it in self.items[:500]:
            self.assertEqual(it.prompt, f"Sentence: {it.sentence}\nQuestion: {it.question}\nAnswer:")

    def test_every_name_follows_a_space_in_the_prompt(self):
        for it in self.items:
            for tok in it.names:
                self.assertEqual(it.prompt[tok.prompt_char_start:tok.prompt_char_end], tok.name)
                self.assertEqual(it.prompt[tok.prompt_char_start - 1], " ", it.item_id)

    def test_names_and_verbs_are_balanced(self):
        as_agent = Counter(i.agent for i in self.main)
        as_patient = Counter(i.patient for i in self.main)
        self.assertEqual(set(as_agent.values()), {as_agent[ors.NAMES[0]]})
        self.assertEqual(as_agent, as_patient)
        verb_counts = Counter(i.lemma for i in self.main)
        self.assertLessEqual(max(verb_counts.values()) - min(verb_counts.values()), len(ors.MAIN_CONSTRUCTIONS) * 4)

    def test_verbs_differ_within_an_orientation(self):
        verbs = defaultdict(list)
        for it in self.items:
            if it.construction == "active" and it.role_asked == "agent" and it.question_voice == "active":
                verbs[(it.agent, it.patient)].append(it.lemma)
        for key, vs in verbs.items():
            self.assertEqual(len(vs), len(set(vs)), key)


class TestSplits(unittest.TestCase):

    def test_families_are_disjoint_across_splits(self):
        items = ors.generate_items()
        by_family = defaultdict(set)
        for it in items:
            by_family[it.family_id].add((it.split, it.test_set))
        for fam, assignments in by_family.items():
            self.assertEqual(len(assignments), 1, fam)

    def test_split_proportions(self):
        splits = ors.split_families(ors.NAMES, seed=2026, dev_fraction=0.15, n_test_sets=4)
        counts = Counter(splits.values())
        n = len(splits)
        self.assertEqual(sum(v for (s, _), v in counts.items() if s == "dev"), round(0.15 * n))
        test_sizes = [v for (s, _), v in counts.items() if s == "test"]
        self.assertEqual(len(test_sizes), 4)
        self.assertLessEqual(max(test_sizes) - min(test_sizes), 1)

    def test_deterministic_and_seed_sensitive(self):
        a = ors.generate_items(seed=7)
        b = ors.generate_items(seed=7)
        c = ors.generate_items(seed=8)
        self.assertEqual(a, b)
        self.assertNotEqual([i.split for i in a], [i.split for i in c])


class TestValidationAndIO(unittest.TestCase):

    def test_rejects_substring_names(self):
        with self.assertRaises(ValueError):
            ors.generate_items(names=("Ann", "Anna", "Tom"))

    def test_rejects_duplicate_names(self):
        with self.assertRaises(ValueError):
            ors.generate_items(names=("Tom", "Tom", "Anna"))

    def test_filter_names(self):
        self.assertEqual(ors.filter_names(["Anna", "Maximilian"], lambda n: len(n) < 6), ("Anna",))

    def test_write_dataset_round_trip_and_hash(self):
        items = ors.generate_items(names=ors.NAMES[:6])
        with tempfile.TemporaryDirectory() as d:
            m1 = ors.write_dataset(items, Path(d), {"seed": 2026})
            rows = [json.loads(line) for line in (Path(d) / "items.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), len(items))
            self.assertEqual(rows[0]["item_id"], items[0].item_id)
            m2 = ors.write_dataset(items, Path(d), {"seed": 2026})
            self.assertEqual(m1["items_sha256"], m2["items_sha256"])
            for r in m1["label_correlations_main"].values():
                self.assertTrue(math.isclose(r, 0.0, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
