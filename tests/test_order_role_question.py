"""OTR-04 question-side patching: pairing, shared tail, exact nulls and
reproduction checks (raw and chat-template mode), aggregation."""

import importlib.util
import math
import unittest

from relational_lab.experiments import order_role_question as orq
from relational_lab.experiments import order_role_stimuli as ors

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "tokenizers"))


def some_items(n_names=4):
    items = ors.generate_items(names=ors.NAMES[:n_names])
    return [i for i in items if i.construction in ors.MAIN_CONSTRUCTIONS]


class TestPairing(unittest.TestCase):

    def test_flip_partner_asks_the_other_role_in_the_same_voice(self):
        for it in some_items()[:40]:
            s = orq.partner(it, "flip")
            self.assertEqual(s.sentence, it.sentence)
            self.assertEqual(s.question_voice, it.question_voice)
            self.assertNotEqual(s.role_asked, it.role_asked)
            self.assertEqual(s.answer, it.distractor)

    def test_repair_partner_asks_the_same_role_in_the_other_voice(self):
        for it in some_items()[:40]:
            s = orq.partner(it, "repair")
            self.assertEqual(s.sentence, it.sentence)
            self.assertNotEqual(s.question_voice, it.question_voice)
            self.assertEqual(s.role_asked, it.role_asked)
            self.assertEqual(s.answer, it.answer)
            self.assertNotEqual(s.question, it.question)

    def test_cross_partners_use_disjoint_names_and_the_same_verb(self):
        names = ors.NAMES[:6]
        expect = {"repair_x": ("same", "other"), "control_x": ("same", "same"), "flip_x": ("other", "same")}
        pairs = set()
        for it in some_items(6)[:60]:
            for mode, (role, voice) in expect.items():
                s = orq.partner(it, mode, names)
                self.assertFalse({s.agent, s.patient} & {it.agent, it.patient})
                self.assertTrue({s.agent, s.patient} <= set(names))
                self.assertEqual((s.verb, s.construction), (it.verb, it.construction))
                self.assertEqual(s.role_asked == it.role_asked, role == "same")
                self.assertEqual(s.question_voice == it.question_voice, voice == "same")
            self.assertEqual(orq.partner(it, "flip_x", names), orq.partner(it, "flip_x", names))
            pairs.add(orq.foreign_names(it, names))
        self.assertGreater(len(pairs), 3)  # foreign pairs vary across items

    def test_shared_tail(self):
        self.assertEqual(orq.shared_tail([1, 2, 3, 9, 8], [5, 3, 9, 8]), 3)
        self.assertEqual(orq.shared_tail([1, 2], [3, 4]), 0)
        self.assertEqual(orq.shared_tail([7, 7, 7], [7, 7, 7], cap=2), 2)

    def test_label_order(self):
        self.assertEqual(orq.label_order(["end-1", "q_first", "end-3", "period", "end-2"]),
                         ["period", "q_first", "end-3", "end-2", "end-1"])


class TestAggregate(unittest.TestCase):

    def test_ratio_of_means_and_flip_rate(self):
        r1 = orq.QuestionRecord("a", "f1", "active", "agent", "active", "flip", "s", 4.0, -4.0,
                                ld_patched={"end-1": [4.0, 0.0, -4.0]})
        r2 = orq.QuestionRecord("b", "f2", "active", "agent", "active", "flip", "s", 2.0, -2.0,
                                ld_patched={"end-1": [2.0, 2.0, -2.0]})
        agg = orq.aggregate([r1, r2])
        cell = agg["flip/agent/active"]["end-1"]
        self.assertEqual(cell["n"], 2)
        self.assertAlmostEqual(cell["ratio_of_means"][0], 0.0)
        self.assertAlmostEqual(cell["ratio_of_means"][1], 4.0 / 12.0)
        self.assertAlmostEqual(cell["ratio_of_means"][2], 1.0)
        self.assertEqual(cell["flip_rate"], [0.0, 0.0, 1.0])
        self.assertAlmostEqual(cell["mean_shift"][1], (4.0 + 0.0) / 2)
        self.assertIn("flip/ALL", agg)


@unittest.skipUnless(HAVE_TORCH, "torch/transformers not installed")
class TestQuestionSweep(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from test_chat_template_patching import tiny_chat_model_and_tokenizer
        from test_hf_scoring import tiny_model_and_tokenizer

        items = some_items(6)
        cls.items = [i for i in items if i.split == items[0].split][:16]
        cls.names = ors.NAMES[:6]
        vocab_items = items + [orq.partner(i, m, cls.names) for i in items for m in orq.ALL_MODES]
        cls.raw = tiny_model_and_tokenizer(vocab_items)
        cls.chat = tiny_chat_model_and_tokenizer(vocab_items)

    def _check(self, model, tok, chat, final_token, modes=orq.ALL_MODES):
        last = model.config.n_layer - 1
        for mode in modes:
            recs, skipped = orq.sweep(model, tok, self.items, mode=mode, batch_size=5,
                                      require_selection=False, chat_template=chat, names=self.names)
            self.assertEqual(len(recs), len(self.items), skipped)
            for rec in recs:
                if mode in orq.MODES:
                    # identical prefix: patching the last sentence token changes nothing
                    for v in rec.ld_patched["period"]:
                        self.assertAlmostEqual(v, rec.ld_base, places=4)
                else:
                    self.assertNotIn("period", rec.ld_patched)
                    self.assertFalse(math.isnan(rec.ld_source_own))
                # final position at the last layer reproduces the source run (base names)
                self.assertAlmostEqual(rec.ld_patched["end-1"][last], rec.ld_source, places=3)
                self.assertEqual(rec.tail_tokens[0].strip(), final_token)
                # any earlier position at the last layer has no effect
                for lab, vals in rec.ld_patched.items():
                    if lab != "end-1":
                        self.assertAlmostEqual(vals[last], rec.ld_base, places=4, msg=f"{mode} {lab}")
                self.assertIn("q_first", rec.ld_patched)
                self.assertGreaterEqual(len(rec.tail_tokens), 2)

    def test_raw_mode_exact_checks(self):
        model, tok = self.raw
        self._check(model, tok, chat=False, final_token=":")

    def test_chat_mode_exact_checks(self):
        model, tok = self.chat
        self._check(model, tok, chat=True, final_token="model")

    def test_selection_filters(self):
        model, tok = self.raw
        recs, _ = orq.sweep(model, tok, self.items, mode="flip", require_selection=True)
        for r in recs:
            self.assertTrue(r.ld_base > 0 and r.ld_source < 0)
        recs, _ = orq.sweep(model, tok, self.items, mode="repair", require_selection=True)
        for r in recs:
            self.assertTrue(r.ld_base < 0 and r.ld_source > 0)
        for mode, ok in (("repair_x", lambda r: r.ld_base < 0 and r.ld_source_own > 0),
                         ("control_x", lambda r: r.ld_base < 0),
                         ("flip_x", lambda r: r.ld_base > 0 and r.ld_source_own > 0)):
            recs, _ = orq.sweep(model, tok, self.items, mode=mode, require_selection=True, names=self.names)
            for r in recs:
                self.assertTrue(ok(r), mode)


if __name__ == "__main__":
    unittest.main()
