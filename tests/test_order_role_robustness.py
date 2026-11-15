"""OTR-05 robustness variants and the shared chat-template kwargs."""

import importlib.util
import unittest

from relational_lab.experiments import order_role_robustness as orr
from relational_lab.experiments import order_role_stimuli as ors
from relational_lab.instrumentation import chat

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "tokenizers"))


class TestVariants(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.items = orr.make_items(n_events=3)

    def test_counts_and_ids(self):
        n_per_role = 3 * len(ors.MAIN_CONSTRUCTIONS)
        self.assertEqual(len(self.items), n_per_role * len(orr.VARIANTS))
        self.assertEqual(len({i.item_id for i in self.items}), len(self.items))

    def test_sentences_names_and_answers_are_the_v13_ones(self):
        base = {(i.event_id, i.construction, i.role_asked): i for i in ors.generate_items()
                if i.question_voice == "active"}
        for it in self.items:
            b = base[(it.event_id, it.construction, it.role_asked)]
            self.assertEqual((it.sentence, it.answer, it.distractor, it.family_id, it.split),
                             (b.sentence, b.answer, b.distractor, b.family_id, b.split))
            self.assertTrue(it.prompt.startswith(ors.PROMPT_PREFIX + it.sentence + "\n"))

    def test_formats(self):
        for it in self.items:
            v = orr.VARIANT_BY_NAME[it.item_id.rsplit(":", 1)[1]]
            self.assertEqual(it.role_asked, v.role)
            if v.fmt == "cloze":
                self.assertNotIn("Question:", it.prompt)
                self.assertTrue(it.prompt.endswith(" was"))
            else:
                self.assertTrue(it.prompt.endswith("?\nAnswer:"))
            self.assertNotIn("{", it.prompt)

    def test_originals_reproduce_v13_questions(self):
        it = next(i for i in self.items if i.item_id.endswith("P_orig_active") and i.construction == "active")
        self.assertEqual(it.question, ors.render_question("patient", "active", it.lemma, it.verb))
        it = next(i for i in self.items if i.item_id.endswith("A_orig_passive"))
        self.assertEqual(it.question, ors.render_question("agent", "passive", it.lemma, it.verb))

    def test_summarize_subject_choice(self):
        its = [i for i in self.items if i.item_id.endswith("P_orig_active")]
        # wrong everywhere: chooses the agent; subject is the agent in active / object cleft
        margins = {i.item_id: -1.0 for i in its}
        s = orr.summarize(its, margins)["P_orig_active"]
        self.assertEqual(s["accuracy"], 0.0)
        self.assertEqual(s["by_construction"]["active"]["chose_subject"], 1.0)
        self.assertEqual(s["by_construction"]["passive"]["chose_subject"], 0.0)
        self.assertEqual(s["by_construction"]["object_cleft"]["chose_subject"], 1.0)
        self.assertEqual(s["by_construction"]["by_cleft"]["chose_subject"], 0.0)


class TestChatKwargs(unittest.TestCase):

    def tearDown(self):
        chat.configure(no_thinking=False)

    def test_template_kwargs_and_prefix(self):
        out = chat.configure(template_kwargs=["reasoning_strength=low", "flag=true"], assistant_prefix=" to=user")
        self.assertEqual(chat.CHAT_TEMPLATE_KWARGS, {"reasoning_strength": "low", "flag": True})
        self.assertEqual(chat.assistant_prefix(), " to=user")
        self.assertEqual(out["_assistant_prefix"], " to=user")
        with self.assertRaises(ValueError):
            chat.configure(template_kwargs=["novalue"])
        chat.configure()
        self.assertEqual(chat.assistant_prefix(), "")

    @unittest.skipUnless(HAVE_TORCH, "torch/transformers not installed")
    def test_assistant_prefix_in_both_code_paths(self):
        from test_chat_template_patching import tiny_chat_model_and_tokenizer

        from relational_lab.instrumentation.hf_scoring import _prompt_ids
        from relational_lab.instrumentation.patching import chat_text, encode_batch, token_index

        items = ors.generate_items(names=ors.NAMES[:4])[:6]
        _, tok = tiny_chat_model_and_tokenizer(items)
        tok.add_tokens(["FINAL"])
        fin = tok.convert_tokens_to_ids("FINAL")
        chat.configure()
        base_ids = _prompt_ids(tok, items[0].prompt, True)
        chat.configure(assistant_prefix=" FINAL")
        text, at = chat_text(tok, items[0].prompt)
        self.assertTrue(text.endswith(" FINAL"))
        ids = _prompt_ids(tok, items[0].prompt, True)
        self.assertEqual(ids[-1], fin)
        self.assertEqual(ids[:-1], base_ids)
        b_ids, _, lens, offs = encode_batch(tok, [it.prompt for it in items], tok.pad_token_id, chat_template=True)
        for r, it in enumerate(items):
            self.assertEqual(b_ids[r, :lens[r]].tolist(), _prompt_ids(tok, it.prompt, True))
            n = it.names[0]
            i = token_index(offs[r], n.prompt_char_start, n.prompt_char_end)
            self.assertEqual(tok.decode([int(b_ids[r, i])]).strip(), n.name)

    def test_configure(self):
        self.assertEqual(chat.configure(no_thinking=False), {})
        self.assertEqual(chat.configure(no_thinking=True), {"enable_thinking": False})
        self.assertEqual(chat.CHAT_TEMPLATE_KWARGS, {"enable_thinking": False})

    @unittest.skipUnless(HAVE_TORCH, "torch/transformers not installed")
    def test_kwargs_reach_the_template_in_both_code_paths(self):
        from test_chat_template_patching import tiny_chat_model_and_tokenizer

        from relational_lab.instrumentation.hf_scoring import _prompt_ids
        from relational_lab.instrumentation.patching import chat_text, encode_batch

        items = ors.generate_items(names=ors.NAMES[:4])[:4]
        _, tok = tiny_chat_model_and_tokenizer(items)
        tok.add_tokens(["NOTHINK"])
        tok.chat_template = ("{% if enable_thinking is defined and not enable_thinking %}NOTHINK{% endif %}"
                             + tok.chat_template)
        nothink = tok.convert_tokens_to_ids("NOTHINK")
        chat.configure(no_thinking=False)
        self.assertNotIn("NOTHINK", chat_text(tok, items[0].prompt)[0])
        self.assertNotIn(nothink, _prompt_ids(tok, items[0].prompt, True))
        chat.configure(no_thinking=True)
        self.assertIn("NOTHINK", chat_text(tok, items[0].prompt)[0])
        self.assertIn(nothink, _prompt_ids(tok, items[0].prompt, True))
        ids, _, _, _ = encode_batch(tok, [items[0].prompt], tok.pad_token_id, chat_template=True)
        self.assertIn(nothink, ids[0].tolist())


if __name__ == "__main__":
    unittest.main()
