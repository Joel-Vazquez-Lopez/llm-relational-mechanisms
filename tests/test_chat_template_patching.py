"""Chat-template mode for OTR-02/03: spans map into the templated sequence, the
answer is scored without a leading space, and the exact patching checks still
hold. Uses a tiny word-level tokenizer with a Gemma-style template (skipped
without torch)."""

import importlib.util
import math
import unittest

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "tokenizers"))

CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<start_of_turn>{{ m['role'] }}\n"
    "{{ m['content'] }}<end_of_turn>\n{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)
N_TEMPLATE_PREFIX = 3  # <bos> <start_of_turn> user


def tiny_chat_model_and_tokenizer(items):
    from test_hf_scoring import tiny_model_and_tokenizer
    from transformers import GPT2Config, GPT2LMHeadModel

    import torch

    _, tok = tiny_model_and_tokenizer(items)
    tok.add_special_tokens({"bos_token": "<bos>",
                            "additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"]})
    tok.add_tokens(["user", "model"])
    tok.chat_template = CHAT_TEMPLATE
    torch.manual_seed(1)
    cfg = GPT2Config(vocab_size=len(tok), n_positions=64, n_embd=32, n_layer=2, n_head=2,
                     bos_token_id=None, eos_token_id=None)
    return GPT2LMHeadModel(cfg).eval(), tok


class FakeTok:
    """encode() only: 'Ma...' names share a first token without a leading space."""

    def encode(self, text, add_special_tokens=False):
        if text.startswith(" "):
            return [hash(text) % 10_000]
        return [1, hash(text) % 10_000] if text.startswith("Ma") else [hash(text) % 10_000]


class TestNameCheck(unittest.TestCase):

    def test_shared_first_answer_token_is_reported_only_in_chat_mode(self):
        from relational_lab.instrumentation.hf_scoring import check_names_for_patching

        names = ["Mark", "Maria", "Anna"]
        self.assertEqual(check_names_for_patching(FakeTok(), names, chat_template=False), [])
        problems = check_names_for_patching(FakeTok(), names, chat_template=True)
        self.assertEqual(len(problems), 1)
        self.assertIn("share their first answer token", problems[0])


@unittest.skipUnless(HAVE_TORCH, "torch/transformers not installed")
class TestChatTemplatePatching(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from relational_lab.experiments import order_role_stimuli as ors

        cls.ors = ors
        cls.names = ors.NAMES[:6]
        all_items = ors.generate_items(names=cls.names)
        cls.items = [i for i in all_items if i.split == all_items[0].split][:24]
        cls.model, cls.tok = tiny_chat_model_and_tokenizer(all_items)

    def test_spans_land_on_the_same_words_shifted_by_the_template(self):
        from relational_lab.instrumentation.patching import encode_batch, token_index

        prompts = [i.prompt for i in self.items]
        r_ids, _, r_len, r_off = encode_batch(self.tok, prompts, self.tok.pad_token_id)
        c_ids, _, c_len, c_off = encode_batch(self.tok, prompts, self.tok.pad_token_id, chat_template=True)
        model_tok = self.tok.convert_tokens_to_ids("model")
        for r, it in enumerate(self.items):
            self.assertEqual(int(c_ids[r, c_len[r] - 1]), model_tok)  # generation prompt is last
            for lab in self.ors.POSITION_LABELS:
                span = it.span(lab)
                if span is None:
                    continue
                ri, ci = token_index(r_off[r], *span), token_index(c_off[r], *span)
                self.assertEqual(ci, ri + N_TEMPLATE_PREFIX, lab)
                self.assertEqual(int(c_ids[r, ci]), int(r_ids[r, ri]), lab)
            for n in it.names:
                ci = token_index(c_off[r], n.prompt_char_start, n.prompt_char_end)
                self.assertEqual(self.tok.decode([int(c_ids[r, ci])]).strip(), n.name)

    def test_base_logit_difference_matches_apply_chat_template(self):
        import torch

        from relational_lab.experiments import order_role_patching as orp

        records, skipped = orp.sweep(self.model, self.tok, self.items[:6], batch_size=3,
                                     require_correct=False, chat_template=True)
        self.assertEqual(skipped, {})
        for rec, it in zip(records, self.items[:6]):
            ids = self.tok.apply_chat_template([{"role": "user", "content": it.prompt}],
                                               add_generation_prompt=True, tokenize=True)
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            with torch.no_grad():
                logits = self.model(input_ids=torch.tensor([list(ids)])).logits[0, -1]
            a = self.tok.encode(it.answer, add_special_tokens=False)[0]  # no leading space
            d = self.tok.encode(it.distractor, add_special_tokens=False)[0]
            self.assertAlmostEqual(rec.ld_base, float(logits[a] - logits[d]), places=4)

    def test_last_layer_prompt_positions_have_no_effect(self):
        from relational_lab.experiments import order_role_patching as orp

        records, _ = orp.sweep(self.model, self.tok, self.items, batch_size=8,
                               require_correct=False, chat_template=True)
        last = self.model.config.n_layer - 1
        for rec in records:
            for lab, eff in rec.effects.items():
                if not math.isnan(eff[last]):
                    # in chat mode even answer_cue is not the final position
                    self.assertAlmostEqual(eff[last], 0.0, places=3, msg=lab)

    def test_chat_sweep_is_batch_invariant(self):
        from relational_lab.experiments import order_role_patching as orp

        a, _ = orp.sweep(self.model, self.tok, self.items[:8], batch_size=1, require_correct=False, chat_template=True)
        b, _ = orp.sweep(self.model, self.tok, self.items[:8], batch_size=8, require_correct=False, chat_template=True)
        for x, y in zip(a, b):
            for lab in x.effects:
                for u, v in zip(x.effects[lab], y.effects[lab]):
                    if not (math.isnan(u) and math.isnan(v)):
                        self.assertAlmostEqual(u, v, places=3)

    def test_transplant_self_and_last_layer_have_no_effect(self):
        from relational_lab.experiments import order_role_transplant as otr

        bases = [i for i in self.items if i.construction in self.ors.MAIN_CONSTRUCTIONS][:6]
        transplants = otr.make_transplants(bases, names=self.names)[:24]
        recs, _ = otr.sweep(self.model, self.tok, transplants, batch_size=8,
                            require_base_correct=False, chat_template=True)
        self.assertTrue(recs)
        for r in recs:
            self.assertAlmostEqual(r.y[-1], 0.0, places=4)
        selfs = [otr.Transplant(b, b, k, k) for b in bases[:4] for k in otr.SLOTS]
        recs, _ = otr.sweep(self.model, self.tok, selfs, batch_size=4,
                            require_base_correct=False, chat_template=True)
        for r in recs:
            for v in r.y:
                self.assertAlmostEqual(v, 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
