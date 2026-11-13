"""Scorer tests on a tiny randomly initialized GPT-2 with a word-level tokenizer.

Skipped when torch/transformers are not installed (the CI job is dependency-free).
No weights are downloaded.
"""

import importlib.util
import math
import unittest

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "tokenizers"))


def tiny_model_and_tokenizer(items):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    words = set()
    for it in items:
        words.update(it.prompt.replace("\n", " ").replace("?", " ? ").replace(".", " . ").replace(":", " : ").split())
    vocab = {w: i for i, w in enumerate(["[UNK]", "[PAD]", *sorted(words)])}
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]")
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=len(vocab), n_positions=64, n_embd=32, n_layer=2, n_head=2,
                     bos_token_id=None, eos_token_id=None)
    model = GPT2LMHeadModel(cfg).eval()
    return model, tok


@unittest.skipUnless(HAVE_TORCH, "torch/transformers not installed")
class TestHFScorer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from relational_lab.experiments import order_role_stimuli as ors

        cls.items = ors.generate_items(names=ors.NAMES[:4])[:96]
        cls.model, cls.tok = tiny_model_and_tokenizer(cls.items)

    def test_names_are_single_tokens(self):
        from relational_lab.instrumentation.hf_scoring import single_token_checker

        check = single_token_checker(self.tok)
        self.assertTrue(all(check(it.answer) for it in self.items))

    def test_matches_manual_log_probability(self):
        import torch

        from relational_lab.instrumentation.hf_scoring import HFScorer

        it = self.items[5]
        score = HFScorer(self.model, self.tok, batch_size=4)([it])[0]
        for name, got in ((it.answer, score.logprob_answer), (it.distractor, score.logprob_distractor)):
            ids = self.tok.encode(it.prompt, add_special_tokens=True)
            cont = self.tok.encode(" " + name, add_special_tokens=False)
            with torch.no_grad():
                logits = self.model(torch.tensor([ids + cont])).logits[0]
            lp = torch.log_softmax(logits.float(), -1)
            expected = sum(float(lp[len(ids) - 1 + k, t]) for k, t in enumerate(cont))
            self.assertTrue(math.isclose(got, expected, rel_tol=1e-5, abs_tol=1e-5), (got, expected))

    def test_batching_and_padding_do_not_change_scores(self):
        from relational_lab.instrumentation.hf_scoring import HFScorer

        one = HFScorer(self.model, self.tok, batch_size=1)(self.items[:24])
        many = HFScorer(self.model, self.tok, batch_size=24)(self.items[:24])
        for a, b in zip(one, many):
            self.assertEqual(a.item_id, b.item_id)
            self.assertAlmostEqual(a.logprob_answer, b.logprob_answer, places=4)
            self.assertAlmostEqual(a.logprob_distractor, b.logprob_distractor, places=4)
            self.assertEqual(a.answer_is_top1, b.answer_is_top1)

    def test_end_to_end_summary(self):
        from relational_lab.experiments import order_role_behavior as beh
        from relational_lab.experiments import order_role_stimuli as ors
        from relational_lab.instrumentation.hf_scoring import HFScorer

        items = ors.generate_items(names=ors.NAMES[:4])
        items = [i for i in items if i.family_id == items[0].family_id]
        s = beh.summarize(items, HFScorer(self.model, self.tok, batch_size=32)(items), n_boot=50)
        self.assertEqual(s["n_items"], len(items))
        self.assertIn("passive/agent", s["by_construction_role"])
        self.assertTrue(0.0 <= s["overall_choice_accuracy"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
