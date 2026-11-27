"""Scoring (HFScorer, OTR-01/05/15) and patching (OTR-02/03/04/09/...) must build the
same token sequence for the same prompt under every chat setting a runner can set:
raw prompts, chat template, template keyword arguments (``enable_thinking``),
assistant prefix, system prompt, a template that has no system role, and both
settings together. They must also feed inputs to the same device.

Tiny random GPT-2 with a word-level tokenizer; no weights are downloaded.
Skipped without torch/transformers/tokenizers.
"""

import importlib.util
import unittest

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "tokenizers"))

WITH_SYSTEM = (
    "{{ bos_token }}{% for m in messages %}<start_of_turn>{{ m['role'] }}\n"
    "{{ m['content'] }}<end_of_turn>\n{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n"
    "{% if enable_thinking is defined and not enable_thinking %}<nothink>{% endif %}{% endif %}"
)
NO_SYSTEM = (
    "{{ bos_token }}{% for m in messages %}"
    "{% if m['role'] == 'system' %}{{ raise_exception('no system role') }}{% endif %}"
    "<start_of_turn>{{ m['role'] }}\n{{ m['content'] }}<end_of_turn>\n{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)
SYSTEM_TEXT = "You are careful ."
PREFIX = "<answer>"

# (label, chat_template, template, no_thinking, assistant_prefix, system_prompt)
CONFIGS = [
    ("raw", False, WITH_SYSTEM, False, "", ""),
    ("chat", True, WITH_SYSTEM, False, "", ""),
    ("chat+no_thinking", True, WITH_SYSTEM, True, "", ""),
    ("chat+prefix", True, WITH_SYSTEM, False, PREFIX, ""),
    ("chat+system", True, WITH_SYSTEM, False, "", SYSTEM_TEXT),
    ("chat+system+prefix", True, WITH_SYSTEM, False, PREFIX, SYSTEM_TEXT),
    ("chat+system+no_thinking+prefix", True, WITH_SYSTEM, True, PREFIX, SYSTEM_TEXT),
    ("no-system-role+system", True, NO_SYSTEM, False, "", SYSTEM_TEXT),
    ("no-system-role+system+prefix", True, NO_SYSTEM, False, PREFIX, SYSTEM_TEXT),
]


def tiny_chat_setup(items):
    from test_hf_scoring import tiny_model_and_tokenizer
    from transformers import GPT2Config, GPT2LMHeadModel

    import torch

    _, tok = tiny_model_and_tokenizer(items)
    tok.add_special_tokens({"bos_token": "<bos>", "additional_special_tokens":
                            ["<start_of_turn>", "<end_of_turn>", "<nothink>", PREFIX]})
    tok.add_tokens(["user", "model", "system", "You", "are", "careful"])
    torch.manual_seed(3)
    cfg = GPT2Config(vocab_size=len(tok), n_positions=96, n_embd=32, n_layer=2, n_head=2,
                     bos_token_id=None, eos_token_id=None)
    return GPT2LMHeadModel(cfg).eval(), tok


@unittest.skipUnless(HAVE_TORCH, "torch/transformers not installed")
class TestScoringPatchingConsistency(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from relational_lab.experiments import order_role_stimuli as ors

        all_items = ors.generate_items(names=ors.NAMES[:6])
        cls.all_items = all_items
        cls.items = [i for i in all_items if i.split == all_items[0].split][:8]
        cls.model, cls.tok = tiny_chat_setup(all_items)

    def tearDown(self):
        from relational_lab.instrumentation import chat

        chat.configure()
        chat.SYSTEM_PROMPT[0] = ""

    def _set(self, template, no_thinking, prefix, system):
        from relational_lab.instrumentation import chat

        self.tok.chat_template = template
        chat.configure(no_thinking=no_thinking, assistant_prefix=prefix)
        chat.SYSTEM_PROMPT[0] = system

    def test_scoring_and_patching_build_identical_sequences(self):
        from relational_lab.instrumentation.hf_scoring import _prompt_ids
        from relational_lab.instrumentation.patching import encode_batch

        prompts = [it.prompt for it in self.items]
        for label, chat_on, template, no_think, prefix, system in CONFIGS:
            with self.subTest(config=label):
                self._set(template, no_think, prefix, system)
                ids, _, lengths, _ = encode_batch(self.tok, prompts, self.tok.pad_token_id, chat_template=chat_on)
                for r, p in enumerate(prompts):
                    patch_ids = [int(x) for x in ids[r, : lengths[r]]]
                    self.assertEqual(_prompt_ids(self.tok, p, chat_on), patch_ids)
                    text = self.tok.decode(patch_ids)
                    if chat_on:
                        self.assertEqual("careful" in text, bool(system))
                        self.assertEqual("<nothink>" in text, no_think)
                        self.assertEqual(text.rstrip().endswith(PREFIX), bool(prefix))

    def test_item_spans_still_map_onto_the_prompt_words(self):
        from relational_lab.instrumentation.patching import encode_batch, token_index

        prompts = [it.prompt for it in self.items]
        for label, chat_on, template, no_think, prefix, system in CONFIGS:
            with self.subTest(config=label):
                self._set(template, no_think, prefix, system)
                ids, _, _, offsets = encode_batch(self.tok, prompts, self.tok.pad_token_id, chat_template=chat_on)
                for r, it in enumerate(self.items):
                    for n in it.names:
                        ci = token_index(offsets[r], n.prompt_char_start, n.prompt_char_end)
                        self.assertEqual(self.tok.decode([int(ids[r, ci])]).strip(), n.name)

    def test_scorer_and_patching_baseline_agree(self):
        """Single-token names: log p(answer) - log p(distractor) from HFScorer equals the
        unpatched logit difference the patching sweep records."""
        from relational_lab.experiments import order_role_patching as orp
        from relational_lab.instrumentation.hf_scoring import HFScorer

        for label, chat_on, template, no_think, prefix, system in CONFIGS:
            with self.subTest(config=label):
                self._set(template, no_think, prefix, system)
                scores = HFScorer(self.model, self.tok, batch_size=4, use_chat_template=chat_on)(self.items)
                records, skipped = orp.sweep(self.model, self.tok, self.items, batch_size=4,
                                             require_correct=False, chat_template=chat_on)
                self.assertEqual(skipped, {})
                for s, rec in zip(scores, records):
                    self.assertAlmostEqual(s.logprob_answer - s.logprob_distractor, rec.ld_base, places=4)

    def test_scorer_feeds_inputs_to_the_embedding_device(self):
        """A model whose first parameter is not on the embedding's device (e.g. a vision tower
        placed elsewhere by a device map): the scorer must use the embedding device, like
        ResidualPatcher and load_for_run."""
        import torch

        from relational_lab.instrumentation.hf_scoring import HFScorer, input_device
        from relational_lab.instrumentation.patching import ResidualPatcher

        self._set(WITH_SYSTEM, False, "", "")
        model, _ = tiny_chat_setup(self.all_items)  # same vocabulary as self.tok
        model.register_parameter("a_first_param_elsewhere", torch.nn.Parameter(torch.zeros(1, device="meta")))
        self.assertEqual(next(model.parameters()).device.type, "meta")
        self.assertEqual(input_device(model), ResidualPatcher(model).device)
        self.assertEqual(input_device(model).type, "cpu")
        scores = HFScorer(model, self.tok, batch_size=4)(self.items[:4])
        self.assertEqual(len(scores), 4)


if __name__ == "__main__":
    unittest.main()
