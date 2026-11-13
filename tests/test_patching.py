"""Exact-answer checks for residual patching on a tiny random GPT-2 (skipped without torch)."""

import importlib.util
import math
import unittest

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "tokenizers"))


class TestTokenIndex(unittest.TestCase):

    def test_last_overlapping_token(self):
        from relational_lab.instrumentation.patching import token_index

        offsets = [(0, 0), (0, 4), (5, 7), (7, 9), (10, 11)]  # BOS, word, two-piece word, punct
        self.assertEqual(token_index(offsets, 0, 4), 1)
        self.assertEqual(token_index(offsets, 5, 9), 3)
        self.assertEqual(token_index(offsets, 10, 11), 4)
        with self.assertRaises(ValueError):
            token_index(offsets, 20, 21)


@unittest.skipUnless(HAVE_TORCH, "torch/transformers not installed")
class TestResidualPatching(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from test_hf_scoring import tiny_model_and_tokenizer

        from relational_lab.experiments import order_role_stimuli as ors

        cls.ors = ors
        all_items = ors.generate_items(names=ors.NAMES[:4])
        cls.items = [i for i in all_items if i.split == all_items[0].split][:24]
        cls.model, cls.tok = tiny_model_and_tokenizer(all_items)
        cls.model.config.n_layer  # noqa: B018

    def _batch(self, prompts):
        from relational_lab.instrumentation.patching import encode_batch

        return encode_batch(self.tok, prompts, self.tok.pad_token_id)

    def test_self_patch_changes_nothing(self):
        import torch

        from relational_lab.instrumentation.patching import ResidualPatcher

        p = ResidualPatcher(self.model)
        ids, mask, lengths, _ = self._batch([i.prompt for i in self.items[:6]])
        store, logits = p.capture(ids, mask)
        for layer in range(p.n_layers):
            pos = [n - 3 for n in lengths]
            vecs = store[layer][torch.arange(6), torch.tensor(pos)]
            patched = p.run_patched(ids, mask, layer, pos, vecs)
            self.assertTrue(torch.allclose(patched, logits, atol=1e-5))

    def test_final_position_last_layer_reproduces_source_logits(self):
        import torch

        from relational_lab.instrumentation.patching import ResidualPatcher

        p = ResidualPatcher(self.model)
        bases = self.items[:6]
        sources = [self.ors.role_swapped(b) for b in bases]
        b_ids, b_mask, b_len, _ = self._batch([b.prompt for b in bases])
        s_ids, s_mask, s_len, _ = self._batch([s.prompt for s in sources])
        self.assertEqual(b_len, s_len)
        store, s_logits = p.capture(s_ids, s_mask)
        last = [n - 1 for n in b_len]
        vecs = store[p.n_layers - 1][torch.arange(6), torch.tensor(last)]
        patched = p.run_patched(b_ids, b_mask, p.n_layers - 1, last, vecs)
        for r, n in enumerate(last):
            self.assertTrue(torch.allclose(patched[r, n], s_logits[r, n], atol=1e-4))

    def test_earlier_position_last_layer_has_no_effect(self):
        import torch

        from relational_lab.instrumentation.patching import ResidualPatcher

        p = ResidualPatcher(self.model)
        bases = self.items[:6]
        sources = [self.ors.role_swapped(b) for b in bases]
        b_ids, b_mask, b_len, _ = self._batch([b.prompt for b in bases])
        s_ids, s_mask, _, _ = self._batch([s.prompt for s in sources])
        store, _ = p.capture(s_ids, s_mask)
        _, base_logits = p.capture(b_ids, b_mask)
        pos = [2] * 6  # a name-region token, read by nothing after the last block
        vecs = store[p.n_layers - 1][torch.arange(6), torch.tensor(pos)]
        patched = p.run_patched(b_ids, b_mask, p.n_layers - 1, pos, vecs)
        for r, n in enumerate(b_len):
            self.assertTrue(torch.allclose(patched[r, n - 1], base_logits[r, n - 1], atol=1e-5))

    def test_sweep_effects_hit_known_values(self):
        from relational_lab.experiments import order_role_patching as orp

        records, skipped = orp.sweep(self.model, self.tok, self.items, batch_size=8, require_correct=False)
        self.assertEqual(skipped, {})
        self.assertEqual(len(records), len(self.items))
        last = self.model.config.n_layer - 1
        for rec in records:
            if math.isnan(rec.effects["answer_cue"][last]):
                continue
            self.assertAlmostEqual(rec.effects["answer_cue"][last], 1.0, places=3)
            self.assertAlmostEqual(rec.effects["first_name"][last], 0.0, places=3)
            self.assertAlmostEqual(rec.effects["period"][last], 0.0, places=3)

    def test_sweep_is_batch_invariant(self):
        from relational_lab.experiments import order_role_patching as orp

        a, _ = orp.sweep(self.model, self.tok, self.items[:10], batch_size=1, require_correct=False)
        b, _ = orp.sweep(self.model, self.tok, self.items[:10], batch_size=10, require_correct=False)
        for x, y in zip(a, b):
            self.assertEqual(x.item_id, y.item_id)
            for lab in x.effects:
                for u, v in zip(x.effects[lab], y.effects[lab]):
                    if not (math.isnan(u) and math.isnan(v)):
                        self.assertAlmostEqual(u, v, places=3)

    def test_aggregate_shapes(self):
        from relational_lab.experiments import order_role_patching as orp

        records, _ = orp.sweep(self.model, self.tok, self.items, batch_size=8, require_correct=False)
        agg = orp.aggregate(records)
        self.assertIn("ALL", agg)
        cell = agg["ALL"]["answer_cue"]
        self.assertEqual(len(cell["mean_effect"]), self.model.config.n_layer)
        last = self.model.config.n_layer - 1
        self.assertAlmostEqual(cell["ratio_of_means"][last], 1.0, places=3)
        self.assertAlmostEqual(agg["ALL"]["first_name"]["ratio_of_means"][last], 0.0, places=3)
        self.assertEqual(cell["n"], len(records))
        for c in agg:
            if c in ("active", "object_cleft", "subject_cleft", "relative_active"):
                self.assertNotIn("by", agg[c])


if __name__ == "__main__":
    unittest.main()
