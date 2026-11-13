import importlib.util
import math
import random
import unittest

from relational_lab.experiments import order_role_stimuli as ors
from relational_lab.experiments import order_role_transplant as otr

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "tokenizers"))


def _bases(n=200, names=ors.NAMES):
    items = ors.generate_items(names=names)
    return [i for i in items if i.construction in ors.MAIN_CONSTRUCTIONS][:n]


class TestDesign(unittest.TestCase):

    def test_sources_use_foreign_names_and_same_verb(self):
        for t in otr.make_transplants(_bases(80)):
            self.assertFalse({t.source.agent, t.source.patient} & {t.base.agent, t.base.patient})
            self.assertEqual(t.source.verb, t.base.verb)

    def test_foreign_names_are_counterbalanced(self):
        from collections import Counter
        tr = otr.make_transplants(_bases(60))
        by_base = {}
        for t in tr:
            by_base.setdefault(t.base.item_id, []).append(t)
        for bid, ts in by_base.items():
            as_agent = Counter(t.source.agent for t in ts)
            as_patient = Counter(t.source.patient for t in ts)
            self.assertEqual(as_agent, as_patient, bid)
            self.assertEqual(len(as_agent), 2, bid)
            self.assertEqual(len(ts), 2 * len(ors.MAIN_CONSTRUCTIONS) * 4)

    def test_left_context_null_is_zero_for_identity_only_effects(self):
        # y depends only on the foreign name's identity, never on role:
        # counterbalancing must drive the active/passive first-name role slope to exactly 0.
        tr = otr.make_transplants(_bases(120))
        bias = {n: (hash(n) % 7) - 3.0 for n in ors.NAMES}
        recs = []
        for t in tr:
            slot_name = t.source.names[0].name if t.j == "first_name" else t.source.names[1].name
            recs.append(otr.TransplantRecord(t.base.item_id, t.base.family_id, t.base.construction,
                                             t.source.construction, t.k, t.j, t.base.role_asked, t.x, 0.0,
                                             [bias[slot_name]]))
        self.assertAlmostEqual(otr.left_context_null(recs)[0], 0.0, places=9)
        prof = otr.fit_profile(recs, n_boot=0)
        for c in otr.CODES:
            self.assertAlmostEqual(prof[c]["coef"][0], 0.0, places=9)

    def test_predictors_are_uncorrelated(self):
        tr = otr.make_transplants(_bases(400))
        for key, r in otr.predictor_correlations(tr).items():
            self.assertAlmostEqual(r, 0.0, places=9, msg=key)

    def test_predictor_signs(self):
        base = ors.build_item("Leon", "Mira", "praise", "praised", "active", "agent", "active")
        src_passive = ors.build_item("Anna", "Tom", "praise", "praised", "passive", "agent", "active")
        # Base slot 1 = Leon: first, subject, agent; agent question wants all three.
        # Source passive slot 1 = Tom... no: passive is "Tom was praised by Anna", slot 1 = Tom (patient, subject, first).
        x = otr.predictors(base, "first_name", src_passive, "first_name")
        self.assertEqual(x, {"order": 0, "function": 0, "role": -1})
        src_ocleft = ors.build_item("Anna", "Tom", "praise", "praised", "object_cleft", "agent", "active")
        # "It was Tom that Anna praised": slot 1 = Tom (first, object, patient)
        x = otr.predictors(base, "first_name", src_ocleft, "first_name")
        self.assertEqual(x, {"order": 0, "function": -1, "role": -1})
        # Same code transplanted: all zeros.
        src_active = ors.build_item("Anna", "Tom", "praise", "praised", "active", "agent", "active")
        self.assertEqual(otr.predictors(base, "first_name", src_active, "first_name"),
                         {"order": 0, "function": 0, "role": 0})
        # Patient question flips what counts as a match.
        base_p = ors.build_item("Leon", "Mira", "praise", "praised", "active", "patient", "active")
        x = otr.predictors(base_p, "first_name", src_passive, "first_name")
        self.assertEqual(x, {"order": 0, "function": 0, "role": 1})

    def test_ols_recovers_planted_coefficients(self):
        rng = random.Random(1)
        rows = []
        for _ in range(500):
            x = [rng.choice((-1, 0, 1)) for _ in range(3)]
            y = 0.3 + 2.0 * x[0] - 1.0 * x[1] + 0.5 * x[2] + rng.gauss(0, 0.01)
            rows.append((x, y))
        b = otr.ols(rows)
        for got, want in zip(b, (0.3, 2.0, -1.0, 0.5)):
            self.assertAlmostEqual(got, want, places=2)

    def test_fit_profile_recovers_a_planted_role_code(self):
        tr = otr.make_transplants(_bases(240))
        rng = random.Random(0)
        recs = []
        for t in tr:
            x = t.x
            # Layers 0-1: order code; layer 2: role code.
            ys = [1.5 * x["order"], 1.5 * x["order"], 2.0 * x["role"]]
            recs.append(otr.TransplantRecord(t.base.item_id, t.base.family_id, t.base.construction,
                                             t.source.construction, t.k, t.j, t.base.role_asked, x, 0.0,
                                             [v + rng.gauss(0, 0.05) for v in ys]))
        prof = otr.fit_profile(recs, n_boot=50)
        self.assertAlmostEqual(prof["order"]["coef"][0], 1.5, places=1)
        self.assertAlmostEqual(prof["role"]["coef"][0], 0.0, places=1)
        self.assertAlmostEqual(prof["role"]["coef"][2], 2.0, places=1)
        self.assertAlmostEqual(prof["order"]["coef"][2], 0.0, places=1)
        lo, hi = prof["role"]["ci95"][2]
        self.assertLessEqual(lo, prof["role"]["coef"][2] + 1e-9)
        self.assertGreaterEqual(hi, prof["role"]["coef"][2] - 1e-9)
        self.assertLess(abs(lo - 2.0), 0.1)
        self.assertLess(abs(hi - 2.0), 0.1)


@unittest.skipUnless(HAVE_TORCH, "torch/transformers not installed")
class TestTransplantSweep(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from test_hf_scoring import tiny_model_and_tokenizer

        names = ors.NAMES[:6]
        cls.bases = _bases(12, names=names)
        cls.transplants = otr.make_transplants(cls.bases, names=names)
        cls.model, cls.tok = tiny_model_and_tokenizer(ors.generate_items(names=names))

    def test_last_layer_name_transplant_has_no_effect(self):
        recs, skipped = otr.sweep(self.model, self.tok, self.transplants, batch_size=16, require_base_correct=False)
        self.assertEqual(skipped, {})
        self.assertEqual(len(recs), len(self.transplants))
        for r in recs:
            self.assertAlmostEqual(r.y[-1], 0.0, places=4)

    def test_self_transplant_has_no_effect_at_any_layer(self):
        selfs = [otr.Transplant(b, b, k, k) for b in self.bases for k in otr.SLOTS]
        recs, _ = otr.sweep(self.model, self.tok, selfs, batch_size=8, require_base_correct=False)
        for r in recs:
            for v in r.y:
                self.assertAlmostEqual(v, 0.0, places=4)

    def test_batch_invariance(self):
        a, _ = otr.sweep(self.model, self.tok, self.transplants[:12], batch_size=1, require_base_correct=False)
        b, _ = otr.sweep(self.model, self.tok, self.transplants[:12], batch_size=12, require_base_correct=False)
        for x, y in zip(a, b):
            for u, v in zip(x.y, y.y):
                self.assertAlmostEqual(u, v, places=3)

    def test_profile_runs_end_to_end(self):
        recs, _ = otr.sweep(self.model, self.tok, self.transplants, batch_size=16, require_base_correct=False)
        prof = otr.fit_profile(recs, n_boot=20)
        self.assertEqual(len(prof["role"]["coef"]), self.model.config.n_layer)
        self.assertTrue(all(math.isfinite(v) for v in prof["role"]["coef"]))


if __name__ == "__main__":
    unittest.main()
