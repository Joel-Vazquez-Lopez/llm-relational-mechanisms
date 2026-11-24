"""Test-split plumbing: split specs, test items in every pool, OTR-09 with frozen heads."""
import importlib.util
import json
from pathlib import Path

import pytest

from relational_lab.experiments import order_role_stimuli as ors
from relational_lab.experiments.splits import in_split, parse

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_parse_and_membership():
    assert parse("dev") == ("dev", -1) and parse("test2") == ("test", 2)
    with pytest.raises(ValueError):
        parse("test")
    items = ors.generate_items()
    t0 = [i for i in items if in_split(i, "test0")]
    assert t0 and all(i.split == "test" and i.test_set == 0 for i in t0)
    assert not {i.family_id for i in t0} & {i.family_id for i in items if in_split(i, "dev")}


def test_pools_respect_split():
    from otr_generate_check import prepared_prompts
    from relational_lab.experiments import bx_adapter as bxa, xl_adapter as xla
    assert all(it.test_set == 1 for it, _ in prepared_prompts("otr", 2026, 32, split="test1"))
    assert all(it.split == "dev" for it, _ in prepared_prompts("otr", 2026, 32))
    assert all(i.test_set == 0 for i in bxa.select_bases("bx-transitive", "test0", 4, ["active"], 2026))
    assert all(b.test_set == 3 for b, _ in xla.repair_pairs("es", 2026, ["active"], "", split="test3"))
    o9 = _load("otr09_answer_patching")
    tr = o9.make_pairs(2026, "Who was someone {past} by?", "Who was the {ing} done by?", "", ["active"], 3,
                       split="test0")
    assert tr and all(b.test_set == 0 and d.test_set == 0 for b, _, d in tr)
    with pytest.raises(ValueError):
        o9.make_pairs(2026, "Who was someone {pp} by?", "x", "", ["ud_active"], 3, stimuli="ud-en", split="test0")


def test_otr09_frozen_heads_on_test(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from test_logit_lens import _tiny
    _tiny(monkeypatch)
    o9 = _load("otr09_answer_patching")
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"head_scan": {"top20": [f"{li}.{h}" for li in range(3) for h in range(4)]}}))
    o9.main(["--model", "tiny", "--dtype", "float32", "--n-per", "3", "--layer-stride", "1", "--head-from", "0.6",
             "--top-k", "12", "--min-gap", "1e-4", "--split", "test0", "--skip-scan", "--rank-from", str(prior),
             "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr09_tiny_test0_*.json")).read_text())
    assert r["status"] == "TEST_SET_OPENED" and r["skip_scan"]
    assert abs(r["circuit_eval"]["by_k"]["12"]["top"]["repair"] - 1.0) < 1e-3


def test_otr06_prompts_split():
    o6 = _load("otr06_logit_lens")
    pr = o6.build_prompts(2026, [("agent", "Who was someone {past} by?")], "", 2, split="test0")
    assert pr and all(it.test_set == 0 for _, it in pr)
    assert all(it.split == "dev" for _, it in o6.build_prompts(2026, [("agent", "Who {past} someone?")], "", 2))
