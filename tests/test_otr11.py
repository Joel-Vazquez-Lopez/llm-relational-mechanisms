"""OTR-11: edit positions, steering-vector scaling, zero-edit identity and a tiny-model smoke run."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _o11():
    spec = importlib.util.spec_from_file_location("o11", ROOT / "scripts" / "otr11_role_steering.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_edit_spans():
    o11 = _o11()
    from relational_lab.experiments import order_role_stimuli as ors
    want = {"active": ["."], "passive": ["."], "object_cleft": ["praised", "."],
            "by_cleft": ["was", "praised", "."]}
    for cons, words in want.items():
        it = ors.build_item("Leon", "Mira", "praise", "praised", cons, "agent", "active")
        assert [it.prompt[a:b] for a, b in o11.edit_spans(it)] == words, cons


def test_label_holder():
    o11 = _o11()
    from relational_lab.experiments import order_role_stimuli as ors
    it = ors.build_item("Leon", "Mira", "praise", "praised", "passive", "agent", "active")
    assert (o11.label_holder(it, "subject"), o11.label_holder(it, "agent"), o11.label_holder(it, "first")) == \
        ("Mira", "Leon", "Mira")


def test_steer_vector_moves_margin_exactly():
    torch = pytest.importorskip("torch")
    o11 = _o11()
    torch.manual_seed(0)
    W = torch.randn(32, 5)
    v = o11.steer_vector(W, 2, 4, 3.0)
    x = torch.randn(32)
    before = x @ W[:, 2] - x @ W[:, 4]
    after = (x + v) @ W[:, 2] - (x + v) @ W[:, 4]
    assert abs(float(after - before) - 3.0) < 1e-4


def test_smoke_tiny_zero_edit_is_identity(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from test_logit_lens import _tiny
    _tiny(monkeypatch)
    o11 = _o11()
    o11.main(["--model", "tiny", "--dtype", "float32", "--verbs-per-event", "2", "--layers", "0.5,1.0",
              "--strengths", "0,4", "--n-per", "3", "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr11_*.json")).read_text())
    assert r["steer_layers"] == [1, 2] and r["n_test_prompts"] == 3 * 4 * 3
    for ci in r["results"].values():
        for cons in ci.values():
            for d, by_s in cons.items():
                assert all(abs(v["delta_agent_minus_patient"]) < 1e-4 and v["flip_rate"] == 0 for v in by_s["0.0"].values())
    moved = [v["delta_agent_minus_patient"] for ci in r["results"].values() for cons in ci.values()
             for d in ("subject", "agent") for v in cons[d]["4.0"].values()]
    assert any(abs(x) > 1e-4 for x in moved)
    assert (tmp_path / (next(tmp_path.glob("otr11_*.json")).stem + ".png")).exists()
