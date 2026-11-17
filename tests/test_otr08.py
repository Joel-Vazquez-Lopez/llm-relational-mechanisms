"""OTR-08: probe logic recovers planted role vs order codes; smoke run on a tiny Llama."""
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def _o8():
    spec = importlib.util.spec_from_file_location("o8", ROOT / "scripts" / "otr08_role_probes.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_planted_role_vs_order_codes():
    o8 = _o8()
    items = o8.build_items(2026, 4)
    torch.manual_seed(0)
    D = 64
    names = sorted({it.agent for it in items} | {it.patient for it in items})
    emb = {n: torch.randn(D) for n in names}
    act = [i for i, it in enumerate(items) if it.construction == "active"]
    pas = [i for i, it in enumerate(items) if it.construction == "passive"]
    for code, expect_agent in (("agent", True), ("first", False)):
        X = torch.stack([emb[o8.target_name(it, code)] + 0.1 * torch.randn(D) for it in items])
        Ya = torch.stack([emb[it.agent] for it in items])
        Yp = torch.stack([emb[it.patient] for it in items])
        tr, te = torch.tensor(act[: len(act) // 2]), torch.tensor(pas)
        pred = o8.ridge_fit_predict(X[tr], Ya[tr], X[te], 0.1)
        p_agent = o8.choose(pred, Ya[te], Yp[te]).float().mean().item()
        assert (p_agent > 0.95) if expect_agent else (p_agent < 0.05), (code, p_agent)


def test_items_cover_constructions_and_flags():
    o8 = _o8()
    items = o8.build_items(2026, 2)
    assert {it.construction for it in items} == set(o8.PROBE_CONSTRUCTIONS)
    for it in items[:200]:
        assert o8.target_name(it, "agent") == it.agent
        assert o8.target_name(it, "first") == it.names[0].name
        assert sum(n.is_subject for n in it.names) == 1


def test_smoke_tiny(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from test_logit_lens import _tiny
    _tiny(monkeypatch)
    o8 = _o8()
    o8.main(["--model", "tiny", "--dtype", "float32", "--verbs-per-event", "2", "--layer-stride", "1",
             "--positions", "period,answer_cue", "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr08_*.json")).read_text())
    assert r["positions"] == ["period", "answer_cue"] and len(r["layers"]) == r["n_layers"]
    e1 = r["results"]["period"]["E1"]["agent"]["passive"]
    assert len(e1) == len(r["layers"]) and all(v is None or 0 <= v <= 1 for v in e1)


def test_smoke_tiny_crosslingual(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from test_logit_lens import _tiny
    _tiny(monkeypatch)
    o8 = _o8()
    o8.main(["--model", "tiny", "--dtype", "float32", "--verbs-per-event", "2", "--layer-stride", "2",
             "--langs", "en,es,gl", "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr08_*_en-es-gl_*.json")).read_text())
    assert r["langs"] == ["en", "es", "gl"] and set(r["transfer"]) == {"en->es", "en->gl"}
    t = r["transfer"]["en->es"]["period"]
    assert set(t["all_trained"]) == {"active", "passive", "object_cleft", "by_cleft", "ovs_clitic", "subject_cleft"}
    assert all(len(v) == len(r["layers"]) for v in t["active_trained"].values())


def test_spanish_items_build():
    o8 = _o8()
    es = o8.build_items(2026, 2, "es")
    assert {it.construction for it in es} == set(o8.constructions_for("es"))
    assert all(it.prompt.startswith("Oración: ") for it in es[:50])
