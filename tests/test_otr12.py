"""OTR-12: knockout and role-vector plumbing on a tiny Llama, built on a tiny OTR-09 run."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_auto_layers():
    o12 = _load("o12", "otr12_knockout_and_role_vector.py")
    rep = {"residual_layers": [0, 10, 20, 30, 40], "residual": {"all": {"repair": [0, 0.1, 0.3, 0.6, 0.9]}}}
    assert o12.auto_layers(rep, 41) == [26, 28, 30, 32, 36]


def test_smoke_tiny(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from test_logit_lens import _tiny
    _tiny(monkeypatch)
    o9 = _load("o9", "otr09_answer_patching.py")
    o9.main(["--model", "tiny", "--dtype", "float32", "--n-per", "8", "--layer-stride", "1", "--head-from", "0",
             "--n-rank", "4", "--top-k", "1,2", "--min-gap", "1e-4", "--out-dir", str(tmp_path)])
    o9_json = next(tmp_path.glob("otr09_*.json"))
    o12 = _load("o12", "otr12_knockout_and_role_vector.py")
    o12.main(["--model", "tiny", "--dtype", "float32", "--otr09", str(o9_json), "--top-k", "1,3",
              "--vector-layers", "0,1", "--alphas", "0,1", "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr12_*.json")).read_text())
    assert r["n_fit"] + r["n_eval"] == 32 and r["vector_layers"] == [0, 1]
    for q in ("working", "normal"):
        assert set(r["knockout"][q]) == {"baseline", "top1", "top3", "rand1", "rand3"}
    # alpha 0 = no edit
    for key in ("fail+v[all]", "patient+v", "working-v", "fail+rand"):
        for li in (0, 1):
            assert abs(r["steering"][f"{key}|{li}|0.0"]["all"]["mean_shift"]) < 1e-5, key
    assert "non_active" in r["steering"]["fail+v[active]|1|1.0"]
    assert (tmp_path / (next(tmp_path.glob("otr12_*.json")).stem + ".png")).exists()
