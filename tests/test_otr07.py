"""OTR-07 smoke test on a tiny random Llama (skipped without torch)."""
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
from test_logit_lens import _tiny  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_answer_process_runs_and_is_consistent(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    spec = importlib.util.spec_from_file_location("o7", ROOT / "scripts" / "otr07_answer_process.py")
    o7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(o7)
    o7.main(["--model", "tiny", "--dtype", "float32", "--n-per-construction", "15", "--probe-every", "2",
             "--out-dir", str(tmp_path), "--condition", "agent", "Who was someone {past} by?"])
    r = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert len(r["records"]) == 60
    for rec in r["records"]:
        assert abs(rec["lens_bare"][-1] - rec["ld_model_bare"]) < 1e-3
        assert len(rec["attn_agent_by"]) == r["n_layers"]
    assert r["probes"]


def test_by_prefix_control(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    spec = importlib.util.spec_from_file_location("o7b", ROOT / "scripts" / "otr07_answer_process.py")
    o7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(o7)
    o7.main(["--model", "tiny", "--dtype", "float32", "--n-per-construction", "4", "--probe-every", "0",
             "--by-prefix", "Someone was {verb} by",
             "--out-dir", str(tmp_path), "--condition", "agent", "Who was someone {past} by?"])
    f = next(tmp_path.glob("*_byctl_*.json"))
    r = json.loads(f.read_text())
    assert r["by_prefix"] == "Someone was {verb} by" and len(r["records"]) == 16
