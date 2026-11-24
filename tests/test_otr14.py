"""OTR-14 layer pipeline: tiny-model smoke run (blocking self-check, curves, bootstrap summaries)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("o14", ROOT / "scripts" / "otr14_layer_pipeline.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_boot_summary_onset():
    o14 = _mod()
    curves = {f"i{k}": [0.0, 0.1, 0.6, 0.9] for k in range(6)}
    fam = {f"i{k}": f"f{k % 3}" for k in range(6)}
    s = o14.boot_summary(curves, fam, [0, 1, 2, 3], 4, n_boot=50)
    assert s["onset_depth"] == round(2 / 3, 3) and s["peak_depth"] == 1.0 and s["n_families"] == 3


def test_smoke(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from test_bx_mechanism import _tiny_bx
    from relational_lab.experiments import order_role_stimuli as ors
    extra = [i.prompt for i in ors.generate_items()[:3000]] + [" " + n for n in ors.NAMES] * 50
    extra += ["Who was watched by someone? Who was praised by someone?"] * 20
    _tiny_bx(monkeypatch, extra)
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"head_scan": {"top20": ["2.1", "1.0", "0.3"]}}))
    o14 = _mod()
    o14.main(["--model", "tiny", "--dtype", "float32", "--n-per", "2", "--layer-stride", "1", "--min-gap=-1e9",
              "--instruction", "", "--n-boot", "20", "--rank-from", str(prior), "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr14_*.json")).read_text())
    assert r["attention_blocking_layers"] == [0, 1, 2]
    s = r["summary"]
    assert abs(s["swap_answer"]["mean"][-1] - 1.0) < 1e-3
    for k in ("read_question_fail", "read_names_work", "by_from_failing", "by_from_patient"):
        assert s[k] is not None and len(s[k]["mean"]) == 3
    assert r["n_items"] > 0 and r["items"][0]["family_id"]
    assert r["heads_attention_available"] and set(r["heads_attention"]) == {"2.1", "1.0", "0.3"}
