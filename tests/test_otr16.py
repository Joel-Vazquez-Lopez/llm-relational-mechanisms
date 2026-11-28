"""OTR-16 "by" -> heads -> answer mediation: tiny-model smoke run and identities."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("o16", ROOT / "scripts" / "otr16_by_to_heads.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run(monkeypatch, tmp_path, heads, by_layer, extra=()):
    pytest.importorskip("transformers")
    from test_logit_lens import _tiny
    _tiny(monkeypatch)
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"head_scan": {"top20": heads}}))
    _mod().main(["--model", "tiny", "--dtype", "float32", "--n-per", "3", "--instruction", "", "--min-gap=-1e9",
                 "--min-by-effect=-1e9", "--by-layer", str(by_layer), "--rank-from", str(prior), "--k", str(len(heads)),
                 "--n-random", "2", "--out-dir", str(tmp_path), *extra])
    return json.loads(next(tmp_path.glob("otr16_*.json")).read_text())


def test_smoke_and_identities(monkeypatch, tmp_path):
    r = _run(monkeypatch, tmp_path, ["1.0", "2.1"], 0)
    assert r["n_items"] > 0 and r["heads"] == ["1.0", "2.1"] and len(r["random_head_sets"]) == 2
    for it in r["items"]:
        # with no change reaching the heads (B = F), M = F and K = B; checked on the raw logits:
        if abs(it["ld_by"] - it["ld_failing"]) < 1e-9:
            assert abs(it["ld_mediated"] - it["ld_failing"]) < 1e-4
        assert all(x == x for x in (it["ld_by"], it["ld_mediated"], it["ld_blocked"]))
    s = r["summary"]["all"]
    assert all(s[k] is not None for k in ("by_repair", "mediated", "blocked", "mediated_random"))


def test_by_at_last_layer_cannot_reach_answer(monkeypatch, tmp_path):
    """the "by" token is not the answer position: patching it at the last layer leaves the answer unchanged"""
    r = _run(monkeypatch, tmp_path, ["1.0"], 2)
    for it in r["items"]:
        assert abs(it["ld_by"] - it["ld_failing"]) < 1e-4


def test_unknown_head_refused(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="heads not found"):
        _run(monkeypatch, tmp_path, ["1.99"], 0)
