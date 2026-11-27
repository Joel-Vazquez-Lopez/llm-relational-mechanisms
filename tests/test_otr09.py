"""OTR-09: pairs and donors, exact last-layer repair, head scan and circuit plumbing on a tiny Llama."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _o9():
    spec = importlib.util.spec_from_file_location("o9", ROOT / "scripts" / "otr09_answer_patching.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_make_pairs():
    o9 = _o9()
    triples = o9.make_pairs(2026, "Who was someone {past} by?", "Who was the {ing} done by?", "Reply with one name only.",
                            ["active", "object_cleft"], 5)
    assert len(triples) == 10
    for b, s, d in triples:
        assert b.sentence == s.sentence and b.agent == s.agent and b.construction == d.construction
        assert not {d.agent, d.patient} & {b.agent, b.patient}
        assert b.question == f"Who was someone {b.verb} by?" and s.question.startswith("Who was the ")
        assert b.prompt.endswith("Reply with one name only.\nAnswer:") and s.prompt.count(s.question) == 1


def test_repair_normalisation():
    o9 = _o9()
    assert o9.repair(5.0, -5.0, 5.0) == 1.0 and o9.repair(-5.0, -5.0, 5.0) == 0.0 and o9.repair(0.0, -5.0, 5.0) == 0.5


def test_smoke_tiny_exact_last_layer_repair(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from test_logit_lens import _tiny
    _tiny(monkeypatch)
    o9 = _o9()
    o9.main(["--model", "tiny", "--dtype", "float32", "--n-per", "4", "--layer-stride", "1", "--head-from", "0",
             "--n-rank", "4", "--top-k", "1,2,12", "--min-gap", "1e-4", "--heads", "1.0,2.3",
             "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr09_*.json")).read_text())
    res = r["residual"]["all"]
    assert r["residual_layers"] == [0, 1, 2]
    # patching the full residual at the answer position in the LAST layer reproduces the working run
    assert abs(res["repair"][-1] - 1.0) < 1e-3 and abs(res["reverse_break"][-1] - 1.0) < 1e-3
    assert r["head_layers"] == [0, 1, 2] and len(r["head_scan"]["scores"]) == 12
    assert set(r["circuit_eval"]["by_k"]) == {"1", "2", "12"}
    # all heads of all layers patched: the answer position is computed exactly as in the working run
    assert abs(r["circuit_eval"]["by_k"]["12"]["top"]["repair"] - 1.0) < 1e-3
    assert r["circuit_eval"]["named_heads"]["heads"] == ["1.0", "2.3"]
    assert (tmp_path / (next(tmp_path.glob("otr09_*.json")).stem + ".png")).exists()


def test_layers_option(monkeypatch, tmp_path):
    """--layers restricts the residual scan (lean test runs); a layer outside the model is refused."""
    pytest.importorskip("transformers")
    from test_logit_lens import _tiny
    _tiny(monkeypatch)
    o9 = _o9()
    o9.main(["--model", "tiny", "--dtype", "float32", "--n-per", "4", "--layers", "0,2", "--head-from", "0",
             "--n-rank", "4", "--top-k", "2", "--min-gap", "1e-4", "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr09_*.json")).read_text())
    assert r["residual_layers"] == [0, 2] and len(r["residual"]["all"]["repair"]) == 2
    with pytest.raises(SystemExit):
        o9.main(["--model", "tiny", "--dtype", "float32", "--n-per", "2", "--layers", "7", "--out-dir", str(tmp_path)])
