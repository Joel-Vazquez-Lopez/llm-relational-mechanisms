"""OTR-13 question-word patching: anchors and a tiny-model smoke run."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("o13", ROOT / "scripts" / "otr13_question_words.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_question_words():
    o13 = _mod()
    b = o13.question_words("Who was someone watched by?", "base")
    s = o13.question_words("Who was the watching done by?", "source")
    assert list(b) == list(o13.BASE_WORDS)
    assert set(o13.SOURCE_OF.values()) <= set(s)
    assert "Who was someone watched by?"[slice(*b["VERB"])] == "watched"


def test_smoke(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from test_logit_lens import _tiny
    _tiny(monkeypatch)
    o13 = _mod()
    o13.main(["--model", "tiny", "--dtype", "float32", "--n-per", "3", "--n-tail", "3", "--layer-stride", "1",
              "--min-gap=-1e9", "--instruction", "", "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr13_*.json")).read_text())
    last = str(r["layers"][-1])
    # patching the answer position at the last layer = the working run's answer: exact repair
    assert abs(r["interchange"]["tail0"][last] - 1.0) < 1e-3
    assert abs(r["reverse_damage"]["tail0"][last] - 1.0) < 1e-3
    assert r["n_items"] > 0 and set(r["words"]) >= {"VERB", "by", "tail0"}
