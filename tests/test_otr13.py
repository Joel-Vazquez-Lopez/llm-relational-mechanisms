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
    assert {o13.source_of(w, s) for w in b} == {"Who", "was", "the", "ING", "by", "?"}
    assert "Who was someone watched by?"[slice(*b["VERB"])] == "watched"
    s2 = o13.question_words("Who was the watching of someone done by?", "source")
    assert o13.source_of("someone", s2) == "someone" and o13.source_of("by", s2) == "by"


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


def test_smoke_src2_and_bx(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from test_bx_mechanism import _tiny_bx
    from relational_lab.experiments import order_role_stimuli as ors
    extra = [i.prompt for i in ors.generate_items()[:3000]] + [" " + n for n in ors.NAMES] * 50
    _tiny_bx(monkeypatch, extra)
    o13 = _mod()
    o13.main(["--model", "tiny", "--dtype", "float32", "--n-per", "2", "--n-tail", "2", "--layer-stride", "1",
              "--min-gap=-1e9", "--instruction", "", "--source-question", "Who was the {ing} of someone done by?",
              "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr13_*_src2_*.json")).read_text())
    assert r["source_of"]["someone"] == "someone"
    o13.main(["--model", "tiny", "--dtype", "float32", "--stimuli", "bx-transitive", "--n-per", "2", "--n-tail", "2",
              "--layer-stride", "1", "--min-gap=-1e9", "--instruction", "", "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr13_*_bx-transitive_*.json")).read_text())
    assert r["stimuli"] == "bx-transitive" and abs(r["interchange"]["tail0"][str(r["layers"][-1])] - 1.0) < 1e-3
