"""OTR-15 prompt formats and a tiny-model smoke run; OTR-09 head controls."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_formats_keep_sentence_and_question():
    o15 = _load("otr15_prompt_robustness")
    from otr_generate_check import prepared_prompts
    it, prompt = prepared_prompts("otr", 2026, 8, o15.QUESTIONS["failing"], "agent", "Reply with one name only.")[0]
    for f in o15.FORMATS:
        t = o15.reformat(it, prompt, f, "Reply with one name only.")
        assert it.sentence in t and it.question.lower() in t.lower() and "Reply with one name only." in t
    assert "Sentence:" not in o15.reformat(it, prompt, "nolabel", "x")
    few = o15.reformat(it, prompt, "fewshot", "Reply with one name only.")
    assert few.count("Answer:") == 3 and "someone by?" not in few.split("Sentence: " + it.sentence)[0]


def test_smoke(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from test_logit_lens import _tiny
    _tiny(monkeypatch)
    o15 = _load("otr15_prompt_robustness")
    o15.main(["--model", "tiny", "--dtype", "float32", "--n", "8", "--max-new-tokens", "3", "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr15_*.json")).read_text())
    assert set(r["working_minus_failing"]) == set(o15.FORMATS)
    assert len(r["rows"]) == 2 * len(o15.FORMATS) * 8


def test_otr09_controls(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from test_logit_lens import _tiny
    _tiny(monkeypatch)
    o9 = _load("otr09_answer_patching")
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"head_scan": {"top20": [f"{li}.{h}" for li in range(3) for h in range(4)]}}))
    o9.main(["--model", "tiny", "--dtype", "float32", "--n-per", "3", "--layer-stride", "1", "--head-from", "0.6",
             "--n-rank", "3", "--top-k", "1,12", "--min-gap", "1e-4", "--controls", "--rank-from", str(prior),
             "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr09_*.json")).read_text())
    t = r["transfer"]["by_k"]["12"]
    assert abs(t["all_items"]["repair"] - 1.0) < 1e-3  # all heads from the working run: exact
    assert "other_failing" in t and "random_nudge" in t
    assert "top_other_failing" in r["circuit_eval"]["by_k"]["1"]
