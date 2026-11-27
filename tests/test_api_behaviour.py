"""API-BEH prompts, scoring and caching with a fake endpoint (no network)."""
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("apib", ROOT / "scripts" / "api_behaviour.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_fake_endpoint(monkeypatch, tmp_path):
    ab = _mod()
    calls = []

    def fake_chat(base, key, model, prompt, max_tokens, extra, tries=8):
        calls.append(prompt)
        sent = re.search(r"Sentence: (.*)\n", prompt).group(1)
        names = re.findall(r"\b[A-Z][a-z]+\b", sent)
        if "someone" in prompt.split("Question:")[1] and "done by" not in prompt:  # failing: patient-ish
            return names[-1] if "full sentence" not in prompt else sent
        return names[0]

    monkeypatch.setattr(ab, "chat", fake_chat)
    monkeypatch.setenv("FAKE_KEY", "x")
    args = ["--base-url", "http://fake/v1", "--key-env", "FAKE_KEY", "--model", "m/x", "--n-per", "3",
            "--out-dir", str(tmp_path)]
    ab.main(args)
    r = json.loads((tmp_path / "apibeh_m__x_otr.json").read_text())
    assert r["n_answers"] == 4 * 3 * 4
    assert {row["question"] for row in r["rows"] if row["cond"] == "working"} and \
        all(row["question"].endswith("done by?") for row in r["rows"] if row["cond"] == "working")
    assert all(row["question"].startswith("Who was someone ") for row in r["rows"] if row["cond"] == "failing")
    n = len(calls)
    ab.main(args)  # resumes from the cache: no new requests
    assert len(calls) == n
