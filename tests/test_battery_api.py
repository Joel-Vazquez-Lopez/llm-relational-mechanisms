"""Survey battery (tiny model) and API runner (local fake OpenAI-compatible server)."""
import importlib.util
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_battery_runs_all_conditions_with_one_load(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from test_logit_lens import _tiny
    _tiny(monkeypatch)
    import relational_lab.instrumentation.hf_scoring as hs
    loads = {"n": 0}
    orig = hs.load_lm
    monkeypatch.setattr(hs, "load_lm", lambda *a, **k: (loads.__setitem__("n", loads["n"] + 1), orig())[1])
    bat = _load("otr_battery", "otr_battery.py")
    bat.main(["--model", "tiny", "--dtype", "float32", "--n-scale", "0.05", "--max-new-tokens", "4",
              "--out-dir", str(tmp_path), "--only", "B0,B1,B3"])
    man = json.loads(next(tmp_path.glob("battery_*.json")).read_text())
    assert [r["id"] for r in man["runs"]] == ["B0", "B1", "B3"]
    assert all(r["file"] and (tmp_path / r["file"]).exists() for r in man["runs"])
    assert loads["n"] == 1
    assert len({r["file"] for r in man["runs"]}) == 3


class _Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        m = re.search(r"Sentence: (.*)\n", prompt)
        names = re.findall(r"\b[A-Z][a-z]+\b", m.group(1)) if m else []
        names = [n for n in names if n not in ("It",)]
        ans = names[0] if names else "OK"
        out = {"model": body["model"], "choices": [{"message": {"content": ans}}]}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_api_runner_against_fake_server(monkeypatch, tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("FAKE_KEY", "x")
    api = _load("otr_generate_api", "otr_generate_api.py")
    api.main(["--base-url", f"http://127.0.0.1:{srv.server_port}/v1", "--model", "fake-1", "--key-env", "FAKE_KEY",
              "--only", "B1,B4", "--n-scale", "0.1", "--workers", "4", "--out-dir", str(tmp_path)])
    srv.shutdown()
    files = sorted(tmp_path.glob("otrgen_fake-1_dev_q-*.json"))
    assert len(files) == 2
    for f in files:
        r = json.loads(f.read_text())
        assert r["api"] and r["n"] > 0 and r["kinds"]["direct"] == 1.0
        # the fake always answers with the sentence's first name: actives -> agent, passives -> patient
        acc = {k.split("/")[0]: v for k, v in r["cells"].items()}
        if r["question_role"] == "agent":
            assert acc["active"] == 1.0 and acc["passive"] == 0.0
        else:
            assert acc["active"] == 0.0 and acc["passive"] == 1.0
