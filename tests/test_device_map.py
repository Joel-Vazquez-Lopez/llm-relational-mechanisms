"""--device-map (70B-class models split over 2 GPUs): the loader and every script that uses it.

A real split over two GPUs cannot run here; these tests run the same code path on CPU
(Accelerate dispatch, ``hf_device_map`` set, no ``model.to``) and check that results match the
ordinary single-device path. The first real two-GPU run is a short smoke job on Pelle.
"""
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _dispatching(monkeypatch):
    """Wrap the tiny-model stub so that a device_map request goes through Accelerate's dispatch."""
    import relational_lab.instrumentation.hf_scoring as hs
    base = hs.load_lm
    seen = {}

    def lm(*a, **k):
        m = base(*a, **k)
        if "device_map" in k:
            from accelerate import dispatch_model
            seen["device_map"] = k["device_map"]
            m = dispatch_model(m, device_map={"": "cpu"})
            m.hf_device_map = {"": "cpu"}
        return m
    monkeypatch.setattr(hs, "load_lm", lm)
    return seen


def test_load_for_run_from_disk(tmp_path):
    """Real from_pretrained with a device map (the path Llama/Qwen 70B take on Pelle)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from relational_lab.instrumentation.hf_scoring import load_for_run
    torch.manual_seed(0)
    LlamaForCausalLM(LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                 num_attention_heads=2, num_key_value_heads=2)).save_pretrained(tmp_path)
    a, dev_a = load_for_run(str(tmp_path), dtype=torch.float32, device="cpu")
    b, dev_b = load_for_run(str(tmp_path), dtype=torch.float32, device_map="cpu")
    assert str(dev_a) == str(dev_b) == "cpu" and not b.training
    x = torch.tensor([[1, 5, 7, 9]])
    with torch.no_grad():
        assert torch.allclose(a(input_ids=x).logits, b(input_ids=x).logits)


def test_otr09_same_with_device_map(monkeypatch, tmp_path):
    from test_bx_mechanism import _tiny_bx
    _tiny_bx(monkeypatch)
    seen = _dispatching(monkeypatch)
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"head_scan": {"top20": [f"{li}.{h}" for li in range(3) for h in range(4)]}}))
    out = {}
    for tag, extra in (("plain", []), ("split", ["--device-map", "cpu"])):
        d = tmp_path / tag
        _script("otr09_answer_patching").main(
            ["--model", "tiny", "--dtype", "float32", "--stimuli", "bx-transitive", "--n-per", "3",
             "--layer-stride", "1", "--head-from", "0.6", "--n-rank", "3", "--top-k", "1,12", "--min-gap", "1e-4",
             "--rank-from", str(prior), "--skip-scan", "--controls", "--controls-extra", "--n-random", "2",
             "--out-dir", str(d)] + extra)
        out[tag] = json.loads(next(d.glob("otr09_*.json")).read_text())
    assert seen["device_map"] == {"": "cpu"}
    a, b = out["plain"], out["split"]
    assert a["residual"]["all"]["repair"] == pytest.approx(b["residual"]["all"]["repair"], abs=1e-5)
    for mode in ("all_items", "other_failing", "other_working", "mean_failing"):
        assert a["transfer"]["by_k"]["12"][mode]["repair"] == pytest.approx(
            b["transfer"]["by_k"]["12"][mode]["repair"], abs=1e-5)


def test_otr03_same_with_device_map(monkeypatch, tmp_path):
    from test_bx_mechanism import _tiny_bx
    _tiny_bx(monkeypatch)
    _dispatching(monkeypatch)
    ys = {}
    for tag, extra in (("plain", []), ("split", ["--device-map", "cpu"])):
        d = tmp_path / tag
        _script("otr03_code_transplant").main(
            ["--model", "tiny", "--dtype", "float32", "--device", "cpu", "--stimuli", "bx-transitive",
             "--n-events", "4", "--layers", "1", "--n-boot", "0", "--out-dir", str(d)] + extra)
        r = json.loads(next(d.glob("otr03_*.json")).read_text())
        ys[tag] = [rec["y"][0] for rec in r["records"]]
        if extra:
            assert r["device_map"] == ["cpu"]
    assert ys["plain"] and ys["plain"] == pytest.approx(ys["split"], abs=1e-5)


def test_otr14_runs_with_device_map(monkeypatch, tmp_path):
    """includes the attention-blocking pre-hook, which builds its mask on the layer's device"""
    from test_bx_mechanism import _tiny_bx
    from relational_lab.experiments import order_role_stimuli as ors
    extra = [i.prompt for i in ors.generate_items()[:3000]] + [" " + n for n in ors.NAMES] * 50
    extra += ["Who was watched by someone? Who was praised by someone?"] * 20
    _tiny_bx(monkeypatch, extra)
    _dispatching(monkeypatch)
    _script("otr14_layer_pipeline").main(
        ["--model", "tiny", "--dtype", "float32", "--n-per", "2", "--layer-stride", "1", "--min-gap=-1e9",
         "--instruction", "", "--n-boot", "10", "--device-map", "cpu", "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr14_*.json")).read_text())
    assert r["attention_blocking_layers"] == [0, 1, 2] and r["n_items"] > 0


def test_otr07_runs_with_device_map(monkeypatch, tmp_path):
    from test_logit_lens import _tiny
    _tiny(monkeypatch)
    _dispatching(monkeypatch)
    _script("otr07_answer_process").main(
        ["--model", "tiny", "--dtype", "float32", "--n-per-construction", "4", "--probe-every", "0",
         "--device-map", "cpu", "--out-dir", str(tmp_path), "--condition", "agent", "Who was someone {past} by?"])
    r = json.loads(next(tmp_path.glob("*.json")).read_text())
    for rec in r["records"]:
        assert abs(rec["lens_bare"][-1] - rec["ld_model_bare"]) < 1e-3


def test_otr16_same_with_device_map(monkeypatch, tmp_path):
    from test_logit_lens import _tiny
    _tiny(monkeypatch)
    _dispatching(monkeypatch)
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"head_scan": {"top20": ["1.0", "2.1"]}}))
    out = {}
    for tag, extra in (("plain", []), ("split", ["--device-map", "cpu"])):
        d = tmp_path / tag
        _script("otr16_by_to_heads").main(
            ["--model", "tiny", "--dtype", "float32", "--n-per", "3", "--instruction", "", "--min-gap=-1e9",
             "--min-by-effect=-1e9", "--by-layer", "0", "--rank-from", str(prior), "--k", "2", "--n-random", "2",
             "--out-dir", str(d)] + extra)
        out[tag] = json.loads(next(d.glob("otr16_*.json")).read_text())
    for a, b in zip(out["plain"]["items"], out["split"]["items"]):
        for k in ("ld_failing", "ld_by", "ld_mediated", "ld_blocked"):
            assert a[k] == pytest.approx(b[k], abs=1e-5)
