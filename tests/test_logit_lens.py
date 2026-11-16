"""OTR-06 logit lens: the last layer must reproduce the model's own logits (tiny random Llama)."""
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

ROOT = Path(__file__).resolve().parents[1]


def _tiny(monkeypatch):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from relational_lab.experiments import order_role_stimuli as ors
    import relational_lab.instrumentation.hf_scoring as hs

    texts = [i.prompt for i in ors.generate_items(seed=2026)[:2000]]
    tk = Tokenizer(models.BPE(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=600, special_tokens=["[UNK]", "<s>"],
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", unk_token="[UNK]")
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4))
    monkeypatch.setattr(hs, "load_tokenizer", lambda *a, **k: tok)
    monkeypatch.setattr(hs, "load_lm", lambda *a, **k: model)


def test_last_layer_matches_model(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    spec = importlib.util.spec_from_file_location("lens", ROOT / "scripts" / "otr06_logit_lens.py")
    lens = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lens)
    lens.main(["--model", "tiny", "--dtype", "float32", "--n-per-construction", "2", "--out-dir", str(tmp_path),
               "--condition", "agent", "Who was someone {past} by?",
               "--condition", "patient", "Who was {past} by someone?"])
    r = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert r["final_layer_check_max_abs"] < 1e-3
    assert len(r["records"]) == 16 and len(r["records"][0]["ld"]) == 3
    assert {c["template"] for c in r["conditions"]} == {"Who was someone {past} by?", "Who was {past} by someone?"}
