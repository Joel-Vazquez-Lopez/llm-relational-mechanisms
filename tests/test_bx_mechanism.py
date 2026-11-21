"""OTR-03 / OTR-09 on the broad English stimuli (bx_adapter): design checks and tiny-model smoke runs."""
import importlib.util
import json
from pathlib import Path

import pytest

from relational_lab.experiments import broad_stimuli as bx
from relational_lab.experiments import bx_adapter as bxa
from relational_lab.experiments import order_role_transplant as otr

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_normalize_maps_block_roles_onto_slots():
    for block, (ra, rb) in bx.ROLES.items():
        it = bx.build_item(block, "Beth", "Adam", {"transitive": "kick", "ditransitive": "give", "psych": "fear"}[block],
                           bxa.BASE_CONSTRUCTIONS[block][-1], rb, "active", "a book" if block == "ditransitive" else "")
        n = bxa.normalize(it)
        assert bxa.real_role(n) == rb and n.role_asked == "patient" and n.answer == "Adam"


@pytest.mark.parametrize("stimuli", bxa.STIMULI)
def test_transplant_sources_share_verb_theme_and_class(stimuli):
    bases = bxa.select_bases(stimuli, "dev", 6, ["active"], 2026)
    assert bases and {b.role_asked for b in bases} == {"agent", "patient"}
    ts = otr.make_transplants(bases, seed=2026, names=bx.NAMES, builder=bxa.builder,
                              constructions_for=bxa.source_constructions)
    for t in ts:
        assert t.source.lemma == t.base.lemma and bx.theme_of(t.source) == bx.theme_of(t.base)
        assert not {t.source.agent, t.source.patient} & {t.base.agent, t.base.patient}
        assert t.source.role_asked == t.base.role_asked
        if stimuli == "bx-psych":
            assert t.source.construction.split("_")[0] == t.base.construction.split("_")[0]
    n_src = 3 if stimuli == "bx-psych" else 4
    assert len(ts) == len(bases) * 2 * n_src * 4


def test_main_blocks_decorrelate_exactly_psych_reports_correlations():
    for stimuli in ("bx-transitive", "bx-ditransitive"):
        bases = bxa.select_bases(stimuli, "dev", 8, ["active"], 2026)
        ts = otr.make_transplants(bases, seed=1, names=bx.NAMES, builder=bxa.builder,
                                  constructions_for=bxa.source_constructions)
        assert all(abs(v) < 1e-9 for v in otr.predictor_correlations(ts).values())
    bases = bxa.select_bases("bx-psych", "dev", 8, ["active"], 2026)
    ts = otr.make_transplants(bases, seed=1, names=bx.NAMES, builder=bxa.builder,
                              constructions_for=bxa.source_constructions)
    corr = otr.predictor_correlations(ts)
    assert all(abs(v) < 0.9 for v in corr.values())  # within-class: correlated, but separable


def test_by_question_targets_the_by_phrase_role():
    bases = bxa.select_bases("bx-psych", "dev", 40, ["active"], 2026)
    sel = bxa.with_by_question(bases, "Who was someone {pp} by?")
    assert sel and all(b.question.startswith("Who was someone ") and b.question.endswith(" by?") for b in sel)
    for b in sel:
        if b.construction.startswith("es_"):  # "Mira feared Leon": by-phrase = experiencer = B slot
            assert bxa.real_role(b) == "experiencer" and b.answer == b.patient
        else:
            assert bxa.real_role(b) == "stimulus" and b.answer == b.agent
    assert {b.construction[:3] for b in sel} == {"se_", "es_"}


def test_otr09_bx_pairs():
    o9 = _script("otr09_answer_patching")
    triples = o9.make_pairs(2026, bxa.DEFAULT_BASE_QUESTION["ditransitive"], bxa.DEFAULT_SOURCE_QUESTION,
                            "Reply with one name only.", ["do_active", "pd_active"], 4, stimuli="bx-ditransitive")
    assert len(triples) == 8
    for b, s, d in triples:
        theme = bx.theme_of(b)
        pp = bx.FORMS[b.lemma][2]
        assert b.question == f"Who was someone {pp} {theme} by?" and b.answer == b.agent
        assert b.sentence == s.sentence and s.question.startswith("Who was the ")
        assert b.construction == d.construction and not {d.agent, d.patient} & {b.agent, b.patient}
        assert b.prompt.endswith("Reply with one name only.\nAnswer:")


def test_left_context_null_constructions_exist():
    for block, cons in bxa.NULL_CONSTRUCTIONS.items():
        for c in cons:
            tpl = bx.constructions(block)[c][0]
            assert tpl.startswith("{A}") or tpl.startswith("{B}")


# ------------------------------------------------------------------ tiny-model smoke runs

def _tiny_bx(monkeypatch, extra_texts=()):
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    import relational_lab.instrumentation.hf_scoring as hs

    texts = []
    for block in bx.BLOCKS:
        texts += [i.prompt for i in bx.generate_items(block, split="dev")[:3000]]
    texts += [f"Who was the {g} done by? Who was someone {pp} by?" for _l, _p, pp, g in bx.FORMS.values()]
    texts += [" " + n for n in bx.NAMES] * 50
    texts += list(extra_texts)
    tk = Tokenizer(models.BPE(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=1500, special_tokens=["[UNK]", "<s>"],
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", unk_token="[UNK]")
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4))
    monkeypatch.setattr(hs, "load_tokenizer", lambda *a, **k: tok)
    monkeypatch.setattr(hs, "load_lm", lambda *a, **k: model)


def test_smoke_otr03_bx_psych_failing_question(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    _tiny_bx(monkeypatch)
    o3 = _script("otr03_code_transplant")
    o3.main(["--model", "tiny", "--dtype", "float32", "--device", "cpu", "--stimuli", "bx-psych", "--n-events", "16",
             "--agent-question", "Who was someone {pp} by?", "--include-incorrect", "--n-boot", "0",
             "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr03_*_bx-psych_aq_*.json")).read_text())
    assert r["stimuli"] == "bx-psych" and r["n_analysed"] > 0
    assert set(r["base_real_roles"].values()) <= {"stimulus", "experiencer"}
    # the left-context null holds on the new stimuli too
    assert max(abs(v) for v in r["left_context_null_role_slope"][:-1]) < 1e-3


def test_smoke_otr09_bx_transfer(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    _tiny_bx(monkeypatch)
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"head_scan": {"top20": [f"{li}.{h}" for li in range(3) for h in range(4)]}}))
    o9 = _script("otr09_answer_patching")
    o9.main(["--model", "tiny", "--dtype", "float32", "--stimuli", "bx-transitive", "--n-per", "3",
             "--layer-stride", "1", "--head-from", "0.6", "--n-rank", "3", "--top-k", "1,12", "--min-gap", "1e-4",
             "--rank-from", str(prior), "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr09_*_bx-transitive_*.json")).read_text())
    assert r["stimuli"] == "bx-transitive" and r["base_question"] == "Who was someone {pp} by?"
    assert r["scan_layers"] == [2] and r["head_layers"] == [0, 1, 2]
    assert abs(r["residual"]["all"]["repair"][-1] - 1.0) < 1e-3
    # all 12 transferred heads patched jointly = the whole attention output of every layer: exact repair
    assert abs(r["transfer"]["by_k"]["12"]["all_items"]["repair"] - 1.0) < 1e-3


# ------------------------------------------------------------------ Spanish / Galician (xl_adapter)

def test_xl_transplants_decorrelate_and_share_verb():
    from relational_lab.experiments import xl_adapter as xla
    for lang in ("es", "gl"):
        bases = xla.select_bases(lang, "dev", 6, ["passive"], ["agent"], 2026)
        assert bases and all(b.question.startswith(("¿Por quién", "Por quen")) for b in bases)
        ts = otr.make_transplants(bases, xla.TRANSPLANT_CONSTRUCTIONS, seed=1, builder=xla.builder)
        assert len(ts) == len(bases) * 2 * 4 * 4
        for t in ts:
            assert t.source.lemma == t.base.lemma and t.source.event_id.startswith(lang + ":")
        assert all(abs(v) < 1e-9 for v in otr.predictor_correlations(ts).values())


def test_xl_repair_pairs():
    o9 = _script("otr09_answer_patching")
    tr = o9.make_pairs(2026, None, None, "Responde con un solo nombre.", ["active", "passive", "object_cleft"], 3,
                       stimuli="es")
    assert len(tr) == 9
    for b, s, d in tr:
        assert b.sentence == s.sentence and b.question.startswith("¿Por quién") and s.question.startswith("¿Quién")
        assert d.question.startswith("¿Quién") and b.construction == d.construction
        assert b.prompt.endswith("Responde con un solo nombre.\nRespuesta:") and b.answer == b.agent


def test_smoke_otr03_and_otr09_spanish(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from relational_lab.experiments import multilingual_stimuli as xl
    from relational_lab.experiments import order_role_stimuli as ors
    extra = [i.prompt + " Responde con un solo nombre." for i in xl.generate_items("es")[:4000]]
    extra += [" " + n for n in ors.NAMES] * 50
    _tiny_bx(monkeypatch, extra)
    o3 = _script("otr03_code_transplant")
    o3.main(["--model", "tiny", "--dtype", "float32", "--device", "cpu", "--stimuli", "es", "--n-events", "6",
             "--question-voices", "passive", "--roles", "agent", "--include-incorrect", "--n-boot", "0",
             "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr03_*_es_*.json")).read_text())
    assert r["xl_generator_version"] and r["n_analysed"] > 0
    assert max(abs(v) for v in r["left_context_null_role_slope"][:-1]) < 1e-3
    o9 = _script("otr09_answer_patching")
    o9.main(["--model", "tiny", "--dtype", "float32", "--stimuli", "es", "--n-per", "3", "--layer-stride", "1",
             "--head-from", "0.6", "--n-rank", "2", "--top-k", "1", "--min-gap", "1e-4",
             "--instruction", "Responde con un solo nombre.", "--out-dir", str(tmp_path)])
    r9 = json.loads(next(tmp_path.glob("otr09_*_es_*.json")).read_text())
    assert abs(r9["residual"]["all"]["repair"][-1] - 1.0) < 1e-3


# ------------------------------------------------------------------ natural sentences (UD)

def test_ud_items_are_natural_sentences_with_two_names():
    from relational_lab.experiments import ud_stimuli as uds
    pairs = uds.generate_pairs("Who was someone {pp} by?", "Who was the {ing} done by?", "Reply with one name only.")
    assert len(pairs) > 300 and {b.construction for b, _ in pairs} == set(uds.CONSTRUCTIONS)
    for b, s in pairs:
        assert b.sentence == s.sentence and b.answer == b.agent
        assert b.sentence.count(b.agent) == 1 and b.sentence.count(b.patient) == 1
        assert b.prompt.endswith("Reply with one name only.\nAnswer:")
        for t in b.names:
            assert b.prompt[t.prompt_char_start:t.prompt_char_end] == t.name
        a_tok = next(t for t in b.names if t.is_agent)
        assert a_tok.is_subject == (b.construction == "ud_active")
    assert " n't" not in " ".join(b.sentence for b, _ in pairs)


def test_smoke_otr09_ud(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from relational_lab.experiments import order_role_stimuli as ors
    from relational_lab.experiments import ud_stimuli as uds
    extra = [b.prompt for b, _ in uds.generate_pairs("Who was someone {pp} by?", "Who was the {ing} done by?")]
    extra += [" " + n for n in ors.NAMES] * 50
    _tiny_bx(monkeypatch, extra)
    o9 = _script("otr09_answer_patching")
    o9.main(["--model", "tiny", "--dtype", "float32", "--stimuli", "ud-en", "--n-per", "6", "--layer-stride", "1",
             "--head-from", "0.6", "--n-rank", "2", "--top-k", "1", "--min-gap", "1e-4", "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr09_*_ud-en_*.json")).read_text())
    assert r["ud_version"] and abs(r["residual"]["all"]["repair"][-1] - 1.0) < 1e-3


def test_smoke_otr09_wiki(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from relational_lab.experiments import order_role_stimuli as ors
    sents = [("In {y}, {p} was defeated by {a} in the regional election.", "passive", "defeat", "defeated", "defeating"),
             ("Later that year, {a} hired {p} as an assistant at the museum.", "active", "hire", "hired", "hiring")]
    people = ["Mary Montague", "John Carlton", "Peter Hawthorne", "Susan Fenwick", "Laura Becket", "Owen Marsh",
              "Diane Croft", "Victor Lane"]
    recs = []
    for k in range(16):
        tpl, kind, lm, past, ing = sents[k % 2]
        a, p = people[k % 8], people[(k + 3) % 8]
        s = tpl.format(y=1990 + k, a=a, p=p)
        recs.append({"sentence": s, "kind": kind, "lemma": lm, "past": past, "pp": past, "ing": ing, "agent": a,
                     "patient": p, "agent_char": [s.index(a), s.index(a) + len(a)],
                     "patient_char": [s.index(p), s.index(p) + len(p)], "article_id": str(k), "title": "T"})
    wf = tmp_path / "w.jsonl"
    wf.write_text("\n".join(json.dumps(r) for r in recs))
    extra = [r["sentence"] + " Who was someone defeated by? Who was the hiring done by?" for r in recs] * 20
    extra += [" " + n for n in ors.NAMES + tuple(people)] * 50
    _tiny_bx(monkeypatch, extra)
    o9 = _script("otr09_answer_patching")
    for names in ("real", "neutral"):
        o9.main(["--model", "tiny", "--dtype", "float32", "--stimuli", "wiki-en", "--wiki-file", str(wf),
                 "--wiki-names", names, "--n-per", "8", "--layer-stride", "1", "--head-from", "0.6", "--n-rank", "2",
                 "--top-k", "1", "--min-gap", "1e-4", "--out-dir", str(tmp_path)])
        r = json.loads(next(tmp_path.glob(f"otr09_*_wiki-en-{names}_*.json")).read_text())
        assert r["wiki"]["names"] == names and abs(r["residual"]["all"]["repair"][-1] - 1.0) < 1e-3
