"""Swedish OTR stimuli (SV 0.1.0, not yet native-checked): rendering, flags, spans, splits, templates."""
import importlib.util
from pathlib import Path

from relational_lab.experiments import order_role_stimuli as ors
from relational_lab.experiments import swedish_stimuli as sv

ROOT = Path(__file__).resolve().parents[1]


def test_sentences_and_questions():
    want = {"active": "Daniel berömde Laura.", "passive": "Laura blev berömd av Daniel.",
            "object_cleft": "Det var Laura som Daniel berömde.", "by_cleft": "Det var av Daniel som Laura blev berömd.",
            "subject_cleft": "Det var Daniel som berömde Laura."}
    for c, s in want.items():
        assert sv.render_sentence(c, "Daniel", "Laura", "berömde", "berömd") == s
    assert sv.render_question("agent", "active", "berömde", "berömd") == "Vem var det som berömde någon?"
    assert sv.render_question("patient", "active", "berömde", "berömd") == "Vem var det som någon berömde?"
    assert sv.render_question("agent", "passive", "berömde", "berömd") == "Av vem blev någon berömd?"
    assert sv.fill_template("Vem blev någon {pp} av?", "berömde", "berömma") == "Vem blev någon berömd av?"
    assert sv.fill_template("Av vem {spass} någon?", "berömde", "berömma") == "Av vem berömdes någon?"


def test_items_flags_spans_and_splits():
    items = sv.generate_items()
    assert {i.construction for i in items} == set(sv.SV_CONSTRUCTIONS)
    # same family splits as English: dev families identical
    en_dev = {i.family_id for i in ors.generate_items() if i.split == "dev"}
    assert {i.family_id for i in items if i.split == "dev"} == en_dev
    # order / function / role exactly uncorrelated over the four main constructions (per name token)
    toks = [t for i in items if i.construction in sv.SV_MAIN_CONSTRUCTIONS for t in i.names]
    for f, g in (("is_first", "is_subject"), ("is_first", "is_agent"), ("is_subject", "is_agent")):
        agree = sum(getattr(t, f) == getattr(t, g) for t in toks)
        assert agree * 2 == len(toks), (f, g)
    for it in items[:200]:
        for t in it.names:
            assert it.prompt[t.prompt_char_start:t.prompt_char_end] == t.name
        assert it.prompt.endswith("Svar:") and it.prompt[it.span("period")[0]] == "."


def test_generation_check_uses_swedish():
    spec = importlib.util.spec_from_file_location("gc", ROOT / "scripts" / "otr_generate_check.py")
    gc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gc)
    out = gc.prepared_prompts("sv", n=64, question_template="Vem blev någon {pp} av?", question_role="agent",
                              instruction=sv.INSTRUCTIONS["name"])
    assert out and all(it.question.startswith("Vem blev någon ") and it.question.endswith(" av?") for it, _ in out)
    assert all(p.endswith("Svara med ett enda namn.\nSvar:") for _, p in out)
    # Swedish full-sentence answers are recognised as event statements
    assert gc.event_statement("Laura blev berömd av Daniel.", "Daniel", "Laura") == ("Daniel", "Laura")
    assert gc.event_statement("Daniel berömde Laura.", "Daniel", "Laura") == ("Daniel", "Laura")
    assert gc.event_statement("Det var av Laura som Daniel blev berömd.", "Daniel", "Laura") == ("Laura", "Daniel")
    assert gc.event_statement("Laura berömdes av Daniel.", "Daniel", "Laura") == ("Daniel", "Laura")
