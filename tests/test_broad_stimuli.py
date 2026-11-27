"""Broad English stimuli (BX): new vocabulary, decorrelation per block, questions, templates, generation check."""
import importlib.util
from collections import Counter
from pathlib import Path

from relational_lab.experiments import broad_stimuli as bx
from relational_lab.experiments import order_role_stimuli as ors

ROOT = Path(__file__).resolve().parents[1]


def test_new_vocabulary():
    assert not set(bx.NAMES) & set(ors.NAMES)
    otr_verbs = {v[0] for v in ors.VERBS}
    assert not {v[0] for v in bx.TRANSITIVE} & otr_verbs and len(bx.TRANSITIVE) >= 40


def test_examples():
    it = bx.build_item("transitive", "Adam", "Nora", "accuse", "passive", "agent", "active")
    assert it.sentence == "Nora was accused by Adam." and it.question == "Who accused someone?" and it.answer == "Adam"
    it = bx.build_item("ditransitive", "Adam", "Nora", "give", "recipient_passive", "recipient", "active", "a letter")
    assert it.sentence == "Nora was given a letter by Adam." and it.question == "Who did someone give a letter to?"
    assert it.answer == "Nora"
    it = bx.build_item("ditransitive", "Adam", "Nora", "give", "theme_passive", "agent", "passive", "a letter")
    assert it.sentence == "A letter was given to Nora by Adam." and it.question == "By whom was someone given a letter?"
    it = bx.build_item("psych", "Adam", "Nora", "fear", "es_active", "experiencer", "active")
    assert it.sentence == "Nora feared Adam." and it.question == "Who feared someone?" and it.answer == "Nora"
    it = bx.build_item("psych", "Adam", "Nora", "frighten", "se_passive", "stimulus", "passive")
    assert it.sentence == "Nora was frightened by Adam." and it.question == "By whom was someone frightened?"
    assert it.answer == "Adam" and bx.by_phrase_role(it) == "stimulus"
    it = bx.build_item("psych", "Adam", "Nora", "fear", "es_passive", "stimulus", "active")
    assert it.sentence == "Adam was feared by Nora." and bx.by_phrase_role(it) == "experiencer"


def test_decorrelation_and_spans():
    for block in bx.BLOCKS:
        items = bx.generate_items(block)
        assert {i.split for i in items} == {"dev"}
        main = [t for i in items if i.construction in bx.main_constructions(block) for t in i.names]
        pats = Counter((t.is_first, t.is_subject, t.is_agent) for t in main)
        for f, g in (("is_first", "is_subject"), ("is_first", "is_agent"), ("is_subject", "is_agent")):
            assert sum(getattr(t, f) == getattr(t, g) for t in main) * 2 == len(main), (block, f, g)
        for it in items[:300]:
            for t in it.names:
                assert it.prompt[t.prompt_char_start:t.prompt_char_end] == t.name
            assert it.answer in (it.agent, it.patient)


def test_splits_disjoint_and_stable():
    dev = {i.family_id for i in bx.generate_items("transitive", split="dev")}
    test0 = {i.family_id for i in bx.generate_items("transitive", split="test", constructions_subset=("active",))}
    assert dev and test0 and not dev & test0
    a = bx.generate_items("psych", split="dev")[:50]
    b = [i for i in bx.generate_items("psych", split=None) if i.split == "dev"][:50]
    assert [i.item_id for i in a] == [i.item_id for i in b]  # same verbs whichever split is requested


def test_generation_check_bx():
    spec = importlib.util.spec_from_file_location("gc", ROOT / "scripts" / "otr_generate_check.py")
    gc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gc)
    out = gc.prepared_prompts("bx-psych", n=120, question_template="Who was someone {pp} by?", question_role="by")
    assert out
    for it, _ in out:
        assert it.question.startswith("Who was someone ") and it.question.endswith(" by?")
        assert it.role_asked == bx.by_phrase_role(it)
    out = gc.prepared_prompts("bx-ditransitive", n=56, question_template="Who was someone given {theme} by?",
                              question_role="agent")
    assert all(bx.theme_of(it) in it.question for it, _ in out)
    assert gc.event_statement("Nora was given a letter by Adam.", "Adam", "Nora") == ("Adam", "Nora")
