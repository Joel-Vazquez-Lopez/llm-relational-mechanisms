import re
from collections import Counter

from relational_lab.experiments import order_role_stimuli as ors
from relational_lab.experiments import stranded_p_stimuli as sp

ITEMS = sp.generate_items()
DEV = [i for i in ITEMS if i.split == "dev"]


def test_examples():
    it = sp.build_item("Leon", "Mira", "to", "give", "gave", "the book", "do", "stranded")
    assert it.sentence == "Leon gave Mira the book."
    assert it.question == "Who did someone give the book to?"
    assert (it.answer, it.distractor, it.role_asked) == ("Mira", "Leon", "oblique")
    assert it.prompt == "Sentence: Leon gave Mira the book.\nQuestion: Who did someone give the book to?\nAnswer:"
    assert sp.build_item("Leon", "Mira", "from", "borrow", "borrowed", "the bike", "cleft_piped", "piped").sentence \
        == "It was from Leon that Leon borrowed the bike.".replace("from Leon that Leon", "from Mira that Leon")
    assert sp.build_item("Leon", "Mira", "for", "bake", "baked", "a pie", "po", "agent").question \
        == "Who baked a pie for someone?"
    assert sp.build_item("Leon", "Mira", "to", "give", "gave", "the book", "po", "piped").question \
        == "To whom did someone give the book?"


def test_same_families_and_splits_as_otr():
    otr = {i.family_id: (i.split, i.test_set) for i in ors.generate_items()}
    for i in ITEMS:
        assert otr[i.family_id] == (i.split, i.test_set)


def test_counts_and_balance():
    c = Counter((i.construction, i.question_voice) for i in DEV)
    assert len(c) == 6 * 7 + 6 * 7 + 4 * 6
    assert len(set(c.values())) == 1  # every cell the same size
    assert len(DEV) == 108 * next(iter(c.values()))
    assert "from_do" not in {i.construction for i in ITEMS}


def test_names_once_in_sentence_never_in_question():
    for i in ITEMS[:5000]:
        for n in (i.agent, i.patient):
            assert len(re.findall(rf"\b{n}\b", i.sentence)) == 1
            assert n not in i.question
        assert i.answer == (i.agent if i.role_asked == "agent" else i.patient)


def test_ids_unique():
    assert len({i.item_id for i in ITEMS}) == len(ITEMS)


def test_passive_forms():
    b = lambda form, q: sp.build_item("Leon", "Mira", "to", "give", "gave", "the book", form, q)
    assert b("pass_theme", "pass_agent_stranded").sentence == "The book was given to Mira by Leon."
    assert b("pass_recip", "agent").sentence == "Mira was given the book by Leon."
    assert b("po", "pass_agent_stranded").question == "Who was the book given to someone by?"
    assert b("po", "pass_agent_piped").question == "By whom was the book given to someone?"
    assert b("po", "pass_obl_stranded").question == "Who was the book given to by someone?"
    assert b("po", "pass_obl_subject").question == "Who was given the book by someone?"
    assert b("po", "pass_agent_piped").answer == "Leon" and b("po", "pass_obl_subject").answer == "Mira"
    f = sp.build_item("Leon", "Mira", "for", "cook", "cooked", "dinner", "pass_theme", "pass_obl_stranded")
    assert f.sentence == "Dinner was cooked for Mira by Leon."
    assert not [i for i in ITEMS if i.frame == "from" and (i.form == "pass_recip" or i.question_voice == "pass_obl_subject")]


def test_v1_0_items_unchanged():
    """1.0.0 items (forms po/do/clefts x questions stranded/piped/agent) keep ids and text."""
    old = sp.build_item("Leon", "Mira", "to", "give", "gave", "the book", "do", "stranded")
    assert old.item_id == "Leon>Mira:give:to:do:oblique:stranded"
    assert old.sentence == "Leon gave Mira the book."
