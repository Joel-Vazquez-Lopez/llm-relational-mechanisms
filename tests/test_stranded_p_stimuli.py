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
    assert len(c) == 4 * 3 + 4 * 3 + 3 * 3
    assert len(set(c.values())) == 1  # every cell the same size
    assert len(DEV) == 33 * next(iter(c.values()))
    assert "from_do" not in {i.construction for i in ITEMS}


def test_names_once_in_sentence_never_in_question():
    for i in ITEMS[:5000]:
        for n in (i.agent, i.patient):
            assert len(re.findall(rf"\b{n}\b", i.sentence)) == 1
            assert n not in i.question
        assert i.answer == (i.agent if i.role_asked == "agent" else i.patient)


def test_ids_unique():
    assert len({i.item_id for i in ITEMS}) == len(ITEMS)
