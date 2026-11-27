from relational_lab.experiments import multilingual_stimuli as xl
from relational_lab.experiments import order_role_stimuli as ors


def b(lang, cons, role="agent", voice="active", a="Daniel", p="Laura"):
    lemma, pret, stem = xl.VERBS[lang][0]
    return xl.build_item(lang, a, p, lemma, pret, stem, cons, role, voice)


def test_spanish_sentences_as_checked():
    assert b("es", "active").sentence == "Daniel elogió a Laura."
    assert b("es", "passive").sentence == "Laura fue elogiada por Daniel."
    assert b("es", "object_cleft").sentence == "Fue a Laura a quien elogió Daniel."
    assert b("es", "by_cleft").sentence == "Fue por Daniel por quien Laura fue elogiada."
    assert b("es", "subject_cleft").sentence == "Fue Daniel quien elogió a Laura."
    assert b("es", "ovs_clitic").sentence == "A Laura la elogió Daniel."
    assert b("es", "ovs_clitic", a="Laura", p="Daniel").sentence == "A Daniel lo elogió Laura."
    assert b("es", "passive", a="Laura", p="Daniel").sentence == "Daniel fue elogiado por Laura."


def test_galician_sentences_as_checked():
    assert b("gl", "active").sentence == "Daniel eloxiou a Laura."
    assert b("gl", "passive").sentence == "Laura foi eloxiada por Daniel."
    assert b("gl", "object_cleft").sentence == "Foi a Laura a quen eloxiou Daniel."
    assert b("gl", "by_cleft").sentence == "Foi por Daniel por quen Laura foi eloxiada."
    assert b("gl", "ovs_clitic").sentence == "A Laura eloxiouna Daniel."
    assert b("gl", "ovs_clitic", a="Laura", p="Daniel").sentence == "A Daniel eloxiouno Laura."


def test_questions_and_answers():
    assert b("es", "active", "agent", "active").question == "¿Quién elogió a alguien?"
    assert b("es", "active", "patient", "active").question == "¿A quién elogió alguien?"
    assert b("es", "active", "agent", "passive").question == "¿Por quién fue elogiado alguien?"
    assert b("gl", "active", "patient", "passive").question == "Quen foi eloxiado por alguén?"
    it = b("es", "ovs_clitic", "patient", "active")
    assert (it.answer, it.distractor) == ("Laura", "Daniel")
    assert it.prompt.endswith("\nRespuesta:") and b("gl", "active").prompt.endswith("\nResposta:")


def test_flags_decorrelate_like_english():
    for lang in xl.LANGS:
        for cons, (first, subj) in xl._FLAGS.items():
            it = b(lang, cons)
            first_name = it.names[0]
            assert first_name.name == (it.agent if first == "agent" else it.patient), (lang, cons)
            subject = next(n for n in it.names if n.is_subject)
            assert subject.name == (it.agent if subj == "agent" else it.patient), (lang, cons)
            assert it.prompt[first_name.prompt_char_start:first_name.prompt_char_end] == first_name.name
            spans = dict((k, (s, e)) for k, s, e in it.spans)
            assert it.prompt[slice(*spans["period"])] == "."


def test_full_generation_and_splits():
    otr = {i.family_id: (i.split, i.test_set) for i in ors.generate_items()}
    for lang in xl.LANGS:
        items = xl.generate_items(lang)
        assert len({i.item_id for i in items}) == len(items)
        assert all(otr[i.family_id] == (i.split, i.test_set) for i in items)
        assert {i.construction for i in items} == set(xl.XL_CONSTRUCTIONS)
        for it in items[:3000]:
            assert it.agent not in it.question and it.patient not in it.question
