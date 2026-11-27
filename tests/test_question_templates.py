from relational_lab.experiments import order_role_stimuli as ors
from relational_lab.experiments.question_templates import fill, gerund


def test_gerunds_for_all_otr_verbs():
    for lemma, _ in ors.VERBS:
        g = gerund(lemma)
        assert g.endswith("ing") and "eing" not in g, (lemma, g)
    assert gerund("praise") == "praising" and gerund("visit") == "visiting"


def test_fill():
    assert fill("Who was the {ing} done by?", "praised", "praise") == "Who was the praising done by?"
    assert fill("Who was someone {past} by?", "praised", "praise") == "Who was someone praised by?"
