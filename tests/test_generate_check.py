import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("gc", Path(__file__).resolve().parents[1] / "scripts" / "otr_generate_check.py")
gc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gc)
N = ["Julia", "Lisa"]


def test_direct():
    assert gc.extract_answer("Julia", N) == ("direct", "Julia")
    assert gc.extract_answer(" **Lisa**.\nbecause", N) == ("direct", "Lisa")


def test_quoted_sentence_ignored():
    out = 'In the sentence "Lisa chased Julia," someone chased Julia.'
    assert gc.extract_answer(out, N, "Lisa chased Julia.", "Who did someone chase?") == ("inferred", "Julia")


def test_both_names_is_ambiguous():
    assert gc.extract_answer("Julia was chased by Lisa.", N)[0] == "ambiguous"


def test_none():
    assert gc.extract_answer("The sentence is a cleft sentence", N) == ("none", None)


R = dict(agent="Lisa", patient="Julia")


def test_passive_statement_names_asked_role():
    out = "Julia was chased by Lisa."
    assert gc.extract_answer(out, N, "Lisa chased Julia.", "Who was someone chased by?", role="agent", **R) == ("statement", "Lisa")
    assert gc.extract_answer(out, N, role="patient", **R) == ("statement", "Julia")


def test_reversed_statement_is_wrong_answer():
    assert gc.extract_answer("Lisa was chased by Julia.", N, role="agent", **R) == ("statement", "Julia")


def test_active_and_cleft_statements():
    assert gc.extract_answer("Lisa chased Julia.", N, role="agent", **R) == ("statement", "Lisa")
    assert gc.extract_answer("It was by Lisa that Julia was chased.", N, role="patient", **R) == ("statement", "Julia")


def test_embedded_quote_not_a_statement():
    out = 'In the sentence "Lisa chased Julia," someone chased Julia.'
    assert gc.extract_answer(out, N, "Lisa chased Julia.", "Who did someone chase?", role="patient", **R) == ("inferred", "Julia")


D = dict(agent="Leon", patient="Mira")
NM = ["Leon", "Mira"]


def test_three_place_statements():
    for out in ("Leon gave the book to Mira.", "Leon gave Mira the book.", "Mira was given the book by Leon.",
                "The book was given to Mira by Leon.", "It was Mira that Leon gave the book to.",
                "It was to Mira that Leon gave the book.", "Leon baked dinner for Mira.",
                "Leon borrowed the bike from Mira."):
        assert gc.extract_answer(out, NM, role="oblique", **D) == ("statement", "Mira"), out
        assert gc.extract_answer(out, NM, role="agent", **D) == ("statement", "Leon"), out
    assert gc.extract_answer("Mira gave Leon the book.", NM, role="oblique", **D) == ("statement", "Leon")


def test_perfect_and_indefinite_statements():
    q = "Who has someone been chased by?"
    assert gc.extract_answer("Julia has been chased by Lisa.", N, "Lisa chased Julia.", q, role="agent", **R) == ("statement", "Lisa")
    assert gc.extract_answer("Julia had been chased by Lisa.", N, role="agent", **R) == ("statement", "Lisa")
    assert gc.extract_answer("Julia was a person chased by Lisa.", N, role="agent", **R) == ("statement", "Lisa")
    assert gc.extract_answer("Lisa has been chased by Julia.", N, role="agent", **R) == ("statement", "Julia")


def test_question_frame_answer():
    q = "Who had someone been chased by?"
    assert gc.extract_answer("Lisa had someone been chased by.", N, "Lisa chased Julia.", q, role="agent", **R) == ("frame", "Lisa")
    assert gc.extract_answer("Julia had someone been chased by.", N, "Lisa chased Julia.", q, role="agent", **R) == ("frame", "Julia")
    # not a frame answer when the rest differs
    assert gc.extract_answer("Lisa had been chased by someone.", N, "Lisa chased Julia.", q, role="agent", **R)[0] != "frame"


def test_relative_statement():
    assert gc.extract_answer("Julia was the person who was chased by Lisa.", N, role="agent", **R) == ("statement", "Lisa")
    assert gc.extract_answer("Julia was the one who got chased by Lisa.", N, role="agent", **R) == ("statement", "Lisa")
    assert gc.extract_answer("Lisa had someone been chased by Julia.", N, role="agent", **R)[0] == "ambiguous"


def test_theme_passive_variants():
    for out in ("The keys were taken from Mira by Leon.", "Dinner was cooked for Mira by Leon.",
                "A letter was received by Leon from Mira.", "The song was learned by Leon from Mira."):
        assert gc.extract_answer(out, NM, role="agent", **D) == ("statement", "Leon"), out
    assert gc.extract_answer("The photo was shown to Leon by Mira.", NM, role="agent", **D) == ("statement", "Mira")


def test_other_person_statements():
    for out in ("Julia was the other person chased by Lisa.", "The other person chased by Lisa was Julia.",
                "The other person, Julia, was chased by Lisa.", "Lisa was the other person who chased Julia.",
                "Lisa was the one who chased Julia."):
        assert gc.extract_answer(out, N, role="agent", **R) == ("statement", "Lisa"), out
    assert gc.extract_answer("Lisa was the other person chased by Julia.", N, role="agent", **R) == ("statement", "Julia")


def test_spanish_galician_statements():
    for out in ("Laura fue elogiada por Daniel.", "Daniel elogió a Laura.", "A Laura la elogió Daniel.",
                "Laura foi eloxiada por Daniel.", "Daniel eloxiou a Laura.", "A Laura eloxiouna Daniel.",
                "Fue Daniel quien elogió a Laura.", "Foi a Laura a quen eloxiou Daniel."):
        assert gc.extract_answer(out, ["Daniel", "Laura"], role="agent", agent="Daniel", patient="Laura") == ("statement", "Daniel"), out
    assert gc.extract_answer("Daniel fue elogiado por Laura.", ["Daniel", "Laura"], role="agent",
                             agent="Daniel", patient="Laura") == ("statement", "Laura")


def test_prepared_prompts_spanish():
    pp = gc.prepared_prompts("es", 2026, 40, "", None, "Responde con un solo nombre.")
    assert pp and all(p.startswith("Oración: ") and "Responde con un solo nombre.\nRespuesta:" in p for _, p in pp)
    assert {it.construction for it, _ in pp} <= {"active", "passive", "object_cleft", "by_cleft", "ovs_clitic"}
