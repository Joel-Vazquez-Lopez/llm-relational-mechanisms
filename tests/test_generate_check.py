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
