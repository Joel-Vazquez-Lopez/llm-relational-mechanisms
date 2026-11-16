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
