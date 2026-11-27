"""WIKI-ROLES extraction (needs spaCy + en_core_web_sm; skipped otherwise) and OTR-09 wiring."""
import importlib.util
import json
from pathlib import Path

import pytest

from relational_lab.experiments import wiki_roles as wr

ROOT = Path(__file__).resolve().parents[1]

ARTICLE = """Early life
In 1998, John Carlton was defeated by Mary Montague in the regional election after a long campaign.
Later that year, Peter Hawthorne hired Susan Fenwick as an assistant at the museum.
Robert Prescott appointed James Ellery to the bench, and Ellery thanked Prescott in a speech.
The committee praised the report.
Whitcombe was succeeded by Ashdown as chairman of the board in 2004.
"""


def _nlp():
    spacy = pytest.importorskip("spacy")
    pytest.importorskip("lemminflect")
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")


def test_prefilter_and_extract():
    nlp = _nlp()
    forms = wr.verb_forms()
    cands = list(wr.candidate_sentences(ARTICLE, wr.surface_forms(forms)))
    assert "The committee praised the report." not in cands
    recs = [r for s in cands for r in wr.extract_from_doc(nlp(s), forms, {"article_id": "1", "title": "T"})]
    by = {(r["agent"], r["patient"], r["kind"]) for r in recs}
    assert ("Mary Montague", "John Carlton", "passive") in by
    assert ("Peter Hawthorne", "Susan Fenwick", "active") in by
    # repeated names (Prescott/Ellery appear twice) are dropped as ambiguous
    assert not any("Prescott" in r["agent"] or "Ellery" in r["agent"] for r in recs)
    for r in recs:
        assert r["sentence"][slice(*r["agent_char"])] == r["agent"]
        assert r["sentence"][slice(*r["patient_char"])] == r["patient"]


def _rec(kind="passive"):
    s = "In 1998, Carlton was defeated by Montague in the regional election." if kind == "passive" else \
        "Later that year Hawthorne hired Fenwick as an assistant."
    a, p = ("Montague", "Carlton") if kind == "passive" else ("Hawthorne", "Fenwick")
    return {"sentence": s, "kind": kind, "lemma": "defeat" if kind == "passive" else "hire",
            "past": "defeated" if kind == "passive" else "hired", "pp": "defeated" if kind == "passive" else "hired",
            "ing": "defeating" if kind == "passive" else "hiring", "agent": a, "patient": p,
            "agent_char": [s.index(a), s.index(a) + len(a)], "patient_char": [s.index(p), s.index(p) + len(p)],
            "article_id": "9", "title": "X"}


def test_items_real_and_neutral():
    recs = [_rec("passive"), _rec("active")]
    for names in ("real", "neutral"):
        pairs = wr.generate_pairs(recs, "Who was someone {pp} by?", "Who was the {ing} done by?",
                                  "Reply with one name only.", names)
        assert len(pairs) == 2
        for b, s in pairs:
            assert b.sentence == s.sentence and b.agent == s.agent and b.answer == b.agent
            assert b.prompt.endswith("Reply with one name only.\nAnswer:")
            if names == "neutral":
                assert "Montague" not in b.sentence and "Hawthorne" not in b.sentence
            for t in b.names:
                assert b.prompt[t.prompt_char_start:t.prompt_char_end] == t.name
        assert {b.construction for b, _ in pairs} == set(wr.CONSTRUCTIONS)


def test_extract_script_end_to_end(tmp_path):
    _nlp()
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"id": ["1", "2"], "title": ["A", "B"], "text": [ARTICLE, ARTICLE.replace("1998", "1999")]}),
                   tmp_path / "train-00000-of-00001.parquet")
    spec = importlib.util.spec_from_file_location("we", ROOT / "scripts" / "wiki_extract_roles.py")
    we = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(we)
    out = tmp_path / "w.jsonl"
    we.main(["--parquet", str(tmp_path / "*.parquet"), "--out", str(out), "--spacy-model", "en_core_web_sm",
             "--n-process", "1", "--batch", "3", "--n-active", "5", "--n-passive", "5"])
    recs = wr.load_records(out)
    assert recs and {r["kind"] for r in recs} == {"active", "passive"}
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["kept"]["passive"] >= 1


def test_v02_name_filters_and_confirmation():
    assert wr.looks_like_person("Mary Montague") and wr.looks_like_person("Herbert W.")
    assert wr.looks_like_person("Ludwig van Beethoven")
    assert not wr.looks_like_person("Wigan Highfield FC") and not wr.looks_like_person("Labour Party")
    assert not wr.looks_like_person("Louis XIV 2") and not wr.looks_like_person("the Senate")
    recs = [_rec("passive"), _rec("active")]
    recs[1] = dict(recs[1], title="org")

    def fake_ner(sents):
        out = []
        for s in sents:
            es = [{"entity_group": "PER", "start": s.index(n), "end": s.index(n) + len(n)}
                  for n in ("Montague", "Carlton", "Hawthorne") if n in s]
            if "Fenwick" in s:  # tagged as an organisation by the second model
                es.append({"entity_group": "ORG", "start": s.index("Fenwick"), "end": s.index("Fenwick") + 7})
            out.append(es)
        return out

    kept = wr.confirm_people(recs, fake_ner)
    assert [r["kind"] for r in kept] == ["passive"]


def test_shared_word_pairs_dropped():
    nlp = _nlp()
    forms = wr.verb_forms()
    doc = nlp("In 1999, Wigan Highfield beat Wigan Johnson in the final.")
    assert not wr.extract_from_doc(doc, forms)


def test_v021_initials_not_split_and_truncated_dropped(tmp_path):
    s = list(wr.candidate_sentences("In 1843, Poe contacted Thomas C. Clarke about the magazine. "
                                    "Later Poe met Anna Smith in Boston.", {"contacted", "met"}))
    assert s[0].endswith("magazine.") and len(s) == 2
    bad = dict(_rec("passive"), sentence="Paul Galetto was replaced by Donald F.", agent="Donald F.")
    f = tmp_path / "w.jsonl"
    f.write_text(json.dumps(bad) + "\n" + json.dumps(_rec("active")) + "\n")
    assert [r["kind"] for r in wr.load_records(f)] == ["active"]
