"""Freeze scorer: synthetic result files for one model, checked against the frozen rules."""
import importlib.util
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fs", ROOT / "scripts" / "otr_freeze_score.py")
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)

M = "microsoft/phi-4"
FZ = fs.FROZEN[M]
NAMES = ["Ann", "Bob", "Cat", "Dan", "Eve", "Fay", "Gus", "Hal"]
PAIRS = [(a, b) for i, a in enumerate(NAMES) for b in NAMES[i + 1:]][:24]


def _w(d, name, obj):
    (d / name).write_text(json.dumps(obj))


def _gen(d, q, p_correct, rng):
    rows = []
    for a, b in PAIRS:
        for c in fs.MAIN:
            rows.append({"item_id": f"{a}>{b}:help:{c}:agent:active", "construction": c,
                         "correct": rng.random() < p_correct})
    _w(d, f"otrgen_{FZ['slug']}_test0_q-{q}_20261201T000000.json",
       {"question_role": "agent", "rows": rows, "split": "test0"})


def _otr03(d, aq, role_effect, rng, correct):
    layers = [FZ["role"]]
    recs = []
    for a, b in PAIRS:
        fam = "|".join(sorted((a, b)))
        for base in ("active", "passive"):
            for k in ("first_name", "second_name"):
                for j, x in (("first_name", {"order": 1, "function": 0, "role": 0}),
                             ("second_name", {"order": 0, "function": 1, "role": 0}),
                             ("first_name", {"order": 0, "function": 0, "role": 1}),
                             ("second_name", {"order": 0, "function": 0, "role": 0})):
                    y = role_effect * x["role"] + 0.05 * x["function"] + rng.gauss(0, 0.2)
                    recs.append({"base_id": f"{a}>{b}:help:{base}:agent:active", "family_id": fam, "k": k, "j": j,
                                 "role_asked": "agent", "x": x, "y": [y], "base_correct": correct})
    tag = "_aq" if aq else ""
    _w(d, f"otr03_{FZ['slug']}_test0{tag}_20261201T000000.json",
       {"layers": layers, "agent_question": "Who was someone {past} by?" if aq else None, "records": recs})


def _items(rng, mean):
    return [{"item_id": f"{a}>{b}:help:active:agent:active", "family_id": "|".join(sorted((a, b))),
             "repair": mean + rng.gauss(0, .05)} for a, b in PAIRS]


def test_scorer_all_claims(tmp_path):
    rng = random.Random(1)
    _gen(tmp_path, fs.FAIL_Q, .2, rng)
    _gen(tmp_path, fs.WORK_Q, .95, rng)
    _otr03(tmp_path, False, 0.6, rng, True)
    _otr03(tmp_path, True, -0.5, rng, False)
    h = FZ["handover"]
    layers = list(range(0, 40, 2))
    rep = [0.0 if L < h else 0.9 for L in layers]
    tr = {"all_items": {"repair": .6, "items": _items(rng, .6)},
          "random_same_layers": {"repair": .02, "items": _items(rng, .02), "n_draws": 5},
          "other_failing": {"repair": .01}, "random_nudge": {"repair": .03}}
    _w(tmp_path, f"otr09_{FZ['slug']}_test0_20261201T000000.json",
       {"residual_layers": layers, "residual": {"all": {"repair": rep}}, "transfer": {"by_k": {"20": tr}}})
    curves = {f"{a}>{b}:help:active:agent:active": {"by_from_failing": [0.4 + rng.gauss(0, .05)],
                                                     "by_from_patient": [-0.01 + rng.gauss(0, .05)]}
              for a, b in PAIRS}
    _w(tmp_path, f"otr14_{FZ['slug']}_test0_20261201T000000.json",
       {"layers": [FZ["by"]], "items": [{"item_id": i} for i in curves], "curves": curves})
    table, inputs = fs.score(str(tmp_path), "test0", models=[M])
    row = table[M]
    assert row["C1"]["verdict"] == "PASS"
    assert row["C2"]["verdict"] == "PASS" and abs(row["C2"]["role"] - 0.6) < 0.1
    assert row["C3"]["verdict"] == "PASS" and row["C3"]["role_failing_wrong"] < 0
    assert row["C4"]["verdict"] == "PASS" and row["C4"]["first_crossing"] == h
    assert row["C5"]["verdict"] == "PASS"
    assert row["C6"]["verdict"] == "PASS"
    assert row["C8"]["verdict"] == "FAIL"          # near-zero effect: CI includes 0
    assert row["C7"]["verdict"] == "no claim"      # phi-4 has no C7 claim
    assert len(inputs[M]) == 6


def test_scorer_rules_not_evaluable(tmp_path):
    rng = random.Random(2)
    _otr03(tmp_path, False, 0.6, rng, True)
    h = FZ["handover"]
    layers = list(range(0, 40, 2))
    rep = [0.9 if L >= h - 6 else 0.0 for L in layers]   # crosses too early -> C4 fails
    tr = {"all_items": {"repair": .6, "items": _items(rng, .6)},
          "random_same_layers": {"repair": .02, "items": _items(rng, .02)},
          "other_failing": {"repair": .30}, "random_nudge": {"repair": .03}}   # control too high -> C5 fails
    _w(tmp_path, f"otr09_{FZ['slug']}_test0_20261201T000000.json",
       {"residual_layers": layers, "residual": {"all": {"repair": rep}}, "transfer": {"by_k": {"20": tr}}})
    row = fs.score(str(tmp_path), "test0", models=[M])[0][M]
    assert row["C1"]["verdict"].startswith("not evaluable")
    assert row["C3"]["verdict"].startswith("not evaluable")
    assert row["C4"]["verdict"] == "FAIL"
    assert row["C5"]["verdict"] == "FAIL"
    assert row["C6"]["verdict"].startswith("not evaluable")


def test_family_of():
    assert fs.family_of("Bob>Ann:help:active:agent:active") == "Ann|Bob"


def test_scorer_xl_addendum(tmp_path):
    """FREEZE_XL.md: es file tags, the es/gl failing run (passive agent question, no --agent-question),
    C2 scored on role alone, C4 against the es handover layer, C5b from the other_working control."""
    rng = random.Random(3)
    lang = "es"
    rows = []
    for a, b in PAIRS:
        for c in fs.XL_REPAIR[lang]:
            for voice, p in (("active", .97), ("passive", .3)):
                rows.append({"item_id": f"es:{a}>{b}:saludar:{c}:agent:{voice}", "construction": c, "role": "agent",
                             "voice": voice, "correct": rng.random() < p})
    _w(tmp_path, f"otrgen_{FZ['slug']}_test0_{lang}_20261201T000000.json",
       {"instruction": fs.XL_INSTR[lang], "rows": rows})
    # an English test0 file that must NOT be picked up for es ("test0" contains "es")
    _otr03(tmp_path, False, -2.0, rng, True)

    def otr03_xl(failing, role_effect, correct, stamp):
        recs = []
        for a, b in PAIRS:
            fam = "|".join(sorted((a, b)))
            for base in ("active", "passive"):
                for k in ("first_name", "second_name"):
                    for j, x in (("first_name", {"order": 1, "function": 0, "role": 0}),
                                 ("second_name", {"order": 0, "function": 1, "role": 0}),
                                 ("first_name", {"order": 0, "function": 0, "role": 1}),
                                 ("second_name", {"order": 0, "function": 0, "role": 0})):
                        # function as large as role: English C2 would fail on role - function, XL-C2 must not
                        y = role_effect * (x["role"] + x["function"]) + rng.gauss(0, 0.2)
                        recs.append({"base_id": f"es:{a}>{b}:saludar:{base}:agent:passive", "family_id": fam,
                                     "k": k, "j": j, "role_asked": "agent", "x": x, "y": [y],
                                     "base_correct": correct})
        _w(tmp_path, f"otr03_{FZ['slug']}_test0_{lang}_{stamp}.json",
           {"layers": [FZ["role"]], "agent_question": None, "include_incorrect": failing,
            "question_voices": ["passive"] if failing else ["active"],
            "roles": ["agent"] if failing else ["agent", "patient"], "records": recs})
    otr03_xl(False, 0.5, True, "20261201T000001")
    otr03_xl(True, -0.4, False, "20261201T000002")
    h = fs.XL_HANDOVER[lang][M]
    layers = list(range(0, 40, 2))
    tr = {"all_items": {"repair": .6, "items": _items(rng, .6)},
          "random_same_layers": {"repair": .02, "items": _items(rng, .02), "n_draws": 5},
          "other_failing": {"repair": .3}, "random_nudge": {"repair": .0},
          "mean_failing": {"repair": .3}, "other_working": {"repair": .25, "items": _items(rng, .25)}}
    _w(tmp_path, f"otr09_{FZ['slug']}_test0_{lang}_20261201T000000.json",
       {"residual_layers": layers, "residual": {"all": {"repair": [0.0 if L < h else 0.9 for L in layers]}},
        "transfer": {"by_k": {"20": tr}}})
    row = fs.score(str(tmp_path), "test0", models=[M], stimuli=lang)[0][M]
    assert row["C1"]["verdict"] == "PASS" and row["C1"]["estimate"] > .5
    assert row["C2"]["verdict"] == "PASS" and row["C2"]["role"] > 0      # English file (role -2) not used
    assert row["C3"]["verdict"] == "PASS"
    assert row["C4"]["verdict"] == "PASS" and row["C4"]["frozen"] == h
    assert row["C5"]["verdict"] == "PASS"                                 # other_failing .3 not scored for XL
    assert row["C5b"]["verdict"] == "PASS" and abs(row["C5b"]["top_minus_other_working"] - .35) < .05


def test_scorer_70b_set_only_when_named(tmp_path):
    """FREEZE_70B.md: the two 70B models are scored only with --set 70b / --models; the default stays the core five."""
    assert set(fs.FROZEN_70B).isdisjoint(fs.FROZEN)
    for m, fz in fs.FROZEN_70B.items():
        assert fz["by"] < fz["role"] < fz["handover"] < fz["n_layers"] and fz["rf"].startswith(f"otr09_{fz['slug']}_dev_")
    core = fs.main(["--results", str(tmp_path), "--split", "test0"])
    assert set(core["table"]) == set(fs.FROZEN)
    big = fs.main(["--results", str(tmp_path), "--split", "test0", "--set", "70b"])
    assert set(big["table"]) == set(fs.FROZEN_70B) and set(big["frozen"]) == set(fs.FROZEN_70B)
    assert all(r["C1"]["verdict"].startswith("not evaluable") for r in big["table"].values())  # nothing run yet
