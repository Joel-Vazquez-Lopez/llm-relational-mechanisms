"""OTR-10: positions and labels, planted order vs role codes, exact validity check, tiny-model smoke runs."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _o10():
    spec = importlib.util.spec_from_file_location("o10", ROOT / "scripts" / "otr10_incremental_roles.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_positions_and_labels_en():
    o10 = _o10()
    from relational_lab.experiments import order_role_stimuli as ors
    it = ors.build_item("Leon", "Mira", "praise", "praised", "passive", "agent", "active")
    pos = o10.item_positions(it)
    assert [p[0] for p in pos] == ["w0", "w1", "w2", "w3", "w4", "w5", "question_end", "answer"]
    assert [it.prompt[a:b] for _, a, b, *_ in pos[:6]] == ["Mira", "was", "praised", "by", "Leon", "."]
    assert [p[3] for p in pos] == [False, False, False, False, True, True, False, False]
    assert o10.word_labels(it) == ["P", "was", "praised", "by", "A", ".", "?", "Answer:"]
    assert [p[4] for p in pos] == ["Mira", None, None, None, "Leon", None, None, None]
    assert [p[5] for p in pos] == ["Mira"] * 4 + ["Leon"] * 4
    it = ors.build_item("Leon", "Mira", "praise", "praised", "by_cleft", "agent", "active")
    pos = o10.item_positions(it)
    assert [p[3] for p in pos][:9] == [False, False, False, False, False, True, True, True, True]
    assert [p[5] for p in pos][:4] == [None, None, None, "Leon"]


def test_positions_xl():
    o10 = _o10()
    for lg in ("es", "gl"):
        for it in o10.build_items(2026, 1, lg)[:40]:
            pos = o10.item_positions(it)
            words = [it.prompt[a:b] for lab, a, b, *_ in pos if lab.startswith("w")]
            assert words[-1] == "." and it.names[1].name in words
            assert sum(p[3] for p in pos) >= 2  # second name and period at least


def test_planted_role_vs_order_codes():
    torch = pytest.importorskip("torch")
    o10 = _o10()
    from relational_lab.experiments import order_role_stimuli as ors
    items = o10.build_items(2026, 4)
    names = list(ors.NAMES)
    torch.manual_seed(0)
    E = torch.randn(len(names), 48)
    Y = lambda its: torch.nn.functional.one_hot(torch.tensor([names.index(i.agent) for i in its]), len(names)).float()
    act = [i for i in items if i.construction == "active"]
    pas = [i for i in items if i.construction == "passive"]
    for code, expect in (("agent", 1.0), ("first", 0.0)):
        feat = lambda its: torch.stack([E[names.index(o10.target_name(i, code))] + 0.05 * torch.randn(48) for i in its])
        S = o10.ridge_scores(feat(act), Y(act), feat(pas), 0.1)
        a = torch.tensor([names.index(i.agent) for i in pas])
        p = torch.tensor([names.index(i.patient) for i in pas])
        r = torch.arange(len(pas))
        p_agent = (S[r, a] > S[r, p]).float().mean().item()
        assert abs(p_agent - expect) < 0.05, (code, p_agent)


@pytest.fixture
def tiny(monkeypatch):
    pytest.importorskip("transformers")
    from test_logit_lens import _tiny
    _tiny(monkeypatch)


def test_smoke_tiny_and_exact_validity(tiny, tmp_path):
    o10 = _o10()
    o10.main(["--model", "tiny", "--dtype", "float32", "--verbs-per-event", "2", "--layer-stride", "1",
             "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr10_*.json")).read_text())
    assert len(r["layers"]) == r["n_layers"]
    m = r["results"]["en"]["main"]["agent"]["passive"]
    assert len(m) == len(r["labels"]["en"]["passive"]) == 8 and all(len(row) == len(r["layers"]) for row in m)
    assert all(v is None or 0 <= v <= 1 for row in m for v in row)
    # identical left context at the first name: an agent probe cannot tell active from passive
    # (not exact: the two constructions' test items use different verbs, hence different CV probes)
    assert all(abs(v - 1) < 0.1 for v in r["validity_first_name_active_plus_passive_agent"])
    assert {"relative_active", "relative_passive", "subject_cleft"} <= set(r["results"]["en"]["active"]["agent"])
    assert (tmp_path / (next(tmp_path.glob("otr10_*.json")).stem + "_main.png")).exists()
    # no identity artefact: at the names themselves the agent probe is not pinned to the current name
    act = r["results"]["en"]["main"]["agent"]["active"]  # A praised P .
    assert all(abs(v - 0.5) < 0.25 for v in act[0] + act[2]), (act[0], act[2])
    rec = r["results"]["en"]["main"]["recent"]["object_cleft"]  # undefined before the first name
    assert rec[0] == [None] * len(r["layers"]) and rec[1] == [None] * len(r["layers"])


def test_smoke_tiny_crosslingual(tiny, tmp_path):
    o10 = _o10()
    o10.main(["--model", "tiny", "--dtype", "float32", "--verbs-per-event", "1", "--layer-stride", "2",
             "--langs", "en,es,gl", "--out-dir", str(tmp_path)])
    r = json.loads(next(tmp_path.glob("otr10_*_en-es-gl_*.json")).read_text())
    assert set(r["results"]) == {"en", "es", "gl"}
    gl = r["results"]["gl"]["active"]["agent"]
    assert "ovs_clitic" in gl and all(v is not None for row in gl["ovs_clitic"] for v in row)
    assert r["labels"]["es"]["ovs_clitic"][0] == "A" and r["labels"]["es"]["ovs_clitic"][1] == "P"
    assert (tmp_path / (next(tmp_path.glob("otr10_*.json")).stem + "_active_transfer.png")).exists()


def test_center_name_slots_removes_slot_dependent_identity():
    torch = pytest.importorskip("torch")
    o10 = _o10()
    torch.manual_seed(0)
    E = torch.randn(6, 16)
    rows, groups = [], []
    for strength in (0.5, 3.0):  # identity is encoded more strongly in the second slot
        slot = {}
        for _ in range(60):
            n = int(torch.randint(0, 6, (1,)))
            slot.setdefault(n, []).append(len(rows))
            rows.append(strength * E[n] + 0.01 * torch.randn(16))
        groups.append((torch.tensor([r for g in slot.values() for r in g]), [torch.tensor(g) for g in slot.values()]))
    X = torch.stack(rows)
    o10.center_name_slots(X, groups)
    for at, gs in groups:
        means = torch.stack([X[g].mean(0) for g in gs])
        assert (means - means.mean(0)).abs().max() < 1e-4
