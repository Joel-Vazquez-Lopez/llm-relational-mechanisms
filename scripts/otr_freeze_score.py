#!/usr/bin/env python3
"""OTR freeze scorer: pass/fail table for the pre-registered claims C1-C8 (FREEZE.md).

Reads result JSONs from a results folder, picks the newest matching file per claim and model,
and applies the frozen rules: estimates at the frozen layers, 95% percentile intervals from a
family bootstrap (2000 resamples, seed 2026, paired where two quantities are compared),
minimum counts, and "not evaluable" when a required file, field or layer is missing.

    python scripts/otr_freeze_score.py --results results --split test0 --out freeze_score_test0.json
    python scripts/otr_freeze_score.py --results results --split dev     # dry run on dev outputs

Families are unordered name pairs ("Alice|Frank"). For OTR-03 the within-(base, slot) regression
is refitted on every resample (weighted by how often each family was drawn, which is the same as
concatenating the drawn families).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

SCORER_VERSION = "1.2.0"  # 1.2.0: 70B addendum (FREEZE_70B.md); 1.1.0: es/gl addendum; core scoring unchanged
N_BOOT, SEED = 2000, 2026
FAIL_Q, WORK_Q = "who-was-someone-past-by", "who-was-the-ing-done-by"
MAIN = ("active", "passive", "object_cleft", "by_cleft")
CODES = ("order", "function", "role")

# Frozen per-model settings (FREEZE.md section 4). Never re-chosen.
FROZEN = {
    "google/gemma-4-31B-it": dict(slug="google__gemma-4-31B-it", n_layers=60, handover=40, role=28, by=28, c7=True,
                                  rf="otr09_google__gemma-4-31B-it_dev_20261118T164326.json"),
    "Qwen/Qwen3.8-27B": dict(slug="Qwen__Qwen3.8-27B", n_layers=64, handover=54, role=29, by=30, c7=True,
                             rf="otr09_Qwen__Qwen3.8-27B_dev_20261118T162630.json"),
    "microsoft/phi-4": dict(slug="microsoft__phi-4", n_layers=40, handover=22, role=15, by=8, c7=False,
                            rf="otr09_microsoft__phi-4_dev_20261119T093535.json"),
    "allenai/Olmo-3.1-32B-Instruct": dict(slug="allenai__Olmo-3.1-32B-Instruct", n_layers=64, handover=26, role=18,
                                          by=8, c7=False,
                                          rf="otr09_allenai__Olmo-3.1-32B-Instruct_dev_20261119T095922.json"),
    "ibm-granite/granite-4.2-30b": dict(slug="ibm-granite__granite-4.2-30b", n_layers=64, handover=52, role=37,
                                        by=28, c7=True, rf="otr09_ibm-granite__granite-4.2-30b_dev_20261119T140015.json"),
}
# 70B addendum (FREEZE_70B.md), frozen 2026-11-28 from the dev pass on branch paper2/70b-dev-results @ 56b7651,
# before any 70B test command. Same rules as FREEZE.md section 4. Scored only when named (--set 70b or --models),
# so the default (core five) report is unchanged.
FROZEN_70B = {
    "meta-llama/Llama-3.3-70B-Instruct": dict(slug="meta-llama__Llama-3.3-70B-Instruct", n_layers=80, handover=38,
                                              role=34, by=14, c7=True,
                                              rf="otr09_meta-llama__Llama-3.3-70B-Instruct_dev_20261127T121453.json"),
    "Qwen/Qwen2.5-72B-Instruct": dict(slug="Qwen__Qwen2.5-72B-Instruct", n_layers=80, handover=58, role=48, by=30,
                                      c7=True, rf="otr09_Qwen__Qwen2.5-72B-Instruct_dev_20261127T121538.json"),
}
ALL_FROZEN = {**FROZEN, **FROZEN_70B}
OTHER_SETS = ("bx-", "_es_", "_gl_", "_sv_", "ud-en", "wiki-", "_aq_", "_instr", "byctl", "src2")

# Spanish/Galician addendum (FREEZE_XL.md), frozen 2026-11-26 before any es/gl test run.
XL = ("es", "gl")
# Handover layer per language: the first layer with residual repair >= .5 in the dev es/gl runs
# (otr09_*_dev_{es,gl}_20261121*.json). Not the English layer: dev already showed Gemma and Qwen
# shift in Spanish/Galician, so the English layer is reported descriptively only.
XL_HANDOVER = {
    "es": {"google/gemma-4-31B-it": 46, "Qwen/Qwen3.8-27B": 50, "microsoft/phi-4": 22,
           "allenai/Olmo-3.1-32B-Instruct": 24, "ibm-granite/granite-4.2-30b": 52},
    "gl": {"google/gemma-4-31B-it": 46, "Qwen/Qwen3.8-27B": 50, "microsoft/phi-4": 22,
           "allenai/Olmo-3.1-32B-Instruct": 26, "ibm-granite/granite-4.2-30b": 52},
}


def stim_tag_in(stimuli, name):
    """True if the filename carries the stimulus tag as its own field (``_es_``, ``_bx-transitive_``)."""
    return f"_{stimuli}_" in name


def is_failing_run(D, stimuli=None):
    """OTR-03 failing-question run. English/BX: --agent-question set. es/gl: the failing question is the
    passive agent question, run with --question-voices passive --roles agent --include-incorrect."""
    if D.get("agent_question") is not None:
        return True
    return (stimuli in XL and bool(D.get("include_incorrect")) and list(D.get("question_voices") or []) == ["passive"]
            and list(D.get("roles") or []) == ["agent"])


# ----------------------------------------------------------------------------- helpers
def family_of(item_id: str) -> str:
    names = item_id.split(":")[0].split(">")
    return "|".join(sorted(names))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def newest(paths):
    return sorted(paths, key=lambda p: os.path.basename(p).rsplit("_", 1)[-1])[-1] if paths else None


def ci(values):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) < 0.9 * N_BOOT and len(v) < 10:   # fits failed on (almost) every resample
        return float("nan"), float("nan")
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def family_counts(families, rng, n_boot=N_BOOT):
    """n_boot x n_families matrix of draw counts (sampling families with replacement)."""
    k = len(families)
    idx = rng.integers(0, k, size=(n_boot, k))
    out = np.zeros((n_boot, k))
    for b in range(n_boot):
        out[b] = np.bincount(idx[b], minlength=k)
    return out


def boot_mean(fam_sum, fam_n, counts):
    """Bootstrap distribution of an item mean, given per-family sums and counts."""
    num, den = counts @ fam_sum, counts @ fam_n
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def item_mean_test(items, rng, n_boot=N_BOOT):
    """items: list of (family, value). Returns estimate, CI, n_items, n_families."""
    fams = sorted({f for f, _ in items})
    pos = {f: i for i, f in enumerate(fams)}
    s, n = np.zeros(len(fams)), np.zeros(len(fams))
    for f, v in items:
        s[pos[f]] += v
        n[pos[f]] += 1
    est = float(s.sum() / n.sum())
    dist = boot_mean(s, n, family_counts(fams, rng, n_boot))
    return est, ci(dist[~np.isnan(dist)]), len(items), len(fams)


def load(path):
    with open(path) as f:
        return json.load(f)


def verdict(ok: bool | None, reason: str = "") -> str:
    if ok is None:
        return "not evaluable" + (f" ({reason})" if reason else "")
    return "PASS" if ok else "FAIL"


# ----------------------------------------------------------------------------- finders
def find(res, pattern, exclude=OTHER_SETS, keep=None):
    paths = [p for p in glob.glob(os.path.join(res, pattern))
             if not any(t in os.path.basename(p) for t in exclude) and (keep is None or keep(p))]
    return newest(paths)


def split_tag(split):  # filenames: <exp>_<model>_<split>[...]_<timestamp>.json
    return split


# ----------------------------------------------------------------------------- claims
def c1(res, m, fz, split, rng):
    fail = find(res, f"otrgen_{fz['slug']}_{split}_q-{FAIL_Q}_*.json", exclude=())
    work = find(res, f"otrgen_{fz['slug']}_{split}_q-{WORK_Q}_*.json", exclude=())
    if not fail or not work:
        return dict(verdict=verdict(None, "file missing")), []
    F, W = load(fail), load(work)
    if F.get("question_role") != "agent" or W.get("question_role") != "agent":
        return dict(verdict=verdict(None, "not agent questions")), [fail, work]
    fr = {r["item_id"]: r for r in F["rows"] if r["construction"] in MAIN}
    wr = {r["item_id"]: r for r in W["rows"] if r["construction"] in MAIN}
    ids = sorted(set(fr) & set(wr))
    if not ids:
        return dict(verdict=verdict(None, "no paired items")), [fail, work]
    items = [(family_of(i), float(bool(wr[i]["correct"])) - float(bool(fr[i]["correct"]))) for i in ids]
    est, (lo, hi), n, nf = item_mean_test(items, rng)
    acc_f = sum(float(bool(fr[i]["correct"])) for i in ids) / len(ids)
    acc_w = sum(float(bool(wr[i]["correct"])) for i in ids) / len(ids)
    return dict(measure="working - failing generation score", estimate=est, ci=[lo, hi], n_items=n, n_families=nf,
                failing=acc_f, working=acc_w, verdict=verdict(lo > 0)), [fail, work]


XL_REPAIR = {"es": ("active", "passive", "object_cleft"), "gl": ("active", "passive")}
XL_INSTR = {"es": "Responde con un solo nombre.", "gl": "Responde cun só nome."}


def c1_xl(res, m, fz, split, rng, lang):
    """XL-C1 (FREEZE_XL.md): agent questions on the language's repair constructions, same generation run;
    accuracy of the active agent question (working) minus the passive agent question (failing); family
    bootstrap on the paired per-family difference of means."""
    f = find(res, f"otrgen_{fz['slug']}_{split}_{lang}_*.json", exclude=(),
             keep=lambda p: load(p).get("instruction") == XL_INSTR[lang])
    if not f:
        return dict(verdict=verdict(None, "file missing")), []
    rows = [r for r in load(f)["rows"] if r["role"] == "agent" and r["construction"] in XL_REPAIR[lang]]
    fam = lambda r: "|".join(sorted(r["item_id"].split(":")[1].split(">")))  # noqa: E731  es:A>B:...
    fams = sorted({fam(r) for r in rows})
    pos = {x: i for i, x in enumerate(fams)}
    S, N = np.zeros((2, len(fams))), np.zeros((2, len(fams)))
    for r in rows:
        v = 0 if r["voice"] == "active" else 1
        S[v, pos[fam(r)]] += float(r["correct"] in (True, "True"))
        N[v, pos[fam(r)]] += 1
    counts = family_counts(fams, rng)
    work, fail = S[0].sum() / N[0].sum(), S[1].sum() / N[1].sum()
    dist = boot_mean(S[0], N[0], counts) - boot_mean(S[1], N[1], counts)
    lo, hi = ci(dist)
    return dict(working=float(work), failing=float(fail), estimate=float(work - fail), ci=[lo, hi],
                n_rows=len(rows), n_families=len(fams), verdict=verdict(lo > 0)), [f]


def _otr03_groups(recs, li, codes):
    """Per group (base_id, k): centred X (with intercept column) and y, plus the group's family."""
    groups = defaultdict(list)
    for r in recs:
        y = r["y"][li]
        if y is None or (isinstance(y, float) and math.isnan(y)):
            continue
        groups[(r["base_id"], r["k"])].append(r)
    out = []
    for g in groups.values():
        if len(g) < 2:
            continue
        X = np.array([[float(r["j"] == "second_name") if c == "source_slot" else float(r["x"][c]) for c in codes]
                      for r in g])
        y = np.array([r["y"][li] for r in g], float)
        X = X - X.mean(0)
        y = y - y.mean()
        X = np.hstack([np.ones((len(g), 1)), X])
        out.append((g[0]["family_id"], X, y))
    return out


def _otr03_family_moments(groups):
    fams = sorted({f for f, _, _ in groups})
    pos = {f: i for i, f in enumerate(fams)}
    p = groups[0][1].shape[1]
    XtX, Xty = np.zeros((len(fams), p, p)), np.zeros((len(fams), p))
    for f, X, y in groups:
        XtX[pos[f]] += X.T @ X
        Xty[pos[f]] += X.T @ y
    return fams, XtX, Xty


def _solve_weighted(XtX, Xty, w):
    A, b = np.tensordot(w, XtX, 1), w @ Xty
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.full(len(b), np.nan)


def c2_c3(res, m, fz, split, rng, stimuli=None):
    """C2 from the ordinary-question OTR-03 run, C3 from the failing-question run (agent questions,
    wrong bases), both at the frozen role layer, resampled on the same families."""
    def is_aq(p):
        return is_failing_run(load(p), stimuli)

    def stim_ok(p):
        name = os.path.basename(p)
        return stim_tag_in(stimuli, name) if stimuli else not any(t in name for t in ("bx-", "_es_", "_gl_", "_sv_"))

    f2 = find(res, f"otr03_{fz['slug']}_{split}*.json", exclude=(), keep=lambda p: stim_ok(p) and not is_aq(p))
    f3 = find(res, f"otr03_{fz['slug']}_{split}*.json", exclude=(), keep=lambda p: stim_ok(p) and is_aq(p))
    used = [p for p in (f2, f3) if p]
    out = {}
    L = fz["role"]
    D2 = load(f2) if f2 else None
    D3 = load(f3) if f3 else None
    if D2 is None or L not in D2["layers"]:
        out["C2"] = dict(verdict=verdict(None, "file missing" if D2 is None else f"layer {L} not in run"))
        out["C3"] = dict(verdict=verdict(None, "needs C2"))
        return out, used
    r2 = [r for r in D2["records"] if r.get("base_correct") is not False]
    nuis2 = ("source_slot",) if len({r["role_asked"] for r in r2}) > 1 else ()
    g2 = _otr03_groups(r2, D2["layers"].index(L), CODES + nuis2)
    fams2, XtX2, Xty2 = _otr03_family_moments(g2)
    have3 = D3 is not None and L in D3["layers"]
    if have3:
        r3 = [r for r in D3["records"] if r.get("base_correct") is False and r["role_asked"] == "agent"]
        n_wrong = len({r["base_id"] for r in r3})
        g3 = _otr03_groups(r3, D3["layers"].index(L), CODES) if r3 else []
        fams3, XtX3, Xty3 = _otr03_family_moments(g3) if g3 else ([], None, None)
    fams = sorted(set(fams2) | (set(fams3) if have3 and g3 else set()))
    pos = {f: i for i, f in enumerate(fams)}
    counts = family_counts(fams, rng)
    i_role, i_fun = 1 + CODES.index("role"), 1 + CODES.index("function")

    def coefs(fs, XtX, Xty, w_all):
        idx = [pos[f] for f in fs]
        return _solve_weighted(XtX, Xty, w_all[idx])

    b2 = coefs(fams2, XtX2, Xty2, np.ones(len(fams)))
    boot2 = np.array([coefs(fams2, XtX2, Xty2, c) for c in counts])
    role, fun = float(b2[i_role]), float(b2[i_fun])
    lo_r, hi_r = ci(boot2[:, i_role][~np.isnan(boot2[:, i_role])])
    d_rf = boot2[:, i_role] - boot2[:, i_fun]
    lo_d, hi_d = ci(d_rf[~np.isnan(d_rf)])
    if math.isnan(role) or math.isnan(lo_r):
        out["C2"] = dict(verdict=verdict(None, "fit failed"))
        out["C3"] = dict(verdict=verdict(None, "needs C2"))
        return out, used
    out["C2"] = dict(layer=L, role=role, role_ci=[lo_r, hi_r], function=fun, role_minus_function=role - fun,
                     role_minus_function_ci=[lo_d, hi_d], n_families=len(fams2),
                     verdict=verdict(lo_r > 0 and (stimuli in XL or lo_d > 0)))
    if stimuli in XL:
        # FREEZE_XL.md: dev already showed grammatical function also carries signal in es/gl, so XL-C2 tests
        # role > 0 with order and function in the same fit; role - function is reported, not scored.
        out["C2"]["rule"] = "role CI > 0 (role - function reported only)"
    if not have3:
        out["C3"] = dict(verdict=verdict(None, "file missing" if D3 is None else f"layer {L} not in run"))
    elif n_wrong < 30 or not g3:
        out["C3"] = dict(n_wrong_bases=n_wrong, verdict=verdict(None, f"{n_wrong} wrong bases < 30"))
    else:
        b3 = coefs(fams3, XtX3, Xty3, np.ones(len(fams)))
        boot3 = np.array([coefs(fams3, XtX3, Xty3, c) for c in counts])
        diff = boot2[:, i_role] - boot3[:, i_role]
        lo, hi = ci(diff[~np.isnan(diff)])
        lo3, hi3 = ci(boot3[:, i_role][~np.isnan(boot3[:, i_role])])
        if math.isnan(float(b3[i_role])) or math.isnan(lo):
            out["C3"] = dict(n_wrong_bases=n_wrong, verdict=verdict(None, "fit failed"))
            return out, used
        out["C3"] = dict(layer=L, role_failing_wrong=float(b3[i_role]), role_failing_wrong_ci=[lo3, hi3],
                         c2_minus_c3=role - float(b3[i_role]), c2_minus_c3_ci=[lo, hi], n_wrong_bases=n_wrong,
                         verdict=verdict(lo > 0))
    return out, used


def c4_c5(res, m, fz, split, rng, stimuli=None):
    if split == "dev" and not stimuli:
        path = os.path.join(res, fz["rf"])
        path = path if os.path.exists(path) else None
    else:
        tag = f"_{stimuli}" if stimuli else ""
        path = find(res, f"otr09_{fz['slug']}_{split}{tag}_*.json",
                    exclude=() if stimuli else OTHER_SETS)
    if not path:
        nv = dict(verdict=verdict(None, "file missing"))
        return {"C4": nv, "C5": nv}, []
    D = load(path)
    path5 = path
    if split == "dev" and not stimuli and not (D.get("transfer") or {}).get("by_k", {}).get("20"):
        # dev dry run: the ranking runs have no transfer block; use the newest dev run with --rank-from
        def has_tr(p):
            return bool((load(p).get("transfer") or {}).get("by_k", {}).get("20"))
        alt = find(res, f"otr09_{fz['slug']}_dev_*.json", keep=has_tr)
        path5 = alt or path
    D5 = load(path5)
    out = {}
    # C4: first sampled layer with residual repair >= .5 within +-2 of the frozen layer; all layers up to
    # frozen-4 below .5.
    hand = XL_HANDOVER[stimuli][m] if stimuli in XL else (None if stimuli else fz["handover"])
    if hand is None:
        out["C4"] = dict(verdict="not tested (replication set)")
    else:
        try:
            layers, rep = D["residual_layers"], D["residual"]["all"]["repair"]
            first = next((L for L, v in zip(layers, rep) if v is not None and v >= .5), None)
            early = [v for L, v in zip(layers, rep) if L <= hand - 4 and v is not None]
            ok = first is not None and abs(first - hand) <= 2 and all(v < .5 for v in early)
            out["C4"] = dict(frozen=hand, first_crossing=first, max_repair_up_to_frozen_minus_4=
                             max(early) if early else None, verdict=verdict(ok))
            if stimuli in XL:
                out["C4"]["english_layer"] = fz["handover"]  # descriptive only
        except (KeyError, TypeError):
            out["C4"] = dict(verdict=verdict(None, "residual curve missing"))
    # C5: frozen top-20 vs 20 random heads from the same layers (paired by item), plus controls.
    used = sorted({path, path5})
    tr = (D5.get("transfer") or {}).get("by_k", {}).get("20")
    if not tr:
        out["C5"] = dict(verdict=verdict(None, "transfer k=20 missing"))
        return out, used
    top, rnd = tr.get("all_items", {}), tr.get("random_same_layers", {})
    res5 = dict(top=top.get("repair"), random=rnd.get("repair"))
    of, nu = tr.get("other_failing", {}).get("repair"), tr.get("random_nudge", {}).get("repair")
    res5.update(other_failing=of, random_nudge=nu)
    if "items" not in top or "items" not in rnd:
        res5["verdict"] = verdict(None, "no per-item values (run predates patch 0066)")
        out["C5"] = res5
        return out, used
    ri = {x["item_id"]: x for x in rnd["items"]}
    items = [(x.get("family_id") or family_of(x["item_id"]), x["repair"] - ri[x["item_id"]]["repair"])
             for x in top["items"] if x["item_id"] in ri and x["repair"] is not None
             and ri[x["item_id"]]["repair"] is not None]
    est, (lo, hi), n, nf = item_mean_test(items, rng)
    controls_ok = of is not None and nu is not None and of <= .10 and nu <= .10
    res5.update(top_minus_random=est, top_minus_random_ci=[lo, hi], n_items=n, n_families=nf,
                n_random_draws=rnd.get("n_draws"))
    if stimuli in XL:
        # C5 as for BX, plus C5b (FREEZE_XL.md): the frozen heads must repair more with the item's own working-run
        # content than with another item's working-run content (paired by item, family bootstrap). The other
        # controls (other_failing, random_nudge, mean_failing) are reported, not scored: test0 showed in phi-4 that
        # removing the heads' own output repairs on its own, so ``other_failing <= .10`` does not test content.
        ok = lo > 0 and (top.get("repair") or 0) >= .25
        ow = tr.get("other_working", {})
        res5.update(mean_failing=tr.get("mean_failing", {}).get("repair"), other_working=ow.get("repair"))
        if "items" in ow:
            oi = {x["item_id"]: x for x in ow["items"]}
            pairs = [(x.get("family_id") or family_of(x["item_id"]), x["repair"] - oi[x["item_id"]]["repair"])
                     for x in top["items"] if x["item_id"] in oi and x["repair"] is not None
                     and oi[x["item_id"]]["repair"] is not None]
            e_b, (lo_b, hi_b), _, _ = item_mean_test(pairs, rng)
            out["C5b"] = dict(top_minus_other_working=e_b, ci=[lo_b, hi_b], verdict=verdict(lo_b > 0))
        else:
            out["C5b"] = dict(verdict=verdict(None, "other_working control missing (--controls --controls-extra)"))
    elif stimuli:
        ok = lo > 0 and (top.get("repair") or 0) >= .25
    else:
        ok = lo > 0 and (top.get("repair") or 0) >= .25 and controls_ok
        if of is None or nu is None:
            res5["verdict"] = verdict(None, "controls missing")
            out["C5"] = res5
            return out, used
    res5["verdict"] = verdict(ok)
    res5["file"] = os.path.basename(path5)
    out["C5"] = res5
    return out, used


def c6_c8(res, m, fz, split, rng):
    path = find(res, f"otr14_{fz['slug']}_{split}_*.json")
    if not path:
        nv = dict(verdict=verdict(None, "file missing"))
        return {"C6": nv, "C8": nv}, []
    D = load(path)
    L = fz["by"]
    if L not in D["layers"]:
        nv = dict(verdict=verdict(None, f"layer {L} not in run"))
        return {"C6": nv, "C8": nv}, [path]
    li = D["layers"].index(L)
    fam = {it["item_id"]: it.get("family_id") or family_of(it["item_id"]) for it in D["items"]}
    out = {}
    for cid, key in (("C6", "by_from_failing"), ("C8", "by_from_patient")):
        items = [(fam.get(i, family_of(i)), c[key][li]) for i, c in D["curves"].items()
                 if key in c and c[key][li] is not None and not (isinstance(c[key][li], float) and math.isnan(c[key][li]))]
        if not items:
            out[cid] = dict(verdict=verdict(None, "no items"))
            continue
        est, (lo, hi), n, nf = item_mean_test(items, rng)
        out[cid] = dict(layer=L, damage=est, ci=[lo, hi], n_items=n, n_families=nf, verdict=verdict(lo > 0))
    return out, [path]


def c7(res, m, fz, split, rng):
    if not fz["c7"]:
        return dict(verdict="no claim"), []

    def keep(p):
        D = load(p)
        return (D.get("by_prefix") or "").startswith("Someone was")
    path = find(res, f"otr07_{fz['slug']}_{split}*.json", exclude=(), keep=keep)
    if not path:
        return dict(verdict=verdict(None, "file missing")), []
    D = load(path)
    conds = D.get("conditions") or []
    fail_ix = next((i for i, c in enumerate(conds) if "someone" in json.dumps(c).lower() and "done by" not in json.dumps(c).lower()), 0)
    recs = [r for r in D["records"] if r["condition"] == fail_ix and r.get("correct") is False]
    if len(recs) < 20:
        return dict(n_wrong=len(recs), verdict=verdict(None, f"{len(recs)} wrong items < 20")), [path]
    items = [(r.get("family_id") or family_of(r["item_id"]), float(r["ld_model_by"] > 0)) for r in recs]
    est, (lo, hi), n, nf = item_mean_test(items, rng)
    return dict(share_agent_after_prefix=est, ci=[lo, hi], n_wrong=n, n_families=nf,
                verdict=verdict(est > .5 and lo > .5)), [path]


# ----------------------------------------------------------------------------- main
def score(res, split, models=None, stimuli=None, seed=SEED):
    table, inputs = {}, {}
    for m, fz in ALL_FROZEN.items():
        if (models and m not in models) or (not models and m not in FROZEN):
            continue
        rng = np.random.default_rng(seed)
        row, used = {}, []
        if stimuli:
            if stimuli in XL:
                row["C1"], u = c1_xl(res, m, fz, split, rng, stimuli); used += u
            out, u = c2_c3(res, m, fz, split, rng, stimuli); row.update(out); used += u
            out, u = c4_c5(res, m, fz, split, rng, stimuli); used += u
            row.update({k: v for k, v in out.items() if k in (("C4", "C5", "C5b") if stimuli in XL else ("C5",))})
        else:
            row["C1"], u = c1(res, m, fz, split, rng); used += u
            out, u = c2_c3(res, m, fz, split, rng); row.update(out); used += u
            out, u = c4_c5(res, m, fz, split, rng); row.update(out); used += u
            out, u = c6_c8(res, m, fz, split, rng); row.update(out); used += u
            row["C7"], u = c7(res, m, fz, split, rng); used += u
        table[m] = row
        inputs[m] = {os.path.basename(p): sha256(p) for p in used}
    return table, inputs


def markdown(table):
    claims = sorted({c for r in table.values() for c in r})
    models = list(table)
    short = {m: m.split("/")[1].split("-")[0] if "/" in m else m for m in models}
    lines = ["| Claim | " + " | ".join(short[m] for m in models) + " |", "|---" * (len(models) + 1) + "|"]
    for c in claims:
        cells = []
        for m in models:
            r = table[m].get(c, {})
            v = r.get("verdict", "")
            est = next((r[k] for k in ("estimate", "role", "c2_minus_c3", "first_crossing", "top_minus_random",
                                         "damage", "share_agent_after_prefix", "top_minus_other_working") if r.get(k) is not None), None)
            cells.append(v if est is None else f"{v} ({est:.2f})" if isinstance(est, float) else f"{v} ({est})")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="results")
    p.add_argument("--split", default="test0", choices=("dev", "test0", "test1", "test2", "test3"))
    p.add_argument("--stimuli", default=None, help="replication set: bx-transitive (C2, C3, C5) or es / gl (C2, C3, C4, C5, C5b; FREEZE_XL.md)")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--set", default="core", choices=("core", "70b"),
                   help="core: the five models of FREEZE.md (default); 70b: the two models of FREEZE_70B.md")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    models = args.models or (list(FROZEN_70B) if args.set == "70b" else None)
    table, inputs = score(args.results, args.split, models, args.stimuli)
    md = markdown(table)
    print(md)
    report = dict(scorer_version=SCORER_VERSION, split=args.split, stimuli=args.stimuli, n_boot=N_BOOT, seed=SEED,
                  frozen={m: ALL_FROZEN[m] for m in table}, table=table, inputs_sha256=inputs, markdown=md)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"\nwrote {args.out}", file=sys.stderr)
    return report


if __name__ == "__main__":
    main()
