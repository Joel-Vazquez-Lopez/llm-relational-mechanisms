"""ML-091: factorial event-role query binding, exploratory discovery only.

Run in repository root, with ML-088 dependencies/checkpoints installed:
    python3 scripts/ml091_counterfactual_binding.py --phase preflight
    python3 scripts/ml091_counterfactual_binding.py --phase plan
    # Review + commit this exact script and reports/ml091/frozen_discovery_plan.json
    python3 scripts/ml091_counterfactual_binding.py --phase run

The same sentence receives four questions: first/second x agent/patient.
The second clause is shown in matched active and passive voices with identical
participants, verbs and event/role labels. The first clause is held fixed.

Starts after index 8 in the same original first-passive calibration-template
sorting as ML-089 [0:4] and ML-090 [4:8]; skips any sentences
that overlap earlier experiments. Does not read, reconstruct, or
reevaluate previously protected ML-088 cases. This is NOT an independent cohort,
a confirmatory test, or a mechanistic circuit intervention.
"""

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean

import torch

import ml080_learning_curve as ml
import ml088_probe_access as m88

SEEDS = (701, 709, 719, 727, 733, 739, 743, 751)
ARMS = ("familiar", "target")
VOICES = ("active", "passive")
QUESTION_NAMES = ("first_agent", "first_patient", "second_agent", "second_patient")
N_PER_PAIR = 4
START_OFFSET = 8  # ML-089: [0:4]; ML-090: [4:8]
BATCH = 64
ROOT = Path("results/ml088_fresh_models")
PLAN = Path("reports/ml091/frozen_discovery_plan.json")
REPORT = Path("reports/ml091/discovery_evaluation.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sentence(c):
    tokens = ml.prompt(c)
    return tuple(tokens[:tokens.index("<sep>")])


def swap_queried_event(c):
    """Change ONLY the queried event metadata, not the realized sentence."""
    d = replace(
        c, target_pair=c.distractor_pair, distractor_pair=c.target_pair,
        target_verb=c.distractor_verb, distractor_verb=c.target_verb,
        target_dir=c.distractor_dir, distractor_dir=c.target_dir,
        target_voice=c.distractor_voice, distractor_voice=c.target_voice,
        target_pos=0 if c.target_pos == 1 else 1,
    )
    assert sentence(c) == sentence(d)
    return d


def four_questions(second_event):
    assert second_event.target_pos == 1
    q2a = replace(second_event, role="agent")
    q2p = replace(second_event, role="patient")
    q1a = replace(swap_queried_event(q2a), role="agent")
    q1p = replace(q1a, role="patient")
    rows = [q1a, q1p, q2a, q2p]
    assert len({sentence(c) for c in rows}) == 1
    assert len(set(ml.answer(c) for c in rows)) == 4
    assert tuple(c.target_pos for c in rows) == (0, 0, 1, 1)
    assert tuple(c.role for c in rows) == ("agent", "patient", "agent", "patient")
    toks = [ml.prompt(c) for c in rows]
    assert len({len(t) for t in toks}) == 1
    n = len(toks[0])
    # The question event selector and role selector must be the ONLY differences.
    diff_event = [i for i in range(n) if toks[0][i] != toks[2][i]]
    diff_role = [i for i in range(n) if toks[0][i] != toks[1][i]]
    assert len(diff_event) == len(diff_role) == 1
    assert toks[0][diff_event[0]] == "first" and toks[2][diff_event[0]] == "second"
    assert toks[0][diff_role[0]] == "agent" and toks[1][diff_role[0]] == "patient"
    return rows


def prepare():
    # IMPORTANT: m88.prepare mutates ML-084's global generator settings.
    # Call exactly once per process; do not call m90.sample_cases afterwards.
    vocab, train, cal_rounds, held, base, calibration, dose, fit, valid = m88.prepare()
    old_hash = m88.frozen(m88.specification(vocab, train, cal_rounds, held, fit, valid), False)
    assert tuple(ml.SEEDS) == SEEDS and tuple(ml.ARMS) == ARMS
    train_pairs = {p for matching in train for p in matching}
    cal_pairs = sorted({p for matching in cal_rounds for p in matching})
    held_pairs = {p for matching in held for p in matching}
    assert len(train_pairs) == 72 and len(cal_pairs) == len(held_pairs) == 24
    assert not (set(cal_pairs) & train_pairs or set(cal_pairs) & held_pairs)

    # Use exactly the ML-089/ML-090 original template ordering.
    options = defaultdict(list)
    for c in calibration:
        if (c.target_pos == 0 and c.target_voice == "passive"
                and c.role == "agent" and c.target_verb in ml.VERBS[2:]):
            options[c.target_pair].append(c)

    # Prior ML-089 and ML-090 displayed variants at BOTH event positions and
    # BOTH target voices. Exclude every such realized sentence, not merely
    # the particular original first-passive anchor prompt.
    prior = []
    sorted_by_pair = {}
    for pair in cal_pairs:
        entries = sorted(options[pair], key=lambda c: digest(ml.prompt(c)))
        assert len(entries) >= START_OFFSET + N_PER_PAIR, (pair, len(entries))
        sorted_by_pair[pair] = entries
        prior.extend(entries[:START_OFFSET])
    assert len(prior) == 192
    prior_sentences = {
        sentence(replace(c, target_pos=position, target_voice=voice))
        for c in prior for position in (0, 1) for voice in VOICES
    }

    selected = []
    chosen_sentences = set()
    for pair in cal_pairs:
        taken = 0
        for candidate in sorted_by_pair[pair][START_OFFSET:]:
            variants = {
                sentence(replace(candidate, target_pos=1, target_voice=voice))
                for voice in VOICES
            }
            assert len(variants) == 2
            if variants & (prior_sentences | chosen_sentences):
                continue
            selected.append(candidate)
            chosen_sentences.update(variants)
            taken += 1
            if taken == N_PER_PAIR:
                break
        assert taken == N_PER_PAIR, f"Insufficient disjoint templates for pair {pair}"
    assert len(selected) == 96

    def expand(anchors):
        groups = {voice: [] for voice in VOICES}
        for anchor in anchors:
            for voice in VOICES:
                c = replace(anchor, target_pos=1, target_voice=voice)
                assert c.target_pos == 1 and c.target_voice == voice
                groups[voice].append(four_questions(c))
        return groups

    groups = expand(selected)
    new_sentences = {sentence(q[0]) for voice in VOICES for q in groups[voice]}
    assert len(new_sentences) == 192, "New sentence/voice configurations must be unique"
    assert not new_sentences & prior_sentences, "Overlap with prior exploratory sentences"
    all_new_prompts = [tuple(ml.prompt(c)) for voice in VOICES
                       for group in groups[voice] for c in group]
    assert len(all_new_prompts) == len(set(all_new_prompts)) == 768

    # Active vs passive differ ONLY in the second clause; the first source
    # clause stays identical and all four gold labels are preserved.
    for active, passive in zip(groups["active"], groups["passive"]):
        assert tuple(map(ml.answer, active)) == tuple(map(ml.answer, passive))
        assert sentence(active[0]) != sentence(passive[0])
        x, y = sentence(active[0]), sentence(passive[0])
        # Extract the first clause from the frame-specific separators.
        delimiter = "while" if "while" in x else ","
        assert delimiter in x and delimiter in y
        assert x[:x.index(delimiter)] == y[:y.index(delimiter)]
        assert {ml.answer(c) for c in active} == {ml.answer(c) for c in passive}
        # The distractor/first clause has unchanged voice, direction and names.
        assert active[0].target_pair == passive[0].target_pair
        assert active[0].target_voice == passive[0].target_voice

    return old_hash, groups


def specification(old_hash, groups):
    return {
        "experiment": "ML-091",
        "status": "EXPLORATORY_CALIBRATION_FACTORIAL_PREDECLARED",
        "script_sha256": sha(__file__),
        "ml080_source_sha256": sha(ml.__file__),
        "ml088_source_sha256": sha(m88.__file__),
        "ml088_protocol_sha256": old_hash,
        "model_cohort": "Existing 16 ML-088 step-12800 frozen checkpoints; eight paired seeds",
        "seeds": SEEDS,
        "arms": ARMS,
        "case_selection": "ML-089/090 original source selection filter and SHA256-JSON prompt order; after indices [0:8], greedily take first four anchors per each of 24 calibration pairs whose second-active/second-passive realized sentences are unused and do not match ANY of the prior 0:8 anchors in all four first/second x active/passive realizations",
        "prior_exploratory_offsets": "ML-089 [0:4], ML-090 [4:8]",
        "n_source_templates": 96,
        "sentence_variants": 2,
        "questions_per_sentence": 4,
        "n_questions_per_model": 768,
        "query_order": QUESTION_NAMES,
        "voice_variants": VOICES,
        "controls": "Same sentence for four queries; first clause held fixed when second clause changes voice; same participants, verbs, agent/patient semantic labels; 2x2 event/role query matrix",
        "prompt_hashes": {voice: digest([ml.prompt(c) for group in groups[voice] for c in group]) for voice in VOICES},
        "sentences_sha256": {voice: digest([sentence(g[0]) for g in groups[voice]]) for voice in VOICES},
        "primary_diagnostic": "Event-conditioned role-gain interaction for SECOND event participants: role_gain(second-event names | asked second) MINUS role_gain(second-event names | asked first). Paired target-minus-familiar exposure contrast, second-passive MINUS second-active (difference-in-differences)",
        "role_gain": "0.5 * [(logit(agent_e)-logit(patient_e)) when asked agent - same difference when asked patient], evaluated for a specified event e",
        "second_event_role_gating": "role_gain(second-event names | queried second) - role_gain(second-event names | queried first); this measures the effect of switching the EVENT question on how the ROLE question changes second-event name logits",
        "alternative_explanations": [
            "Wrong-event selection with intact role choice in wrong event",
            "Incorrect role selection even when event pair is oracle-restricted",
            "Role cue broadcasts to both events regardless of event instruction",
            "Both event and role cues influence output but do not combine selectively",
            "Simple logit baseline/scale differences or voice-position confounding",
        ],
        "secondary_diagnostics": [
            "Second-event and first-event role gain separately",
            "Off-event role gain, first-event role gating, within-query role-pair selectivity",
            "Event-cue selectivity holding role fixed",
            "Four-choice and full-vocabulary accuracies, question-specific accuracies",
            "Exact event/role wrong-answer decomposition, conditional wrong-event role compliance",
            "Four-query all-correct, query flip rates, 4x4 query-to-response matrices",
            "Mean correct-versus-best-other margin and correct-pair two-choice accuracy",
            "Unnormalized logit contrasts (no claim of invariance to logit scale)",
        ],
        "analysis": "Report eight paired seed effects, mean, seeded bootstrap 95% CI and positive count; no post hoc model or seed exclusion, no tuning, no formal confirmatory pass/fail",
        "scope": "Existing model cohort and calibration pairs; discovery-only even if prompts are new. A positive contrast is a logit-behavior interaction, not a proof of a neural binding circuit.",
        "protected_safety": "No protected cases are read/generated; no ML-088 re-open; refuse repeated ML-091 discovery after saved report exists",
    }


def freeze(spec, create=False):
    document = json.loads(json.dumps({**spec, "sha256": digest(spec)}))
    if create and not PLAN.exists():
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(json.dumps(document, indent=2) + "\n")
    if not PLAN.exists():
        raise FileNotFoundError("Run --phase plan after preflight")
    assert json.loads(PLAN.read_text()) == document, "Frozen ML-091 design or dependencies changed"
    return document["sha256"]


def name_ids(group):
    q1a, q1p, q2a, q2p = group
    return [ml.VOCAB[ml.answer(q)] for q in (q1a, q1p, q2a, q2p)]


def analyze_logits(logits, groups):
    """Input logits [T*4,V] in four-query order; returns source-level metrics."""
    count = len(groups)
    assert logits.ndim == 2 and logits.shape[0] == count * 4
    z = logits.reshape(count, 4, -1).float()
    semantic = torch.tensor([name_ids(group) for group in groups], dtype=torch.long)
    assert semantic.shape == (count, 4)
    assert bool(torch.tensor([len(set(row)) == 4 for row in semantic.tolist()]).all())
    query = torch.arange(4).view(1, 4).expand(count, 4)
    available = z.gather(2, semantic[:, None, :].expand(count, 4, 4))
    predicted = available.argmax(2)
    actual = query
    native_correct = predicted == actual
    full_pred = z.argmax(2)
    correct_ids = semantic.gather(1, query)
    full_correct = full_pred == correct_ids
    event_wrong = predicted // 2 != actual // 2
    role_wrong = (predicted % 2 != actual % 2) & ~event_wrong
    assert bool((native_correct | event_wrong | role_wrong).all())
    assert not bool((native_correct & (event_wrong | role_wrong)).any())

    # Differences in name logits as the query changes; these do NOT require
    # the correct option to be the actual top prediction.
    first_role_diff = available[:, :, 0] - available[:, :, 1]
    second_role_diff = available[:, :, 2] - available[:, :, 3]
    first_own = (first_role_diff[:, 0] - first_role_diff[:, 1]) * 0.5
    first_off = (second_role_diff[:, 0] - second_role_diff[:, 1]) * 0.5
    second_own = (second_role_diff[:, 2] - second_role_diff[:, 3]) * 0.5
    second_off = (first_role_diff[:, 2] - first_role_diff[:, 3]) * 0.5
    # Within-query relative role response between the two name pairs.
    first_specificity = first_own - first_off
    second_specificity = second_own - second_off
    # True 2x2 query-factor interaction for each event-pair logit contrast:
    # does switching FIRST->SECOND change how the AGENT/PATIENT instruction
    # affects the SECOND event names? A broadcast-only role cue gives zero.
    first_role_gating = first_own - second_off
    second_role_gating = second_own - first_off
    # Event cue effect, holding grammatical role fixed.
    event_agent = 0.5 * ((available[:, 0, 0]-available[:, 0, 2]) -
                         (available[:, 2, 0]-available[:, 2, 2]))
    event_patient = 0.5 * ((available[:, 1, 1]-available[:, 1, 3]) -
                           (available[:, 3, 1]-available[:, 3, 3]))
    event_selectivity = 0.5 * (event_agent + event_patient)

    # Four-choice max-other margins, with source identity order fixed.
    gold_logits = available.gather(2, actual[:, :, None]).squeeze(2)
    rival = available.masked_fill(torch.nn.functional.one_hot(actual, 4).bool(),
                                  float("-inf")).max(2).values
    # Oracle two-choice correct-pair accuracy: deliberately labeled oracle.
    pair_scores = torch.where((actual // 2)[:, :, None] == 0,
                              available[:, :, :2], available[:, :, 2:])
    # torch.where above requires shape [count,4,2] and cannot broadcast a
    # [count,4,1] selector over [count,4,2] accidentally if shapes change.
    within_pair = pair_scores.argmax(2) == (actual % 2)
    confusion = [[int(((actual == i) & (predicted == j)).sum())
                  for j in range(4)] for i in range(4)]
    wrong_event_count = int(event_wrong.sum())
    wrong_event_role_correct = ((predicted % 2 == actual % 2) & event_wrong)
    return {
        "n_source_templates": count,
        "n_questions": count * 4,
        "four_choice_accuracy": float(native_correct.float().mean()),
        "full_vocab_accuracy": float(full_correct.float().mean()),
        "per_question_four_choice_accuracy": {QUESTION_NAMES[i]: float(native_correct[:, i].float().mean()) for i in range(4)},
        "wrong_event_rate": float(event_wrong.float().mean()),
        "correct_event_wrong_role_rate": float(role_wrong.float().mean()),
        "wrong_event_n": wrong_event_count,
        "wrong_event_role_correct_rate": (float(wrong_event_role_correct.sum()) / wrong_event_count if wrong_event_count else None),
        "oracle_within_queried_pair_twochoice_accuracy": float(within_pair.float().mean()),
        "all_four_queries_correct_rate": float(native_correct.all(1).float().mean()),
        "role_flip_rate_first_event_query": float((predicted[:, 0] != predicted[:, 1]).float().mean()),
        "role_flip_rate_second_event_query": float((predicted[:, 2] != predicted[:, 3]).float().mean()),
        "event_flip_rate_agent_query": float((predicted[:, 0] != predicted[:, 2]).float().mean()),
        "event_flip_rate_patient_query": float((predicted[:, 1] != predicted[:, 3]).float().mean()),
        "mean_gold_vs_best_other_margin": float((gold_logits-rival).mean()),
        "first_role_gain": float(first_own.mean()),
        "first_off_event_role_gain": float(first_off.mean()),
        "first_within_query_pair_selectivity": float(first_specificity.mean()),
        "first_event_role_gating": float(first_role_gating.mean()),
        "second_role_gain": float(second_own.mean()),
        "second_off_event_role_gain": float(second_off.mean()),
        "second_within_query_pair_selectivity": float(second_specificity.mean()),
        "second_event_role_gating": float(second_role_gating.mean()),
        "event_cue_selectivity": float(event_selectivity.mean()),
        "confusion_query_by_prediction_4x4": confusion,
    }


def bootstrap(xs, n=20000, seed=91091):
    rng = random.Random(seed)
    count = len(xs)
    dist = sorted(mean(xs[rng.randrange(count)] for _ in range(count)) for _ in range(n))
    return [dist[int(n * .025)], dist[int(n * .975) - 1]]


@torch.inference_mode()
def preflight(groups):
    # No trained models are loaded. Test counterfactual semantics and metric
    # calculations with synthetic logits before opening any discovery outcomes.
    assert len(groups["active"]) == len(groups["passive"]) == 96
    sample = groups["passive"][:8]
    flat = [c for group in sample for c in group]
    ids, readout, _, _ = ml.encode(flat)
    torch.manual_seed(91091)
    model = ml.Model().eval()
    direct = model(ids, readout).float()
    traced, _ = m88.m81.forward_trace(model, ids, readout)
    parity = float((direct - traced).abs().max())
    assert parity < 1e-4, parity
    v = len(ml.VOCAB)
    perfect = torch.full((len(flat), v), -9.0)
    for i, c in enumerate(flat):
        perfect[i, ml.VOCAB[ml.answer(c)]] = 9.0
    check = analyze_logits(perfect, sample)
    assert check["four_choice_accuracy"] == 1.0
    assert check["all_four_queries_correct_rate"] == 1.0
    assert check["wrong_event_rate"] == 0.0
    assert check["second_event_role_gating"] > 0
    assert all(check["confusion_query_by_prediction_4x4"][q][q] == len(sample) for q in range(4))
    role_blind = torch.full((len(flat), v), -9.0)
    for i, group in enumerate(sample):
        for qi in range(4):
            picked = group[0] if qi < 2 else group[2]  # fixed 'agent' response
            role_blind[4*i+qi, ml.VOCAB[ml.answer(picked)]] = 9.0
    blind = analyze_logits(role_blind, sample)
    assert abs(blind["four_choice_accuracy"] - 0.5) < 1e-8
    assert abs(blind["second_role_gain"]) < 1e-8
    assert abs(blind["first_role_gain"]) < 1e-8
    assert blind["role_flip_rate_first_event_query"] == 0
    assert blind["role_flip_rate_second_event_query"] == 0
    wrong = torch.full((len(flat), v), -9.0)
    for i, group in enumerate(sample):
        for qi in range(4):
            # Same grammatical role but always from the opposite event.
            other_q = group[qi + 2 if qi < 2 else qi - 2]
            wrong[4*i+qi, ml.VOCAB[ml.answer(other_q)]] = 9.0
    misplaced = analyze_logits(wrong, sample)
    assert misplaced["wrong_event_rate"] == 1.0
    assert misplaced["wrong_event_role_correct_rate"] == 1.0
    assert misplaced["four_choice_accuracy"] == 0.0
    # A broadcast role signal can distinguish agent/patient in BOTH events,
    # but has zero event x role interaction by construction. This is the
    # primary construct-validity test: a role-sensitive model must NOT
    # automatically count as an event-conditioned role-binding model.
    broadcast = torch.zeros((len(flat), v))
    for i, group in enumerate(sample):
        names = name_ids(group)
        for qi in range(4):
            for event_idx in (0, 1):
                broadcast[4*i+qi, names[2*event_idx+(qi % 2)]] = 5.0
    broadcast_score = analyze_logits(broadcast, sample)
    assert broadcast_score["second_role_gain"] > 0
    assert broadcast_score["first_role_gain"] > 0
    assert abs(broadcast_score["second_event_role_gating"]) < 1e-8
    assert abs(broadcast_score["first_event_role_gating"]) < 1e-8
    print(f"PREFLIGHT PASS: {len(groups['active'])} sentence templates, 2 second-clause voices, "
          f"4 questions each = {sum(len(g)*4 for g in groups.values())} questions per model")
    print(f"Model forward-trace parity={parity:.2e}; perfect, role-blind, wrong-event and broadcast-role scoring tests PASS")
    print("No trained checkpoints, previously protected cases, or discovery outcomes accessed.")


@torch.inference_mode()
def run(old_hash, plan_hash, groups, device):
    if REPORT.exists():
        raise FileExistsError(f"ML-091 already evaluated: {REPORT}; refuse to rerun")
    passed, counts = m88.model_gate(old_hash)
    if not passed:
        raise PermissionError(f"ML-088 existing-model competence gate failed: {counts}")
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            model = ml.load_checkpoint(
                ROOT / f"seed_{seed}" / arm / "model_step_12800.pt",
                seed, arm, old_hash, device)
            for voice in VOICES:
                cases = [c for group in groups[voice] for c in group]
                native = []
                for st in range(0, len(cases), BATCH):
                    ids, readout, _, _ = ml.encode(cases[st:st + BATCH])
                    out = model(ids.to(device), readout.to(device)).float().cpu()
                    native.append(out)
                scores = analyze_logits(torch.cat(native, dim=0), groups[voice])
                rows.append({"seed": seed, "arm": arm, "voice": voice, "metrics": scores})
                print(f"seed={seed} arm={arm:8s} second_voice={voice:7s} "
                      f"accuracy4={scores['four_choice_accuracy']:.3f} "
                      f"second_role_gain={scores['second_role_gain']:+.3f} "
                      f"off_role_gain={scores['second_off_event_role_gain']:+.3f} "
                      f"second_role_gating={scores['second_event_role_gating']:+.3f}",
                      flush=True)

    lookup = {(r["seed"], r["arm"], r["voice"]): r["metrics"] for r in rows}
    assert len(lookup) == len(SEEDS) * len(ARMS) * len(VOICES)
    def metric(s, a, v, k):
        return lookup[s, a, v][k]
    key = "second_event_role_gating"
    exposure_effects = {
        voice: [metric(seed,"target",voice,key)-metric(seed,"familiar",voice,key)
                for seed in SEEDS] for voice in VOICES
    }
    did = [b-a for b,a in zip(exposure_effects["passive"], exposure_effects["active"])]
    summary = {
        "predeclared_primary_diagnostic": key,
        "paired_exposure_difference_in_differences_by_seed": dict(zip(map(str, SEEDS), did)),
        "mean_exposure_difference_in_differences": mean(did),
        "seed_bootstrap_95pct_ci": bootstrap(did),
        "positive_seed_pairs": sum(z > 0 for z in did),
        "mean_paired_exposure_effect_active": mean(exposure_effects["active"]),
        "mean_paired_exposure_effect_passive": mean(exposure_effects["passive"]),
        "cells": {
            f"{arm}/{voice}": {
                "four_choice_accuracy": mean(metric(s,arm,voice,"four_choice_accuracy") for s in SEEDS),
                "full_vocab_accuracy": mean(metric(s,arm,voice,"full_vocab_accuracy") for s in SEEDS),
                "all_four_queries_correct_rate": mean(metric(s,arm,voice,"all_four_queries_correct_rate") for s in SEEDS),
                "first_role_gain": mean(metric(s,arm,voice,"first_role_gain") for s in SEEDS),
                "second_role_gain": mean(metric(s,arm,voice,"second_role_gain") for s in SEEDS),
                "second_off_event_role_gain": mean(metric(s,arm,voice,"second_off_event_role_gain") for s in SEEDS),
                "second_event_role_gating": mean(metric(s,arm,voice,key) for s in SEEDS),
                "second_within_query_pair_selectivity": mean(metric(s,arm,voice,"second_within_query_pair_selectivity") for s in SEEDS),
                "first_event_role_gating": mean(metric(s,arm,voice,"first_event_role_gating") for s in SEEDS),
                "event_cue_selectivity": mean(metric(s,arm,voice,"event_cue_selectivity") for s in SEEDS),
                "wrong_event_rate": mean(metric(s,arm,voice,"wrong_event_rate") for s in SEEDS),
                "correct_event_wrong_role_rate": mean(metric(s,arm,voice,"correct_event_wrong_role_rate") for s in SEEDS),
            } for arm in ARMS for voice in VOICES
        },
        "interpretation_warning": (
            "All results remain exploratory with previously examined model cohort and calibration pairs. "
            "A logit-level 2x2 interaction is not proof of a native binding circuit or a necessary interaction for correct task performance (factorized event + role computations may solve it). "
            "An oracle supplies the set of four answer options for analysis only; native output is also scored. "
            "Voice changes source-token order/length, so the active-passive contrast is not purely one feature."
        ),
    }
    doc = {
        "experiment":"ML-091", "status":"EXPLORATORY_DISCOVERY",
        "plan_sha256":plan_hash, "ml088_protocol_sha256":old_hash,
        "protected_examples_generated":False, "n_model_checkpoints":16,
        "rows":rows, "summary":summary,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(doc, indent=2)+"\n")
    print("\nML-091 EXPLORATORY SUMMARY\n"+json.dumps(summary, indent=2))
    print("Saved", REPORT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "plan", "run"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    old_hash, groups = prepare()
    if args.phase == "preflight":
        preflight(groups)
    elif args.phase == "plan":
        h = freeze(specification(old_hash, groups), create=True)
        print(f"ML-091 discovery plan frozen: {PLAN} hash={h}")
        print("No trained-model inference or protected examples accessed.")
    else:
        h = freeze(specification(old_hash, groups), create=False)
        run(old_hash, h, groups, ml.select_device(args.device))


if __name__ == "__main__":
    main()
