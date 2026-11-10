"""ML-092: matched-source-state transfer on second-passive event–role binding.

Exploratory discovery using existing frozen ML-088 checkpoints. NEVER evaluates
or reconstructs ML-088 protected examples. Does not train or alter any model.

Run from the repository root:
    python3 scripts/ml092_source_state_transfer.py --phase preflight
    python3 scripts/ml092_source_state_transfer.py --phase plan
    # REVIEW + COMMIT exact script and frozen plan before viewing trained results.
    python3 scripts/ml092_source_state_transfer.py --phase run

Aim: isolate source-side representational sufficiency using within-model
activation transplants at PRE-QUESTION participant positions. Every sentence
is asked first/second x agent/patient questions, so transplanted source states
have never seen which event or role the question requests (causal attention).

Main donor: second-active sentence, same participants, roles, and verb.
Control donor: second-active sentence with agent/patient roles reversed,
                  matched names/voice/positions/patch magnitude family.
Sanity donors: passive self, identical first-event states, final-layer source
              patches (which cannot affect answer). Single-name transfers
              test whether a specific source token carries the effect.

IMPORTANT: The donor may carry semantic ANSWER-RELEVANT information, just not a
question-specific answer. This is a sufficiency/transplant intervention, NOT
evidence of a unique native mechanism or clean separation of content and
positional encoding. All results are exploratory, same model cohort/task family.
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean

import torch

import ml080_learning_curve as ml
import ml088_probe_access as m88
import ml091_counterfactual_binding as m91

SEEDS = (701, 709, 719, 727, 733, 739, 743, 751)
ARMS = ("familiar", "target")
ROOT = Path("results/ml088_fresh_models")
PLAN = Path("reports/ml092/frozen_discovery_plan.json")
REPORT = Path("reports/ml092/discovery_evaluation.json")
N_PER_PAIR = 4
BATCH_TEMPLATES = 12
SITE_LAYERS = (0, 1)  # after L1/L2; L3 is a structural no-effect preflight control
MAIN_MODES = (
    "aligned_both", "reversed_both", "aligned_agent", "aligned_patient",
)
MODES = MAIN_MODES + ("self_both", "first_event_control")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def participants(c, event):
    if event == "second":
        pair, direction = c.target_pair, c.target_dir
        assert c.target_pos == 1
    elif event == "first":
        pair, direction = c.distractor_pair, c.distractor_dir
    else:
        raise ValueError(event)
    return pair if direction == 0 else pair[::-1]  # (agent, patient)


def source_token_pos(c, name):
    tokens = ml.prompt(c)
    end = tokens.index("<sep>")
    hits = [i for i, token in enumerate(tokens[:end]) if token == name]
    assert len(hits) == 1, (name, hits, tokens)
    return hits[0]


def select_cases():
    # Single generator initialization. DO NOT call m91.prepare after this.
    vocab, tr, ca, held, base, calibration, dose, fit, valid = m88.prepare()
    old_hash = m88.frozen(m88.specification(vocab, tr, ca, held, fit, valid), False)
    assert tuple(ml.SEEDS) == SEEDS and tuple(ml.ARMS) == ARMS
    train_pairs = {p for rd in tr for p in rd}
    cal_pairs = sorted({p for rd in ca for p in rd})
    held_pairs = {p for rd in held for p in rd}
    assert len(cal_pairs) == len(held_pairs) == 24
    assert not set(cal_pairs) & (train_pairs | held_pairs)

    options = defaultdict(list)
    for c in calibration:
        if (c.target_pos == 0 and c.target_voice == "passive"
                and c.role == "agent" and c.target_verb in ml.VERBS[2:]):
            options[c.target_pair].append(c)
    sorted_by_pair = {
        p: sorted(options[p], key=lambda c: m91.digest(ml.prompt(c)))
        for p in cal_pairs
    }

    # Exact reconstruction of ML-089/090 exclusions and ML-091 selected cases.
    prior = []
    for p in cal_pairs:
        assert len(sorted_by_pair[p]) >= 16
        prior.extend(sorted_by_pair[p][:8])
    prior_sentences = {
        m91.sentence(replace(c, target_pos=pos, target_voice=voice))
        for c in prior for pos in (0, 1) for voice in ("active", "passive")
    }
    used = set(prior_sentences)
    ml091_selected = []
    ml091_chosen_second = set()
    for p in cal_pairs:
        taken = 0
        for c in sorted_by_pair[p][8:]:
            variants = {m91.sentence(replace(c, target_pos=1, target_voice=voice))
                        for voice in ("active", "passive")}
            if variants & (prior_sentences | ml091_chosen_second):
                continue
            ml091_selected.append(c)
            ml091_chosen_second.update(variants)
            taken += 1
            if taken == 4:
                break
        assert taken == 4
    assert len(ml091_selected) == 96
    ml091_plan = json.loads(m91.PLAN.read_text())
    for voice in ("active", "passive"):
        realized = [m91.sentence(replace(c, target_pos=1, target_voice=voice))
                    for c in ml091_selected]
        assert m91.digest(realized) == ml091_plan["sentences_sha256"][voice], (
            "Failed to reconstruct ML-091's previously used source sentences", voice
        )
    for c in ml091_selected:
        for pos in (0, 1):
            for voice in ("active", "passive"):
                used.add(m91.sentence(replace(c, target_pos=pos, target_voice=voice)))

    # Select new sentence-realizations distinct from ALL previous studies.
    selected = []
    new_sentences = set()
    previous_set = set(ml091_selected)
    for p in cal_pairs:
        taken = 0
        for c in sorted_by_pair[p][8:]:
            if c in previous_set:
                continue
            all_voices_positions = {
                m91.sentence(replace(c, target_pos=pos, target_voice=voice))
                for pos in (0, 1) for voice in ("active", "passive")
            }
            if all_voices_positions & (used | new_sentences):
                continue
            selected.append(c)
            new_sentences.update(all_voices_positions)
            taken += 1
            if taken == N_PER_PAIR:
                break
        assert taken == N_PER_PAIR, (
            f"Insufficient unused calibration templates for {p}: found {taken}"
        )
    assert len(selected) == 24 * N_PER_PAIR == 96
    assert not used & new_sentences

    groups = {"active": [], "passive": []}
    for c in selected:
        for voice in groups:
            second = replace(c, target_pos=1, target_voice=voice)
            groups[voice].append(m91.four_questions(second))
    assert all(len(groups[v]) == 96 for v in groups)
    assert not {m91.sentence(g[0]) for v in groups for g in groups[v]} & used
    assert len({m91.sentence(g[0]) for v in groups for g in groups[v]}) == 192
    for active, passive in zip(groups["active"], groups["passive"]):
        assert [ml.answer(q) for q in active] == [ml.answer(q) for q in passive]
        assert m91.sentence(active[0]) != m91.sentence(passive[0])
        for g in (active, passive):
            assert len({m91.sentence(q) for q in g}) == 1
            assert [q.role for q in g] == ["agent", "patient", "agent", "patient"]
            assert len({ml.answer(q) for q in g}) == 4
    return old_hash, groups, ml091_plan["sha256"]


def donors(passive_groups):
    passive = [g[2] for g in passive_groups]  # second-agent representative
    aligned = [replace(c, target_voice="active") for c in passive]
    reversed_cases = [replace(c, target_voice="active", target_dir=1-c.target_dir)
                      for c in passive]
    for p, good, bad in zip(passive, aligned, reversed_cases):
        assert participants(p, "second") == participants(good, "second")
        assert participants(p, "second") == participants(bad, "second")[::-1]
        assert set(participants(good, "second")) == set(participants(bad, "second"))
        assert set(participants(p, "first")) == set(participants(good, "first"))
        assert m91.sentence(p) != m91.sentence(good)
        assert m91.sentence(good) != m91.sentence(bad)
        assert all(source_token_pos(c, name) < ml.prompt(c).index("<sep>")
                   for c in (p, good, bad) for name in c.target_pair)
    return {"aligned": aligned, "reversed": reversed_cases, "self": passive}


def specification(old_hash, groups, ml091_plan_hash):
    return {
        "experiment": "ML-092",
        "status": "EXPLORATORY_SOURCE_TRANSPLANT_FREEZE_BEFORE_TRAINED_INFERENCE",
        "script_sha256": sha(__file__),
        "ml080_sha256": sha(ml.__file__),
        "ml088_sha256": sha(m88.__file__),
        "ml091_sha256": sha(m91.__file__),
        "ml088_protocol_sha256": old_hash,
        "ml091_plan_sha256": ml091_plan_hash,
        "seeds": SEEDS,
        "arms": ARMS,
        "model_cohort": "16 existing frozen ML-088 checkpoints at step 12800",
        "selection": "Reconstruct prior ML-089/090/091 source sets on 24 calibration pairs; select next four unused anchors per pair, all first/second x active/passive realized sentences disjoint",
        "n_source_templates": len(groups["passive"]),
        "questions_per_sentence": 4,
        "query_order": m91.QUESTION_NAMES,
        "source_sentence_hashes": {v: digest([m91.sentence(g[0]) for g in groups[v]])
                                   for v in ("active", "passive")},
        "prompts_hashes": {v: digest([ml.prompt(q) for g in groups[v] for q in g])
                           for v in ("active", "passive")},
        "interventions": {
            "sites": "Post-block residual at source participant positions after L1 or L2; continue remaining layers without changes",
            "modes": MODES,
            "aligned_both": "Replace passive second-clause agent/patient states with active donor states matched by SAME NAME and SAME SEMANTIC ROLE",
            "reversed_both": "Identical active-donor construction, but agent/patient assignments reversed. Donor still has same names, aligned by NAME. Main semantic control.",
            "aligned_agent": "Replace second-clause agent state only from semantically matched active donor",
            "aligned_patient": "Replace second-clause patient state only from semantically matched active donor",
            "self_both": "Replace second-clause states by identical passive source states, strict no-op control",
            "first_event_control": "Copy first-clause participant source states from active donor to passive receiver; first clause is unchanged and causal, strict no-op",
            "L3_site": "After final layer, replacing source positions must not influence answer (preflight only)",
            "question_independence": "Source tokens precede question; causal mask prevents source state from accessing queried event or role cue",
        },
        "primary_exploratory_contrast": "For each familiar-arm seed, second-passive aligned_both(L1) minus reversed_both(L1) in second_event_role_gating; report also aligned_both minus native and matched-vs-reversed four-choice rescue. No formal pass/fail threshold.",
        "secondary": "Both arms, L1/L2 sites; agent-only/patient-only; effects on the two second-event questions, all-four accuracy, wrong event/role, full-vocab output, mean source-patch norm and logit change; native second-active control; per-seed paired summaries",
        "limitations": "Even pre-question source states can transmit answer-relevant role semantics; transplant is sufficiency, not native necessity. Active/passive name positions, causal histories and token lengths differ. Controls match source identity but not exact perturbation direction or scale. This uses previously studied models and calibration pairs, not independent confirmation.",
        "protected_safety": "Never access/open ML-088 protected results or generate protected cases; refuse repeated ML-092 report",
    }


def freeze(spec, create):
    doc = json.loads(json.dumps({**spec, "sha256": digest(spec)}))
    if create and not PLAN.exists():
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(json.dumps(doc, indent=2) + "\n")
    if not PLAN.exists():
        raise FileNotFoundError("Run --phase plan before --phase run")
    assert json.loads(PLAN.read_text()) == doc, "Frozen ML-092 plan/code/dependencies differ"
    return doc["sha256"]


@torch.inference_mode()
def trace(model, cases, device):
    ids, ro, _, _ = ml.encode(cases)
    ids = ids.to(device)
    b, n = ids.shape
    x = model.embed(ids) + model.pos(torch.arange(n, device=device))
    mask = torch.ones(n, n, device=device, dtype=torch.bool).triu(1)
    residuals = []
    for block in model.blocks:
        x = block(x, mask)
        residuals.append(x)
    return residuals


def indexes(passive_groups, donor_cases, mode, device):
    recv = [g[2] for g in passive_groups]
    assert len(recv) == len(donor_cases)
    src_positions, dst_positions = [], []
    for c, d in zip(recv, donor_cases):
        second_names = participants(c, "second")
        if mode == "aligned_agent":
            names = [second_names[0]]
        elif mode == "aligned_patient":
            names = [second_names[1]]
        elif mode == "first_event_control":
            names = list(participants(c, "first"))
        else:
            names = list(second_names)
        assert len(names) in (1, 2)
        src_positions.append([source_token_pos(d, n) for n in names])
        dst_positions.append([source_token_pos(c, n) for n in names])
    return (torch.tensor(src_positions, dtype=torch.long, device=device),
            torch.tensor(dst_positions, dtype=torch.long, device=device))


@torch.inference_mode()
def patched_logits(model, passive_groups, donor_cases, donor_trace, device,
                   layer, mode):
    assert 0 <= layer < len(model.blocks)
    assert len(passive_groups) == len(donor_cases)
    cases = [q for g in passive_groups for q in g]
    ids, readout, _, _ = ml.encode(cases)
    ids = ids.to(device)
    readout = readout.to(device)
    batch, n = ids.shape
    assert batch == 4 * len(passive_groups)
    ix = torch.arange(batch, device=device)
    owner = ix // 4
    src, dst = indexes(passive_groups, donor_cases, mode, device)
    positions = dst[owner]
    donor_positions = src[owner]
    donor_states = donor_trace[layer][owner]
    x = model.embed(ids) + model.pos(torch.arange(n, device=device))
    mask = torch.ones(n, n, device=device, dtype=torch.bool).triu(1)
    source_delta = None
    for j, block in enumerate(model.blocks):
        x = block(x, mask)
        if j == layer:
            values = donor_states[ix[:, None], donor_positions]
            original = x[ix[:, None], positions]
            source_delta = (values - original).norm(dim=2).mean(dim=1)
            x[ix[:, None], positions] = values
    logits = model.out(model.norm(x[ix, readout])).float().cpu()
    assert source_delta is not None
    return logits, source_delta.mean().item()


@torch.inference_mode()
def native_logits(model, groups, device):
    qs = [c for g in groups for c in g]
    ids, ro, _, _ = ml.encode(qs)
    out = model(ids.to(device), ro.to(device)).float().cpu()
    return out


def compact(metrics):
    return {k: v for k, v in metrics.items() if k != "confusion_query_by_prediction_4x4"}


def bootstrap(values, n=20000):
    rng = random.Random(92092)
    m = len(values)
    dist = sorted(mean(values[rng.randrange(m)] for _ in range(m)) for _ in range(n))
    return [dist[int(n * .025)], dist[int(n * .975) - 1]]


@torch.inference_mode()
def preflight(groups):
    torch.manual_seed(92092)
    model = ml.Model().eval()
    passive = groups["passive"][:4]
    active = groups["active"][:4]
    donor = donors(passive)
    native = native_logits(model, passive, "cpu")
    direct_active = native_logits(model, active, "cpu")
    assert native.shape[0] == direct_active.shape[0] == 16
    cache = {k: trace(model, v, "cpu") for k, v in donor.items()}
    self_first = trace(model, [g[0] for g in passive], "cpu")
    # Because source positions precede all question tokens, source states
    # from first-agent and second-agent questions must be identical.
    for rep, src in zip([g[2] for g in passive], [g[0] for g in passive]):
        assert m91.sentence(rep) == m91.sentence(src)
    for site in range(3):
        for g in range(len(passive)):
            for name in participants(passive[g][2], "second"):
                a = source_token_pos(passive[g][2], name)
                b = source_token_pos(passive[g][0], name)
                assert torch.allclose(cache["self"][site][g, a],
                                      self_first[site][g, b], atol=1e-6)
    for site in range(3):
        copied, delta = patched_logits(model, passive, donor["self"], cache["self"],
                                      "cpu", site, "self_both")
        assert (native - copied).abs().max().item() < 1e-4
        assert delta < 1e-5
        first, first_delta = patched_logits(model, passive, donor["aligned"],
                                           cache["aligned"], "cpu", site,
                                           "first_event_control")
        assert (native - first).abs().max().item() < 1e-4
        assert first_delta < 1e-5
        if site == 2:
            final, _ = patched_logits(model, passive, donor["aligned"],
                                     cache["aligned"], "cpu", site,
                                     "aligned_both")
            assert (native - final).abs().max().item() < 1e-4
    changed = []
    for site in SITE_LAYERS:
        for mode, donor_key in (("aligned_both", "aligned"),
                                ("reversed_both", "reversed"),
                                ("aligned_agent", "aligned"),
                                ("aligned_patient", "aligned")):
            z, delta = patched_logits(model, passive, donor[donor_key],
                                      cache[donor_key], "cpu", site, mode)
            assert torch.isfinite(z).all() and delta >= 0
            changed.append(float((z-native).abs().max()))
    assert max(changed) > 1e-6, "Source transplant made no change on random model"
    stats = m91.analyze_logits(native, passive)
    assert stats["n_source_templates"] == len(passive)
    print(f"PREFLIGHT PASS: {len(groups['passive'])} novel templates, "
          "four-query paired semantics and sentence exclusion checked")
    print("Passive self and unchanged first-event patches are no-ops at L1/L2/L3; "
          "L3-only source patch is a no-op; prior-question source invariance PASS")
    print(f"Synthetic model nontrivial L1/L2 source effects PASS, max |delta logit|={max(changed):.4g}")
    print("No trained checkpoint loaded; no protected case generated or inspected.")


@torch.inference_mode()
def run(old_hash, plan_hash, groups, device):
    if REPORT.exists():
        raise FileExistsError(f"ML-092 already evaluated: {REPORT}; do not rerun")
    passed, counts = m88.model_gate(old_hash)
    if not passed:
        raise PermissionError(f"ML-088 frozen checkpoints failed competence gate: {counts}")
    results = []
    for seed in SEEDS:
        for arm in ARMS:
            model = ml.load_checkpoint(
                ROOT / f"seed_{seed}" / arm / "model_step_12800.pt",
                seed, arm, old_hash, device)
            native_parts, active_parts = [], []
            patches = {f"L{layer+1}/{mode}": [] for layer in SITE_LAYERS for mode in MODES}
            patch_deltas = {k: [] for k in patches}
            output_deltas = {k: [] for k in patches}
            for start in range(0, len(groups["passive"]), BATCH_TEMPLATES):
                end = start + BATCH_TEMPLATES
                passive = groups["passive"][start:end]
                active = groups["active"][start:end]
                pool = donors(passive)
                states = {k: trace(model, v, device) for k, v in pool.items()}
                native = native_logits(model, passive, device)
                active_native = native_logits(model, active, device)
                native_parts.append(native)
                active_parts.append(active_native)
                for site in SITE_LAYERS:
                    for mode in MODES:
                        key = ("reversed" if mode == "reversed_both" else
                               "self" if mode == "self_both" else "aligned")
                        out, delta = patched_logits(model, passive, pool[key],
                                                    states[key], device, site, mode)
                        label = f"L{site+1}/{mode}"
                        patches[label].append(out)
                        patch_deltas[label].append(delta)
                        output_deltas[label].append(float((out-native).norm(dim=1).mean()))
            baseline_logits = torch.cat(native_parts)
            active_logits = torch.cat(active_parts)
            base = m91.analyze_logits(baseline_logits, groups["passive"])
            active_base = m91.analyze_logits(active_logits, groups["active"])
            inter = {}
            for label, chunks in patches.items():
                logits = torch.cat(chunks)
                metric = m91.analyze_logits(logits, groups["passive"])
                if label.endswith("self_both") or label.endswith("first_event_control"):
                    assert (logits-baseline_logits).abs().max().item() < 1e-4, label
                inter[label] = {
                    "metrics": compact(metric),
                    "delta_second_event_role_gating": metric["second_event_role_gating"]-base["second_event_role_gating"],
                    "delta_four_choice_accuracy": metric["four_choice_accuracy"]-base["four_choice_accuracy"],
                    "delta_second_question_accuracy": 0.5 * sum(
                        metric["per_question_four_choice_accuracy"][q] -
                        base["per_question_four_choice_accuracy"][q]
                        for q in ("second_agent", "second_patient")),
                    "mean_source_patch_norm": mean(patch_deltas[label]),
                    "mean_output_logit_delta_l2": mean(output_deltas[label]),
                }
            row = {
                "seed":seed,"arm":arm,
                "native_passive":compact(base),"native_active":compact(active_base),
                "interventions":inter,
            }
            results.append(row)
            aligned = inter["L1/aligned_both"]
            reversed_role = inter["L1/reversed_both"]
            print(f"seed={seed} arm={arm:8s} native_second="
                  f"{0.5*(base['per_question_four_choice_accuracy']['second_agent']+base['per_question_four_choice_accuracy']['second_patient']):.3f} "
                  f"aligned_second_Delta={aligned['delta_second_question_accuracy']:+.3f} "
                  f"reversed_second_Delta={reversed_role['delta_second_question_accuracy']:+.3f} "
                  f"aligned_gating_Delta={aligned['delta_second_event_role_gating']:+.3f} "
                  f"matched_minus_reversed_gating="
                  f"{aligned['metrics']['second_event_role_gating']-reversed_role['metrics']['second_event_role_gating']:+.3f}",
                  flush=True)

    lookup = {(r["seed"], r["arm"]):r for r in results}
    def select(r, layer, mode, key):
        return r["interventions"][f"L{layer}/{mode}"][key]
    summary = {}
    for arm in ARMS:
        ss = [lookup[seed,arm] for seed in SEEDS]
        for layer in (1, 2):
            contrast = [
                select(r,layer,"aligned_both","metrics")["second_event_role_gating"] -
                select(r,layer,"reversed_both","metrics")["second_event_role_gating"]
                for r in ss
            ]
            aligned_change = [select(r,layer,"aligned_both","delta_second_event_role_gating") for r in ss]
            rescue = [select(r,layer,"aligned_both","delta_second_question_accuracy") for r in ss]
            control_change = [select(r,layer,"reversed_both","delta_second_question_accuracy") for r in ss]
            summary[f"{arm}/L{layer}"] = {
                "matched_minus_reversed_gating_by_seed":dict(zip(map(str,SEEDS), contrast)),
                "matched_minus_reversed_gating_mean":mean(contrast),
                "matched_minus_reversed_gating_ci":bootstrap(contrast),
                "positive_seed_pairs":sum(v>0 for v in contrast),
                "matched_minus_native_gating_mean":mean(aligned_change),
                "matched_mean_second_accuracy_change":mean(rescue),
                "reversed_mean_second_accuracy_change":mean(control_change),
                "matched_mean_patch_norm":mean(select(r,layer,"aligned_both","mean_source_patch_norm") for r in ss),
                "reversed_mean_patch_norm":mean(select(r,layer,"reversed_both","mean_source_patch_norm") for r in ss),
            }
    primary = [
        lookup[seed,"familiar"]["interventions"]["L1/aligned_both"]["metrics"]["second_event_role_gating"] -
        lookup[seed,"familiar"]["interventions"]["L1/reversed_both"]["metrics"]["second_event_role_gating"]
        for seed in SEEDS
    ]
    summary["primary_exploratory"] = {
        "familiar_L1_matched_minus_reversed_gating_mean":mean(primary),
        "bootstrap_95pct_ci":bootstrap(primary),
        "positive_seeds":sum(v>0 for v in primary),
        "n_seeds":len(SEEDS),
        "no_confirmatory_success_rule":True,
    }
    report = {
        "experiment":"ML-092", "status":"EXPLORATORY_DISCOVERY",
        "plan_sha256":plan_hash,"ml088_protocol_sha256":old_hash,
        "protected_cases_generated":False,
        "rows":results,"summary":summary,
        "interpretation_warning":"Source-token activation donor is before question but can encode relation/answer semantics. Matched vs reversed donor differs in grammatical order and latent position. Results show sufficiency of this transplant, not a unique or native circuit. Previously examined models and generator; independent confirmation requires new cohort.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print("\nML-092 EXPLORATORY SUMMARY\n" + json.dumps(summary, indent=2))
    print("Saved", REPORT)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=("preflight","plan","run"),required=True)
    p.add_argument("--device", choices=("auto","cpu","mps","cuda"), default="auto")
    args=p.parse_args()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    old_hash, groups, ml091_hash = select_cases()
    sp = specification(old_hash,groups,ml091_hash)
    if args.phase == "preflight":
        if PLAN.exists():
            freeze(sp, False)
        preflight(groups)
    else:
        h = freeze(sp, args.phase == "plan")
        if args.phase == "plan":
            print("Frozen ML-092 exploratory plan:", PLAN, "sha256", h)
            print("No trained models loaded; no protected cases accessed.")
        else:
            run(old_hash,h,groups,ml.select_device(args.device))


if __name__=="__main__":
    main()
