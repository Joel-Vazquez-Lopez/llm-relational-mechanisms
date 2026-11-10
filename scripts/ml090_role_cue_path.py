"""ML-090: exploratory role-cue path dependence in the existing ML-088 models.

SCIENTIFIC QUESTION
Can the model distinguish agent vs patient within the queried event on the
unfamiliar second-passive construction, even when its four-choice answer is
wrong? How much of that role-selective computation relies on paths from the
question's role token, relative to event and generic-question tokens?

No donor activations, imported gold answers, retraining, or protected examples.
Uses 96 new templates (calibration pairs, indices 4:8 per pair; ML-089 used
0:4), each evaluated with both role questions and 2x2 voice/position controls.

RUN FROM REPOSITORY ROOT
  python3 scripts/ml090_role_cue_path.py --phase plan
  python3 scripts/ml090_role_cue_path.py --phase preflight
  # Review, commit script + reports/ml090/frozen_discovery_plan.json
  python3 scripts/ml090_role_cue_path.py --phase run
  # Inspect then commit reports/ml090/discovery_evaluation.json

This is EXPLORATORY. It is not a novel held-out model cohort or a protected
confirmatory test. Cutting attention to a cue measures sensitivity to that
specific intervention, not the unique native path for semantic binding.
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

SEEDS = (701, 709, 719, 727, 733, 739, 743, 751)
ROOT = Path("results/ml088_fresh_models")
PLAN = Path("reports/ml090/frozen_discovery_plan.json")
OUTPUT = Path("reports/ml090/discovery_evaluation.json")
N_PER_PAIR = 4
START_OFFSET = 4                 # ML-089 selected options 0:4; disjoint templates
BATCH = 64
CONSTRUCTIONS = (
    "first_active", "first_passive", "second_active", "second_passive"
)
KEYS = ("role", "event", "who")
START_LAYERS = (0, 1, 2)         # cut key from this block through final block


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def sample_cases():
    # Calling m88.prepare exactly once is important: it configures ml globals.
    vocab, tr, ca, held, base, cal, dose, fit, valid = m88.prepare()
    old_hash = m88.frozen(
        m88.specification(vocab, tr, ca, held, fit, valid), False
    )
    assert tuple(ml.SEEDS) == SEEDS
    train_pairs = {p for rd in tr for p in rd}
    held_pairs = {p for rd in held for p in rd}
    cal_pairs = sorted({p for rd in ca for p in rd})
    assert len(cal_pairs) == 24
    assert not set(cal_pairs) & (train_pairs | held_pairs)

    candidates = defaultdict(list)
    for c in cal:
        if (c.target_pos == 0 and c.target_voice == "passive"
                and c.role == "agent" and c.target_verb in ml.VERBS[2:]):
            candidates[c.target_pair].append(c)

    selected = []
    previous = []
    for pair in cal_pairs:
        options = sorted(candidates[pair], key=lambda c: digest(ml.prompt(c)))
        assert len(options) >= START_OFFSET + N_PER_PAIR
        previous.extend(options[:START_OFFSET])
        selected.extend(options[START_OFFSET:START_OFFSET + N_PER_PAIR])
    assert len(selected) == 96 and len(previous) == 96
    assert not {tuple(ml.prompt(c)) for c in selected} & {
        tuple(ml.prompt(c)) for c in previous
    }, "ML-089 discovery templates must be disjoint"

    result = {}
    for construction in CONSTRUCTIONS:
        pos = 0 if construction.startswith("first") else 1
        voice = "active" if construction.endswith("active") else "passive"
        cases = []
        for agent_case in selected:
            a = replace(agent_case, target_pos=pos, target_voice=voice)
            p = replace(a, role="patient")
            assert ml.answer(a) != ml.answer(p)
            t0, t1 = ml.prompt(a), ml.prompt(p)
            assert len(t0) == len(t1)
            diffs = [(i, x, y) for i, (x, y) in enumerate(zip(t0, t1)) if x != y]
            assert len(diffs) == 1 and diffs[0][1:] == ("agent", "patient")
            assert diffs[0][0] == len(t0) - 3
            cases.extend([a, p])
        assert len(cases) == 192
        result[construction] = cases

    # The four constructions have identical semantic gold answers, event for event.
    for rows in zip(*[result[k] for k in CONSTRUCTIONS]):
        assert len(set(ml.answer(c) for c in rows)) == 1
        assert len(set(c.target_pair for c in rows)) == 1
        assert len(set(c.target_dir for c in rows)) == 1
        assert len(set(c.role for c in rows)) == 1
    return old_hash, result


def spec(old_hash, examples):
    return {
        "experiment": "ML-090",
        "status": "EXPLORATORY_CALIBRATION_ONLY_PREDECLARED_DISCOVERY",
        "script_sha256": sha(__file__),
        "ml080_source_sha256": sha(ml.__file__),
        "ml088_source_sha256": sha(m88.__file__),
        "ml088_protocol_sha256": old_hash,
        "seeds": SEEDS,
        "arms": ml.ARMS,
        "existing_models": "16 frozen ML-088 checkpoints, step 12800",
        "selection": (
            "Among calibration pairs, SHA256-json sort first-passive agent "
            "templates with exposure-unseen verbs; use indices [4:8] from "
            "each of 24 target pairs, disjoint from ML-089's [0:4]"
        ),
        "n_templates": 96,
        "per_construction_questions": 192,
        "constructions": CONSTRUCTIONS,
        "question_counterfactual": (
            "same entire sentence, agent vs patient single-token cue; "
            "all four voice-by-position constructions preserve event semantics"
        ),
        "prompt_sha256": {
            k: digest([ml.prompt(c) for c in v]) for k, v in examples.items()
        },
        "interventions": {
            "method": (
                "Mask attention FROM the role, event, or who question-token KEY "
                "TO ALL strictly later positions, from a specified block "
                "through the final transformer block. No activation donor."
            ),
            "from_blocks_1indexed": [1, 2, 3],
            "roles": KEYS,
            "control": "first/second event-cue key and fixed who key",
            "sanity": (
                "Cutting the role-key path from layer 1 through all "
                "layers must make agent/patient logits identical. "
                "This is a structural check, NOT the empirical discovery."
            ),
        },
        "primary_exploratory_question": (
            "Among target second-passive questions, is target-pair 2-choice "
            "role accuracy substantially higher than native 4-choice "
            "accuracy, including on native wrong-event predictions? "
            "How does removal of role-key access from L2 onward change "
            "role-conditioned target-pair logit evidence relative to "
            "event and who control cuts, in each exposure arm?"
        ),
        "summary_comparison": (
            "Paired target-minus-familiar arm contrast in the role-cut-vs-who-cut "
            "effect on role-antisymmetric target-pair logit evidence for "
            "second-passive, also report second-active and first-passive controls."
        ),
        "metrics": [
            "four-choice accuracy", "full-vocabulary accuracy",
            "within-target-pair two-choice role accuracy",
            "within-distractor-pair two-choice role accuracy",
            "two-choice target-role accuracy conditional on wrong-event native output",
            "wrong-event prediction role correctness",
            "pairwise agent/patient evidence and flip rates", "gold margin",
            "native-error rescues and native-correct harms", "output-logit delta norm",
        ],
        "interpretation": (
            "No confirmation claim. Even strong two-choice performance need not "
            "show native event binding: two-choice decoding supplies the target "
            "pair externally. Cutting a token key may produce generic "
            "out-of-distribution disruption and paths can be redundant. "
            "The main question is information-flow dependence, not a proof of a circuit."
        ),
        "safety": (
            "Never invoke protected_target_cases or ML-088 protected cases; "
            "never rerun ML-089; one-time ML-090 calibration discovery report."
        ),
    }


def freeze(document, create=False):
    canonical = json.loads(json.dumps({**document, "sha256": digest(document)}))
    if create and not PLAN.exists():
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(json.dumps(canonical, indent=2) + "\n")
    if not PLAN.exists():
        raise FileNotFoundError("Run --phase plan before preflight/run")
    assert json.loads(PLAN.read_text()) == canonical, (
        "ML-090 script/source/selection changed after plan"
    )
    return canonical["sha256"]


def cue_pos(c, key):
    tokens = ml.prompt(c)
    sep = tokens.index("<sep>")
    if key == "role":
        word = c.role
    elif key == "event":
        word = "first" if c.target_pos == 0 else "second"
    elif key == "who":
        word = "who"
    else:
        raise ValueError(key)
    positions = [i for i in range(sep + 1, len(tokens)) if tokens[i] == word]
    assert len(positions) == 1, (key, positions, tokens)
    assert positions[0] < len(tokens) - 1
    return positions[0]


def attn_mask(cases, n, device, dtype, key=None):
    future = torch.ones(n, n, dtype=torch.bool, device=device).triu(1)
    if key is None:
        return future
    mask = torch.zeros(len(cases), n, n, device=device, dtype=dtype)
    mask.masked_fill_(future.unsqueeze(0), float("-inf"))
    for i, c in enumerate(cases):
        position = cue_pos(c, key)
        assert position < len(ml.prompt(c)) - 1
        # Block only transmissions FROM this token to LATER query positions.
        mask[i, position + 1:, position] = -float("inf")
    return mask.repeat_interleave(ml.HEADS, dim=0)


@torch.inference_mode()
def forward(model, cases, device, key=None, start_layer=0):
    ids, readout, gold, candidates = ml.encode(cases)
    ids, readout = ids.to(device), readout.to(device)
    batch, n = ids.shape
    x = model.embed(ids) + model.pos(torch.arange(n, device=device))
    ix = torch.arange(batch, device=device)
    for layer, block in enumerate(model.blocks):
        z = block.n1(x)
        cut = key if (key is not None and layer >= start_layer) else None
        mask = attn_mask(cases, n, device, z.dtype, cut)
        attn = block.attn(z, z, z, attn_mask=mask, need_weights=False)[0]
        x = x + attn
        x = x + block.mlp(block.n2(x))
    logits = model.out(model.norm(x[ix, readout])).float()
    return logits.cpu(), gold, candidates


def semantic_ids(cases):
    targ_agent = []
    targ_patient = []
    other_agent = []
    other_patient = []
    for c in cases:
        ta, tp = c.target_pair if c.target_dir == 0 else c.target_pair[::-1]
        da, dp = c.distractor_pair if c.distractor_dir == 0 else c.distractor_pair[::-1]
        targ_agent.append(ml.VOCAB[ta])
        targ_patient.append(ml.VOCAB[tp])
        other_agent.append(ml.VOCAB[da])
        other_patient.append(ml.VOCAB[dp])
    return tuple(torch.tensor(x, dtype=torch.long) for x in (
        targ_agent, targ_patient, other_agent, other_patient
    ))


def accuracy(pred, gold):
    return float((pred == gold).float().mean())


def scores(logits, cases):
    assert len(cases) % 2 == 0
    _, _, gold, candidates = ml.encode(cases)
    n = len(cases)
    ix = torch.arange(n)
    ta, tp, da, dp = semantic_ids(cases)
    assert bool((gold[::2] == ta[::2]).all())
    assert bool((gold[1::2] == tp[1::2]).all())

    restricted = logits.gather(1, candidates)
    pred = candidates.gather(1, restricted.argmax(1, keepdim=True)).flatten()
    native_correct = pred == gold
    target_pair = torch.stack([ta, tp], dim=1)
    wrong_pair = torch.stack([da, dp], dim=1)
    target_pred = target_pair.gather(1, logits.gather(1, target_pair).argmax(1, keepdim=True)).flatten()
    other_pred = wrong_pair.gather(1, logits.gather(1, wrong_pair).argmax(1, keepdim=True)).flatten()
    wanted_distractor = torch.where(
        torch.tensor([c.role == "agent" for c in cases]), da, dp
    )
    wrong_event = (pred == da) | (pred == dp)
    wrong_role = (pred == ta) | (pred == tp)
    wrong_role &= ~native_correct
    assert bool((native_correct | wrong_event | wrong_role).all())
    assert not bool((wrong_event & wrong_role).any())

    correct_score = logits[ix, gold]
    rival = logits.gather(1, candidates).masked_fill(candidates == gold[:, None], -float("inf")).max(1).values
    target_rival = torch.where(torch.tensor([c.role == "agent" for c in cases]), tp, ta)
    target_margin = logits[ix, gold] - logits[ix, target_rival]

    # Difference-of-differences for paired agent/patient questions about the
    # exact same sentence. Positive if logit evidence tracks role correctly.
    role_evidence = 0.5 * (
        (logits[::2, ta[::2]] - logits[::2, tp[::2]])
        - (logits[1::2, ta[1::2]] - logits[1::2, tp[1::2]])
    )
    # Naively conditioning on wrong-event cases would use zero denominator
    # for high-accuracy models; retain counts + nullable rates.
    wrong_count = int(wrong_event.sum())
    return {
        "n": n,
        "choice_accuracy": accuracy(pred, gold),
        "full_vocab_accuracy": accuracy(logits.argmax(1), gold),
        "both_roles_correct_rate": float((native_correct[::2] & native_correct[1::2]).float().mean()),
        "role_query_prediction_flip_rate": float((pred[::2] != pred[1::2]).float().mean()),
        "within_target_twochoice_accuracy": accuracy(target_pred, gold),
        "within_distractor_twochoice_role_accuracy": accuracy(other_pred, wanted_distractor),
        "target_twochoice_given_wrong_event": (
            accuracy(target_pred[wrong_event], gold[wrong_event]) if wrong_count else None
        ),
        "distractor_role_correct_given_wrong_event": (
            accuracy(pred[wrong_event], wanted_distractor[wrong_event]) if wrong_count else None
        ),
        "wrong_event_count": wrong_count,
        "wrong_event_rate": float(wrong_event.float().mean()),
        "wrong_role_count": int(wrong_role.sum()),
        "wrong_role_rate": float(wrong_role.float().mean()),
        "mean_gold_vs_best_other_margin": float((correct_score - rival).mean()),
        "mean_within_target_role_margin": float(target_margin.mean()),
        "mean_role_antisymmetric_evidence": float(role_evidence.mean()),
        "correct_mask": native_correct,
    }


def compact(s):
    return {k: v for k, v in s.items() if k != "correct_mask"}


def bootstrap(xs):
    rng = random.Random(90090)
    n = len(xs)
    dist = sorted(mean(xs[rng.randrange(n)] for _ in range(n)) for _ in range(20000))
    return [dist[500], dist[19500]]


@torch.inference_mode()
def preflight(examples):
    torch.manual_seed(90090)
    model = ml.Model().eval()
    cases = examples["second_passive"][:16]
    ids, ro, _, _ = ml.encode(cases)
    direct = model(ids, ro).float()
    native, _, _ = forward(model, cases, "cpu")
    role_cut, _, _ = forward(model, cases, "cpu", "role", 0)
    event_cut, _, _ = forward(model, cases, "cpu", "event", 0)
    who_cut, _, _ = forward(model, cases, "cpu", "who", 0)
    role_l2, _, _ = forward(model, cases, "cpu", "role", 1)
    native_err = (native - direct).abs().max().item()
    invariant_err = (role_cut[::2] - role_cut[1::2]).abs().max().item()
    assert native_err < 1e-4, native_err
    assert invariant_err < 1e-4, invariant_err
    assert (role_l2 - role_cut).abs().max() > 1e-6
    assert torch.isfinite(native).all() and torch.isfinite(event_cut).all() and torch.isfinite(who_cut).all()
    assert not torch.allclose(role_cut, event_cut)

    mask = attn_mask(cases, ids.shape[1], "cpu", torch.float32, "role")
    assert mask.shape == (len(cases) * ml.HEADS, ids.shape[1], ids.shape[1])
    for i, c in enumerate(cases):
        p = cue_pos(c, "role")
        for head in range(ml.HEADS):
            q = mask[i * ml.HEADS + head]
            assert torch.isneginf(q[p+1:, p]).all()
            assert q[p, p] == 0
            assert torch.isneginf(q.triu(1)[q.triu(1) == -float("inf")]).all()  # shape/syntax sanity
    a = scores(native, cases)
    b = scores(role_cut, cases)
    assert abs(b["within_target_twochoice_accuracy"] - 0.5) < 1e-6
    assert abs(b["mean_role_antisymmetric_evidence"]) < 1e-4
    print(f"PREFLIGHT PASS: native parity={native_err:.2e}; "
          f"all-layer role-cut paired-logit invariance={invariant_err:.2e}; "
          f"16 prompts, 96 new templates, 4 construction cells")
    print(f"Untrained toy native role evidence={a['mean_role_antisymmetric_evidence']:.4f}; "
          f"all-role-cut evidence={b['mean_role_antisymmetric_evidence']:.6f}")
    print("No trained checkpoints, protected examples, or prior discovery cases accessed.")


@torch.inference_mode()
def run(old_hash, plan_hash, examples, device):
    if OUTPUT.exists():
        raise FileExistsError(f"Discovery already evaluated: {OUTPUT}")
    gate_ok, counts = m88.model_gate(old_hash)
    if not gate_ok:
        raise PermissionError(f"ML-088 saved model competence gate failed: {counts}")
    cases_by_key = examples
    rows = []
    for seed in SEEDS:
        for arm in ml.ARMS:
            model = ml.load_checkpoint(
                ROOT / f"seed_{seed}" / arm / "model_step_12800.pt",
                seed, arm, old_hash, device
            )
            for construction in CONSTRUCTIONS:
                cases = cases_by_key[construction]
                native_parts = []
                for k in range(0, len(cases), BATCH):
                    z, _, _ = forward(model, cases[k:k+BATCH], device)
                    native_parts.append(z)
                native = torch.cat(native_parts)
                baseline = scores(native, cases)
                interventions = {}
                for key in KEYS:
                    for layer in START_LAYERS:
                        logits = []
                        for k in range(0, len(cases), BATCH):
                            z, _, _ = forward(model, cases[k:k+BATCH], device, key, layer)
                            logits.append(z)
                        patched = torch.cat(logits)
                        stats = scores(patched, cases)
                        rescued = int((~baseline["correct_mask"] & stats["correct_mask"]).sum())
                        harmed = int((baseline["correct_mask"] & ~stats["correct_mask"]).sum())
                        label = f"{key}_cut_from_L{layer+1}"
                        interventions[label] = {
                            **compact(stats),
                            "rescued_native_errors": rescued,
                            "harmed_native_correct": harmed,
                            "mean_output_logit_delta_l2": float((patched - native).norm(dim=1).mean()),
                            "delta_role_evidence": (
                                stats["mean_role_antisymmetric_evidence"] -
                                baseline["mean_role_antisymmetric_evidence"]
                            ),
                            "delta_choice_accuracy": stats["choice_accuracy"] - baseline["choice_accuracy"],
                        }
                # Architectural sanity: with all pathways from role word to later
                # query positions cut, both role questions must yield same logits.
                role_all = interventions["role_cut_from_L1"]
                if not abs(role_all["within_target_twochoice_accuracy"] - 0.5) < 1e-5:
                    raise AssertionError("All-role-cut should give exactly 50% target-pair two-choice")
                if abs(role_all["mean_role_antisymmetric_evidence"]) > 1e-4:
                    raise AssertionError("All-role-cut did not remove role-word influence")
                rows.append({
                    "seed":seed, "arm":arm, "construction":construction,
                    "native":compact(baseline), "interventions":interventions
                })
                q = interventions["role_cut_from_L2"]
                control = interventions["who_cut_from_L2"]
                print(f"seed={seed} {arm:8s} {construction:14s} "
                      f"native4={baseline['choice_accuracy']:.3f} "
                      f"target2={baseline['within_target_twochoice_accuracy']:.3f} "
                      f"wrong_event={baseline['wrong_event_rate']:.3f} "
                      f"role_evidence={baseline['mean_role_antisymmetric_evidence']:+.2f} "
                      f"L2+role_vs_who={(control['mean_role_antisymmetric_evidence']-q['mean_role_antisymmetric_evidence']):+.2f}",
                      flush=True)

    def get(seed, arm, construction):
        return next(r for r in rows if (r["seed"],r["arm"],r["construction"]) == (seed,arm,construction))
    overview = {}
    for construction in CONSTRUCTIONS:
        for arm in ml.ARMS:
            subset = [get(seed,arm,construction) for seed in SEEDS]
            mean_native = {k:mean(r["native"][k] for r in subset)
                for k in ("choice_accuracy", "within_target_twochoice_accuracy", "wrong_event_rate",
                          "wrong_role_rate", "mean_role_antisymmetric_evidence", "both_roles_correct_rate")}
            selectivity = [
                r["interventions"]["who_cut_from_L2"]["mean_role_antisymmetric_evidence"] -
                r["interventions"]["role_cut_from_L2"]["mean_role_antisymmetric_evidence"]
                for r in subset
            ]
            overview[f"{arm}/{construction}"] = {
                "native_mean": mean_native,
                "L2plus_role_vs_who_role_evidence_by_seed":dict(zip(map(str,SEEDS),selectivity)),
                "L2plus_role_vs_who_mean":mean(selectivity),
                "L2plus_role_vs_who_positive_seeds":sum(v>0 for v in selectivity),
            }
    effect = []
    for seed in SEEDS:
        def sel(arm,cons):
            rec=get(seed,arm,cons)["interventions"]
            return (rec["who_cut_from_L2"]["mean_role_antisymmetric_evidence"] -
                    rec["role_cut_from_L2"]["mean_role_antisymmetric_evidence"])
        effect.append(sel("target","second_passive")-sel("familiar","second_passive"))
    summary = {
        "cells":overview,
        "paired_exposure_effect_on_L2plus_role_specific_evidence_by_seed":dict(zip(map(str,SEEDS),effect)),
        "paired_exposure_effect_mean":mean(effect),
        "paired_exposure_effect_seed_bootstrap_ci":bootstrap(effect),
        "paired_exposure_effect_positive_seeds":sum(v>0 for v in effect),
        "limitations": (
            "Exploratory calibration-only; 2-choice probes provide the target pair "
            "externally; mask effects are not a unique semantic or grammatical pathway; "
            "model seeds share the generator and architecture."
        ),
    }
    report = {
        "experiment":"ML-090", "status":"EXPLORATORY_DISCOVERY",
        "plan_sha256":plan_hash, "ml088_protocol_sha256":old_hash,
        "protected_examples_generated":False,"rows":rows,"summary":summary
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print("\nML-090 EXPLORATORY SUMMARY\n" + json.dumps(summary, indent=2))
    print("Saved", OUTPUT)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("plan","preflight","run"))
    ap.add_argument("--device", choices=("auto","cpu","mps","cuda"), default="auto")
    args=ap.parse_args()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    old_hash, examples=sample_cases()
    plan_hash=freeze(spec(old_hash,examples), create=args.phase=="plan")
    if args.phase=="plan":
        print(f"ML-090 discovery plan frozen: {PLAN} hash={plan_hash[:12]}")
        print("No model inference or protected examples.")
    elif args.phase=="preflight":
        preflight(examples)
    else:
        run(old_hash,plan_hash,examples,ml.select_device(args.device))


if __name__ == "__main__":
    main()
