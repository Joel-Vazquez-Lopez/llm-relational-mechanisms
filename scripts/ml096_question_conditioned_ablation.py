"""ML-096: Question-conditioned source ablations and error transitions.

A focused EXPLORATORY follow-up to ML-095, using the SAME frozen 16 models and
96 calibration sentences (four questions each). It distinguishes first-event
and second-event effects and tracks transitions between correct, wrong-event
and correct-event/wrong-role responses. It does not evaluate protected items.

From repo root:
  python3 scripts/ml096_question_conditioned_ablation.py --phase preflight
  python3 scripts/ml096_question_conditioned_ablation.py --phase plan
  # Review, then commit script and frozen plan BEFORE trained-model inference.
  python3 scripts/ml096_question_conditioned_ablation.py --phase run

Primary DESCRIPTIVE endpoint: within familiar arm, second-question accuracy
change from HALF-strength ablation of first-event participants. Compare
wrong-event->correct and correct->wrong-event transitions. No pass/fail rule.

Why: ML-095 first-pair zero improved second accuracy, but reported aggregate
prediction flips mixed first- and second-event queries. This audit separates
them, tests dose direction at 50% and 100%, and examines asymmetric outcomes.
Sites do not have identical activation magnitudes; no semantic circuit claim.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean

import torch

import ml080_learning_curve as ml
import ml088_probe_access as m88
import ml091_counterfactual_binding as m91
import ml092_source_state_transfer as m92
import ml095_source_site_dependency as m95

SEEDS = (701, 709, 719, 727, 733, 739, 743, 751)
ARMS = ("familiar", "target")
ROOT = Path("results/ml088_fresh_models")
PLAN = Path("reports/ml096/frozen_discovery_plan.json")
REPORT = Path("reports/ml096/discovery_evaluation.json")
PRIOR = m95.REPORT
BATCH_SENTENCES = 12
MODES = (
    "native", "first_pair_half", "first_pair_zero", "second_pair_half",
    "second_pair_zero", "second_agent_half", "second_patient_half", "sep_zero"
)
ZERO_PARITY = {
    "native": "native",
    "first_pair_zero": "zero_first_pair_L1",
    "second_pair_zero": "zero_second_pair_L1",
    "sep_zero": "zero_sep_L1",
}
SITES = {
    "first_pair_half": ("first_agent", "first_patient"),
    "first_pair_zero": ("first_agent", "first_patient"),
    "second_pair_half": ("second_agent", "second_patient"),
    "second_pair_zero": ("second_agent", "second_patient"),
    "second_agent_half": ("second_agent",),
    "second_patient_half": ("second_patient",),
    "sep_zero": ("sep",),
}
STRENGTH = {mode: (1.0 if mode.endswith("zero") else 0.5)
            for mode in SITES}
STRENGTH["sep_zero"] = 1.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def bootstrap(values, n=20000, seed=96096):
    assert len(values) == len(SEEDS)
    rng = random.Random(seed)
    dist = sorted(mean(values[rng.randrange(len(values))] for _ in values)
                  for _ in range(n))
    return [dist[int(.025*n)], dist[int(.975*n)-1]]


def prepare():
    old_hash, groups, m94_hash = m95.prepare()
    m95_spec = m95.specification(old_hash, groups, m94_hash)
    m95_hash = m95.freeze(m95_spec, False)
    if not PRIOR.is_file():
        raise FileNotFoundError(
            "Please preserve reports/ml095/discovery_evaluation.json first"
        )
    doc = json.loads(PRIOR.read_text())
    assert doc["experiment"] == "ML-095"
    assert doc["plan_sha256"] == m95_hash
    assert doc["protected_examples_generated"] is False
    assert len(doc["rows"]) == 16
    assert len(groups["passive"]) == 96
    return old_hash, groups, m95_hash, doc


def specification(old_hash, groups, prev_hash):
    return {
        "experiment": "ML-096",
        "status": "EXPLORATORY_QUESTION_CONDITIONED_SOURCE_ABLATIONS",
        "script_sha256": sha(__file__),
        "dependencies_sha256": {
            name: sha(mod.__file__) for name,mod in
            (("ml080",ml),("ml088",m88),("ml091",m91),
             ("ml092",m92),("ml095",m95))
        },
        "ml088_protocol_sha256": old_hash,
        "ml095_plan_sha256": prev_hash,
        "ml095_report_sha256": sha(PRIOR),
        "models": "Same 16 ML-088 12800-step checkpoints, eight paired seeds",
        "examples": "Exactly 96 previously examined ML-092–095 passive calibration sentences, four questions each",
        "sentence_sha256": digest(
            [m91.sentence(g[0]) for g in groups["passive"]]
        ),
        "modes": MODES,
        "interventions": {
            "site": "Post-L1 residual source tokens, at original receiver positions",
            "scaling": "For half ablations, multiply selected token residuals by 0.5. For zero ablations set to zero. No donor.",
            "pair": "Both event-participant name tokens; first versus second event",
            "single": "Agent or patient of the second event",
            "control": "Source-end SEP zero; strict native identity",
            "identical_questions": "Same source-side changes across four query variants of each sentence"
        },
        "primary_descriptive": "Familiar arm, first_pair_half minus native accuracy on SECOND-event questions only, with wrong-event->correct, wrong-role->correct, correct->wrong-event, correct->wrong-role counts",
        "secondary": "Per-question accuracy, first/second question prediction flip rates, dose dependence, second-event outcome transitions, all-seed paired summaries, target-arm effects",
        "parity": "Unmodified native, full first-pair zero, full second-pair zero and SEP zero reproduce ML-095's already preserved aggregate endpoints",
        "limits": "Same models and cases; post-hoc motivated diagnostic, not confirmatory. Zero/half ablations are off-manifold and have non-matched norms across sites. They may alter competition rather than representational binding.",
        "safety": "No ML-088 protected examples. No overwrite of earlier reports. One-run report lock."
    }


def freeze(spec, create=False):
    expected = json.loads(json.dumps({**spec, "sha256": digest(spec)}))
    if create and not PLAN.exists():
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(json.dumps(expected, indent=2)+"\n")
    if not PLAN.exists():
        raise FileNotFoundError("Run --phase plan before --phase run")
    assert json.loads(PLAN.read_text()) == expected, "Frozen ML-096 code/inputs differ"
    return expected["sha256"]


@torch.inference_mode()
def evaluate(model, groups, device):
    outputs = {mode: [] for mode in MODES}
    norms = {mode: [] for mode in MODES}
    for start in range(0, len(groups), BATCH_SENTENCES):
        sub = groups[start:start+BATCH_SENTENCES]
        cases = [q for group in sub for q in group]
        ids, ro, _, _ = ml.encode(cases)
        ids,ro = ids.to(device),ro.to(device)
        n,L = ids.shape
        ix = torch.arange(n,device=device)
        locations = m95.positions(sub,device)
        x = model.embed(ids)+model.pos(torch.arange(L,device=device))
        mask = torch.ones(L,L,dtype=torch.bool,device=device).triu(1)
        x = model.blocks[0](x,mask)
        for mode in MODES:
            modified = x.clone()
            if mode == "native":
                norm = 0.0
            else:
                norm_parts = []
                for name in SITES[mode]:
                    pos = locations[name]
                    before = modified[ix,pos].clone()
                    delta = STRENGTH[mode]*before
                    modified[ix,pos] = before-delta
                    norm_parts.append(delta.norm(dim=-1))
                norm = float(torch.stack(norm_parts).mean())
            modified = model.blocks[1](modified,mask)
            modified = model.blocks[2](modified,mask)
            logits = model.out(model.norm(modified[ix,ro])).float().cpu()
            outputs[mode].append(logits)
            norms[mode].append(norm)
    return {mode: torch.cat(outputs[mode]) for mode in MODES}, {
        mode: mean(norms[mode]) for mode in MODES
    }


def outcome_labels(pred):
    """Predicted index in [first-agent, first-patient, second-agent, second-patient].

    Arrays are [sentences,4]; category 0 correct, 1 other-event,
    2 correct-event wrong-role.
    """
    true = torch.arange(4).view(1,4).expand_as(pred)
    return torch.where(pred == true,0,torch.where(pred//2 != true//2,1,2))


def analyze_transitions(native, patched, groups):
    assert native.shape == patched.shape == (len(groups),4)
    c0, c1 = outcome_labels(native), outcome_labels(patched)
    changed = (native != patched).float()
    result = {}
    for label,indices in (("first",[0,1]),("second",[2,3])):
        a,b = c0[:,indices],c1[:,indices]
        result[label] = {
            "native_accuracy":float((a==0).float().mean()),
            "patched_accuracy":float((b==0).float().mean()),
            "accuracy_change":float((b==0).float().mean()-(a==0).float().mean()),
            "flip_rate":float(changed[:,indices].mean()),
            "native_wrong_event_rate":float((a==1).float().mean()),
            "patched_wrong_event_rate":float((b==1).float().mean()),
            "native_wrong_role_rate":float((a==2).float().mean()),
            "patched_wrong_role_rate":float((b==2).float().mean()),
            "transition_counts": {
                f"{u}_to_{v}":int(((a==i)&(b==j)).sum())
                for i,u in enumerate(("correct","wrong_event","wrong_role"))
                for j,v in enumerate(("correct","wrong_event","wrong_role"))
            }
        }
    result["per_question_accuracy"] = {
        name:float((c1[:,i]==0).float().mean())
        for i,name in enumerate(m91.QUESTION_NAMES)
    }
    result["per_question_flip_rate"] = {
        name:float(changed[:,i].mean())
        for i,name in enumerate(m91.QUESTION_NAMES)
    }
    return result


def predictions(z,groups):
    return m95.semantic_predictions(z,groups).cpu()


def evaluate_one(model, groups, device):
    logits,norms = evaluate(model, groups,device)
    ref = logits["native"]
    baseline = predictions(ref,groups)
    outcome={}
    for mode in MODES:
        pred = predictions(logits[mode],groups)
        analyzed = analyze_transitions(baseline,pred,groups)
        scores = m91.analyze_logits(logits[mode],groups)
        outcome[mode] = {
            "mean_intervention_norm_per_edited_token":norms[mode],
            "four_choice_accuracy":scores["four_choice_accuracy"],
            "second_event_role_gating":scores["second_event_role_gating"],
            "second_accuracy":.5*(scores["per_question_four_choice_accuracy"]["second_agent"]+
                                  scores["per_question_four_choice_accuracy"]["second_patient"]),
            "mean_output_logit_l2_change":float((logits[mode]-ref).norm(dim=-1).mean()),
            **analyzed
        }
    assert abs(outcome["native"]["mean_output_logit_l2_change"])<1e-6
    assert outcome["native"]["first"]["flip_rate"]==0
    assert outcome["native"]["second"]["flip_rate"]==0
    return outcome


def previous_parity(results, prior, seed, arm):
    old = next(r["interventions"] for r in prior["rows"]
               if r["seed"]==seed and r["arm"]==arm)
    for mode, prior_mode in ZERO_PARITY.items():
        new = results[mode]
        reference = old[prior_mode]
        assert abs(new["second_accuracy"]-reference["second_accuracy"])<.012, (seed,arm,mode,"accuracy")
        assert abs(new["second_event_role_gating"]-reference["event_role_gating"])<.04, (seed,arm,mode,"gating")
        flips = (new["first"]["flip_rate"]+new["second"]["flip_rate"])/2
        assert abs(flips-reference["changed_four_choice_prediction_rate"])<.012,(seed,arm,mode,"flip")


def preflight(groups):
    torch.manual_seed(96096)
    small=groups[:6]
    model=ml.Model().eval()
    outcome=evaluate_one(model,small,"cpu")
    assert outcome["native"]["first"]["flip_rate"]==0
    assert outcome["native"]["second"]["flip_rate"]==0
    assert outcome["native"]["four_choice_accuracy"] >= 0
    assert outcome["sep_zero"]["mean_intervention_norm_per_edited_token"]>0
    assert outcome["first_pair_zero"]["mean_intervention_norm_per_edited_token"]>0
    assert outcome["second_agent_half"]["mean_intervention_norm_per_edited_token"]>0
    for mode in MODES:
        assert abs(outcome[mode]["first"]["accuracy_change"] +
                   outcome[mode]["second"]["accuracy_change"] -
                   2*(outcome[mode]["four_choice_accuracy"]-outcome["native"]["four_choice_accuracy"])) < 1e-5
        assert sum(outcome[mode]["second"]["transition_counts"].values())==2*len(small)
    print("PREFLIGHT PASS: question-specific category transitions, native identity, paired source positions")
    print("Zero/half two-name and one-name interventions operational; no trained models or protected examples loaded.")


@torch.inference_mode()
def run(old_hash,plan_hash,groups,prior,device):
    if REPORT.exists():
        raise FileExistsError("ML-096 already evaluated; refusing rerun "+str(REPORT))
    gate,counts=m88.model_gate(old_hash)
    if not gate: raise PermissionError("Competence gate failed: "+str(counts))
    rows=[]
    for seed in SEEDS:
        for arm in ARMS:
            model=ml.load_checkpoint(ROOT/f"seed_{seed}"/arm/"model_step_12800.pt",
                                     seed,arm,old_hash,device)
            result=evaluate_one(model,groups,device)
            previous_parity(result,prior,seed,arm)
            rows.append({"seed":seed,"arm":arm,"interventions":result})
            for label in ("first_pair_half","first_pair_zero","second_pair_half","second_pair_zero"):
                s=result[label]["second"]
                t=s["transition_counts"]
                if label=="first_pair_half":
                    primary = (s["accuracy_change"],
                               t["wrong_event_to_correct"],
                               t["correct_to_wrong_event"])
            print(f"seed={seed} arm={arm:8s} "
                  f"firstHalf_second_dAcc={primary[0]:+.3f} "
                  f"wrongEventToCorrect={primary[1]} "
                  f"correctToWrongEvent={primary[2]} "
                  f"firstZero_second_dAcc={result['first_pair_zero']['second']['accuracy_change']:+.3f} "
                  f"secondZero_second_dAcc={result['second_pair_zero']['second']['accuracy_change']:+.3f}",
                  flush=True)
    summary={}
    for arm in ARMS:
        sub=[r for r in rows if r["arm"]==arm]
        summary[arm]={}
        for mode in MODES:
            xs=[r["interventions"][mode] for r in sub]
            summary[arm][mode]={
                "second_accuracy_change_mean":mean(x["second"]["accuracy_change"] for x in xs),
                "first_accuracy_change_mean":mean(x["first"]["accuracy_change"] for x in xs),
                "second_prediction_flip_mean":mean(x["second"]["flip_rate"] for x in xs),
                "first_prediction_flip_mean":mean(x["first"]["flip_rate"] for x in xs),
                "second_wrong_event_change_mean":mean(x["second"]["patched_wrong_event_rate"]-
                                                       x["second"]["native_wrong_event_rate"] for x in xs),
                "wrong_event_to_correct_mean_count_per_seed":mean(x["second"]["transition_counts"]["wrong_event_to_correct"] for x in xs),
                "wrong_role_to_correct_mean_count_per_seed":mean(x["second"]["transition_counts"]["wrong_role_to_correct"] for x in xs),
                "correct_to_wrong_event_mean_count_per_seed":mean(x["second"]["transition_counts"]["correct_to_wrong_event"] for x in xs),
                "correct_to_wrong_role_mean_count_per_seed":mean(x["second"]["transition_counts"]["correct_to_wrong_role"] for x in xs),
                "mean_edit_norm_per_token":mean(x["mean_intervention_norm_per_edited_token"] for x in xs)
            }
        vals=[r["interventions"]["first_pair_half"]["second"]["accuracy_change"]
              for r in sub]
        summary[arm]["primary_descriptive_firstpair_half_second_accuracy"]={
            "by_seed":dict(zip(map(str,SEEDS),vals)),
            "mean":mean(vals),
            "seed_bootstrap_95pct":bootstrap(vals),
            "positive_seeds":sum(v>0 for v in vals)
        }
    doc={"experiment":"ML-096","status":"EXPLORATORY_QUESTION_CONDITIONED_SOURCE_ABLATIONS",
         "plan_sha256":plan_hash,"ml088_protocol_sha256":old_hash,
         "protected_examples_generated":False,"n_models":16,
         "rows":rows,"summary":summary,
         "warning":"Same trained checkpoints and previously explored calibration sentences; no independent replication. Half and full ablations may be off manifold, do not match norms across sites, and affect answer competition. Conditional transitions are descriptive, not identification of a native semantic algorithm."}
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(doc,indent=2)+"\n")
    for arm in ARMS:
        print("\n"+arm.upper()+" ML-096 SUMMARY")
        for mode in MODES:
            v=summary[arm][mode]
            print(f"  {mode:<20} second_dAcc={v['second_accuracy_change_mean']:+.3f} "
                  f"second_flip={v['second_prediction_flip_mean']:.3f} "
                  f"first_flip={v['first_prediction_flip_mean']:.3f} "
                  f"dWrongEvent={v['second_wrong_event_change_mean']:+.3f}")
        print("  PRIMARY:",json.dumps(summary[arm]["primary_descriptive_firstpair_half_second_accuracy"]))
    print("Saved",REPORT)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",required=True,choices=("preflight","plan","run"))
    ap.add_argument("--device",default="auto",choices=("auto","cpu","mps","cuda"))
    args=ap.parse_args()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    old_hash,groups,prev_hash,prior=prepare()
    spec=specification(old_hash,groups,prev_hash)
    if args.phase=="preflight":
        if PLAN.exists(): freeze(spec,False)
        preflight(groups["passive"])
    else:
        h=freeze(spec,create=args.phase=="plan")
        if args.phase=="plan":
            print("ML-096 frozen plan:",PLAN,"hash:",h)
            print("No trained models or protected cases accessed.")
        else:
            run(old_hash,h,groups["passive"],prior,ml.select_device(args.device))


if __name__=="__main__":
    main()
