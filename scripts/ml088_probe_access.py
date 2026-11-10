
"""ML-088 (revised, pre-freeze): availability vs output-readout diagnostic.

Fresh paired models, strictly familiar-grammar-trained linear readouts.
Compare final-residual probe to BOTH native head and a supervised
recalibration of native name logits (same ridge loss and fit data).
Explore earlier layers and shuffled-label negative control.

Workflow from repository root:
  python3 scripts/ml088_probe_access.py --phase plan
  python3 scripts/ml088_probe_access.py --phase preflight
  # Commit script and frozen_protocol.json before training.
  python3 scripts/ml088_probe_access.py --phase train --until 3200
  python3 scripts/ml088_probe_access.py --phase summarize
  python3 scripts/ml088_probe_access.py --phase train --until 12800
  python3 scripts/ml088_probe_access.py --phase summarize
  python3 scripts/ml088_probe_access.py --phase calibrate
  # Commit calibration files BEFORE opening protected.
  python3 scripts/ml088_probe_access.py --phase evaluate --confirm-protected

A positive probe result demonstrates supervised recoverability,
NOT native routing necessity or causal mediation.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

import ml080_learning_curve as ml
import ml081_layerwise_event_role as m81
import ml084_prospective_geometry as m84

NAMES = tuple(sorted((
    "Adrian", "Beatrix", "Cassian", "Daphne", "Emmett", "Flora",
    "Griffin", "Helena", "Iris", "Jonas", "Kendra", "Lucian",
    "Marina", "Nico", "Odette", "Pavel",
)))
SEEDS = (701, 709, 719, 727, 733, 739, 743, 751)
MATCHING_SEED = 88088
LAYERS = (0, 1, 2)
LAMBDA = 1.0
BATCH = 128
ROOT = Path("results/ml088_fresh_models")
PROTOCOL = Path("reports/ml088/frozen_protocol.json")
PROBES = Path("reports/ml088/calibration_probes.pt")
CAL_REPORT = Path("reports/ml088/calibration_manifest.json")
PROTECTED_REPORT = Path("reports/ml088/protected_evaluation.json")
PROTECTED_LOCK = Path("reports/ml088/protected_opened_once.json")


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode()
    ).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    """Configure the new generator exactly once."""
    assert NAMES == tuple(sorted(NAMES))
    assert len(set(NAMES)) == 16
    assert not set(NAMES) & set(ml.NAMES)
    assert not set(NAMES) & set(m84.NEW_NAMES)

    m84.NEW_NAMES = NAMES
    m84.NEW_SEEDS = SEEDS
    m84.MATCHING_SEED = MATCHING_SEED

    vocab, train, cal_rounds, held, base, cal, dose, _ = (
        m84.prepare()
    )

    fit_pairs = {
        p for matching in cal_rounds[:2] for p in matching
    }
    val_pairs = {
        p for matching in cal_rounds[2:] for p in matching
    }

    assert len(fit_pairs) == 16
    assert len(val_pairs) == 8
    assert not fit_pairs & val_pairs

    fit = [c for c in cal if c.target_pair in fit_pairs]
    valid = [c for c in cal if c.target_pair in val_pairs]

    assert len(fit) == 4608
    assert len(valid) == 2304
    assert not (
        set(c.target_pair for c in fit)
        & set(c.target_pair for c in valid)
    )

    return (
        vocab, train, cal_rounds, held,
        base, cal, dose, fit, valid,
    )


def specification(vocab, train, cal_rounds, held, fit, valid):
    return {
        "experiment": "ML-088",
        "revision": "pre-freeze-strengthened-readout-controls",
        "script_sha256": sha_file(__file__),
        "ml080_sha256": sha_file(ml.__file__),
        "ml081_sha256": sha_file(m81.__file__),
        "ml084_sha256": sha_file(m84.__file__),
        "names": NAMES,
        "seeds": SEEDS,
        "arms": ml.ARMS,
        "matching_seed": MATCHING_SEED,
        "vocab_sha256": digest(vocab),
        "splits": {
            "training_pairs": train,
            "calibration_pairs": cal_rounds,
            "protected_pair_ids": held,
        },
        "training": (
            "ML-080 frozen architecture and optimization; "
            "12800 steps; 64 distinct repeated exposure "
            "questions per arm"
        ),
        "model_gate": (
            "at 12800 >=6/8 each arm familiar calibration "
            "choice>=.95 and correct-both>=.90"
        ),
        "probes": {
            "fit_cases": len(fit),
            "validation_cases": len(valid),
            "fit_pairs": 16,
            "validation_pairs": 8,
            "fit_prompt_hash": digest([
                ml.prompt(c) for c in fit
            ]),
            "validation_prompt_hash": digest([
                ml.prompt(c) for c in valid
            ]),
            "sites": (
                "answer-token post-block residuals at L1,L2,L3 "
                "after fixed final LayerNorm"
            ),
            "labels": (
                "one of 16 participant names, "
                "fit/calibration only"
            ),
            "objective": (
                "same per-feature-standardized, "
                "bias-inclusive ridge one-hot regression "
                "for all readouts"
            ),
            "ridge_lambda": LAMBDA,
            "output_head_control": (
                "ridge rehead trained on frozen native "
                "16 name logits, exact same fitting "
                "labels/objective"
            ),
            "negative_control": (
                "final residual probe trained on "
                "deterministically randomized "
                "within-candidate labels"
            ),
            "hyperparameter_selection": "none",
            "validation_gate": (
                "at least 6/8 each arm with BOTH L3 probe "
                "and logit rehead >=.90 4-choice on disjoint "
                "familiar calibration; no seed exclusion"
            ),
            "no_feature_labels_at_test": True,
        },
        "protected_target": (
            "1536 passive-second construction questions "
            "using protected pairs and exposure-unseen verbs"
        ),
        "primary": {
            "arm": "familiar",
            "seed_statistic": (
                "L3 4-choice probe accuracy minus "
                "max(native head 4-choice accuracy, "
                "logit-rehead 4-choice accuracy) "
                "on protected questions"
            ),
            "success": (
                "mean difference >+.15, positive in >=6/8 "
                "seed pairs, paired-seed bootstrap "
                "95% lower bound >0"
            ),
            "note": (
                "best baseline selected by a frozen "
                "per-seed accuracy maximum; conservative "
                "comparison, not hypothesis tuning"
            ),
        },
        "secondary": [
            "L1/L2 decodability",
            "native vs rehead vs probe for both arms",
            "native/external 16-name accuracy",
            "label-shuffle negative control",
            "paired exposure advantage after probing",
            "validation failure or low target generalization",
        ],
        "safety": (
            "protected examples ONLY generated after "
            "competence gate, frozen calibration readouts "
            "and explicit unlock; output+one-time lock "
            "refuse rerun"
        ),
        "scope": (
            "diagnostic supervised recoverability only; "
            "same synthetic grammar/model family; "
            "does NOT identify native routing "
            "or prove causal mediation"
        ),
    }


def frozen(doc, create):
    data = json.loads(json.dumps({
        **doc,
        "sha256": digest(doc),
    }))

    if create and not PROTOCOL.exists():
        PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
        PROTOCOL.write_text(
            json.dumps(data, indent=2) + "\n"
        )

    if not PROTOCOL.exists():
        raise FileNotFoundError("Run --phase plan first")

    assert json.loads(PROTOCOL.read_text()) == data, (
        "Frozen script/dependencies/protocol mismatch"
    )

    return data["sha256"]


def model_gate(h):
    scores = ml.all_metrics(
        ROOT, h, 12800, complete=True
    )

    counts = {
        arm: sum(
            scores[(seed, arm)]["calibration"][
                "choice_accuracy"
            ] >= .95
            and scores[(seed, arm)]["calibration"][
                "correct_both_rate"
            ] >= .90
            for seed in SEEDS
        )
        for arm in ml.ARMS
    }

    return all(n >= 6 for n in counts.values()), counts


def name_lookup():
    ids = torch.tensor(
        [ml.VOCAB[name] for name in NAMES],
        dtype=torch.long,
    )

    lookup = torch.full(
        (len(ml.VOCAB),),
        -1,
        dtype=torch.long,
    )

    lookup[ids] = torch.arange(len(NAMES))
    return ids, lookup


@torch.inference_mode()
def extract(model, cases, device, name_ids):
    trace_parts = [[] for _ in LAYERS]
    native_parts = []
    name_logit_parts = []
    gold_parts = []
    option_parts = []

    for start in range(0, len(cases), BATCH):
        ids, ro, labels, choices = ml.encode(
            cases[start:start+BATCH]
        )

        logits, trace = m81.forward_trace(
            model,
            ids.to(device),
            ro.to(device),
        )

        native_parts.append(
            logits.cpu().float()
        )
        name_logit_parts.append(
            logits[:, name_ids.to(device)].cpu().float()
        )
        gold_parts.append(labels)
        option_parts.append(choices)

        for j, layer in enumerate(LAYERS):
            trace_parts[j].append(
                model.norm(
                    trace[layer]["residual"]
                ).cpu().float()
            )

    return {
        "states": [
            torch.cat(x)
            for x in trace_parts
        ],
        "name_logits": torch.cat(name_logit_parts),
        "native_logits": torch.cat(native_parts),
        "gold": torch.cat(gold_parts),
        "options": torch.cat(option_parts),
    }


def ridge_fit(x, labels, lookup):
    """Same objective and regularization for all readouts."""
    x = x.double()

    target = lookup[labels]
    assert bool((target >= 0).all())

    y = F.one_hot(
        target,
        num_classes=16,
    ).double()

    center = x.mean(dim=0)
    scale = x.std(
        dim=0,
        unbiased=False,
    ).clamp_min(1e-4)

    z = (x - center) / scale
    bias = y.mean(dim=0)

    gram = (
        z.T @ z
        + LAMBDA * torch.eye(
            z.shape[1],
            dtype=torch.float64,
        )
    )

    weight = torch.linalg.solve(
        gram,
        z.T @ (y - bias),
    )

    return {
        "mean": center.float(),
        "scale": scale.float(),
        "weight": weight.float(),
        "bias": bias.float(),
    }


def ridge_predict(x, readout):
    return (
        (x - readout["mean"])
        / readout["scale"]
    ) @ readout["weight"] + readout["bias"]


def train_probes(extracted, lookup, negative_seed):
    reads = {
        f"L{j+1}": ridge_fit(
            state,
            extracted["gold"],
            lookup,
        )
        for j, state in enumerate(
            extracted["states"]
        )
    }

    reads["logit_rehead"] = ridge_fit(
        extracted["name_logits"],
        extracted["gold"],
        lookup,
    )

    rng = torch.Generator(
        device="cpu"
    ).manual_seed(negative_seed)

    choice_cols = torch.randint(
        0,
        4,
        (len(extracted["gold"]),),
        generator=rng,
    )

    shuffled_gold = extracted["options"][
        torch.arange(len(choice_cols)),
        choice_cols,
    ]

    reads["shuffled_L3"] = ridge_fit(
        extracted["states"][2],
        shuffled_gold,
        lookup,
    )

    return reads


def decode_candidate(scores, cls_choices):
    return cls_choices.gather(
        1,
        scores.gather(
            1,
            cls_choices,
        ).argmax(
            1,
            keepdim=True,
        ),
    ).flatten()


def accuracy(pred, gold):
    return float(
        (pred == gold).float().mean().item()
    )


def measure(extracted, probes, lookup, name_ids):
    gold_id = extracted["gold"]
    choices_id = extracted["options"]

    gold = lookup[gold_id]
    choices = lookup[choices_id]

    assert bool((gold >= 0).all())
    assert bool((choices >= 0).all())

    native_full = extracted["native_logits"]

    n4 = decode_candidate(
        native_full,
        choices_id,
    )

    native_16 = (
        extracted["name_logits"].argmax(1)
    )

    metrics = {
        "native_choice": accuracy(n4, gold_id),
        "native_names16": accuracy(
            native_16, gold
        ),
        "native_full_vocab": accuracy(
            native_full.argmax(1),
            gold_id,
        ),
    }

    sources = {
        f"L{j+1}": state
        for j, state in enumerate(
            extracted["states"]
        )
    }

    sources["logit_rehead"] = (
        extracted["name_logits"]
    )
    sources["shuffled_L3"] = (
        extracted["states"][2]
    )

    for k, x in sources.items():
        scores = ridge_predict(
            x,
            probes[k],
        )

        pred4 = decode_candidate(
            scores,
            choices,
        )

        metrics[f"{k}_choice"] = accuracy(
            pred4,
            gold,
        )
        metrics[f"{k}_names16"] = accuracy(
            scores.argmax(1),
            gold,
        )

    return metrics


def bootstrap(xs):
    rng = random.Random(880883)
    n = len(xs)

    dist = sorted(
        mean(
            xs[rng.randrange(n)]
            for _ in range(n)
        )
        for _ in range(20000)
    )

    return [dist[500], dist[19500]]


def train_gate_and_calibrate(h, fit, valid, device):
    passed, counts = model_gate(h)

    if not passed:
        raise PermissionError(
            f"Frozen model competence gate FAILED: {counts}"
        )

    if PROBES.exists() or CAL_REPORT.exists():
        raise FileExistsError(
            "Calibration already exists; refusing overwrite"
        )

    name_ids, lookup = name_lookup()
    bank, scores = {}, {}

    for seed in SEEDS:
        for arm in ml.ARMS:
            name = f"{seed}/{arm}"

            model = ml.load_checkpoint(
                ROOT / f"seed_{seed}" / arm
                / "model_step_12800.pt",
                seed,
                arm,
                h,
                device,
            )

            fitted = extract(
                model,
                fit,
                device,
                name_ids,
            )

            validated = extract(
                model,
                valid,
                device,
                name_ids,
            )

            probes = train_probes(
                fitted,
                lookup,
                seed * 2
                + (arm == "target")
                + 880880,
            )

            vals = measure(
                validated,
                probes,
                lookup,
                name_ids,
            )

            bank[name] = probes
            scores[name] = vals

            print(
                f"calibrate {name:14s} "
                f"native={vals['native_choice']:.3f} "
                f"L3={vals['L3_choice']:.3f} "
                f"logits={vals['logit_rehead_choice']:.3f} "
                f"shuffled={vals['shuffled_L3_choice']:.3f}",
                flush=True,
            )

    pass_counts = {
        arm: sum(
            scores[f"{seed}/{arm}"]["L3_choice"] >= .90
            and scores[f"{seed}/{arm}"][
                "logit_rehead_choice"
            ] >= .90
            for seed in SEEDS
        )
        for arm in ml.ARMS
    }

    doc = {
        "experiment": "ML-088",
        "protocol_sha256": h,
        "competence_gate": counts,
        "readout_gate": pass_counts,
        "readout_gate_passed": all(
            n >= 6 for n in pass_counts.values()
        ),
        "validation_only": True,
        "validation_results": scores,
    }

    PROBES.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(bank, PROBES)

    doc["probes_file_sha256"] = (
        sha_file(PROBES)
    )
    doc["sha256"] = digest(doc)

    CAL_REPORT.write_text(
        json.dumps(doc, indent=2) + "\n"
    )

    print(
        "CALIBRATION GATE",
        "PASS" if doc["readout_gate_passed"] else "FAIL",
        pass_counts,
    )

    print("Saved", PROBES, "and", CAL_REPORT)
    print(
        "Commit both unchanged calibration artifacts "
        "BEFORE protected evaluation."
    )


def load_calibration(h):
    doc = json.loads(
        CAL_REPORT.read_text()
    )

    assert doc["sha256"] == digest({
        k: v
        for k, v in doc.items()
        if k != "sha256"
    })

    assert doc["protocol_sha256"] == h
    assert doc["readout_gate_passed"]
    assert sha_file(PROBES) == doc["probes_file_sha256"]

    bank = torch.load(
        PROBES,
        weights_only=True,
        map_location="cpu",
    )

    assert set(bank) == {
        f"{s}/{a}"
        for s in SEEDS
        for a in ml.ARMS
    }

    return bank, doc


def evaluate(h, held, device, confirmed):
    if not confirmed:
        raise PermissionError(
            "Need --confirm-protected"
        )

    if PROTECTED_REPORT.exists() or PROTECTED_LOCK.exists():
        raise FileExistsError(
            "ML-088 protected already opened or locked; "
            "do not rerun"
        )

    passed, counts = model_gate(h)

    if not passed:
        raise PermissionError(
            f"Frozen model competence gate FAILED: {counts}"
        )

    bank, doc = load_calibration(h)

    PROTECTED_LOCK.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROTECTED_LOCK.write_text(
        json.dumps({
            "experiment": "ML-088",
            "protocol_sha256": h,
            "calibration_manifest_sha256": doc["sha256"],
            "protected_opened_once": True,
            "note": (
                "Created before protected case generation; "
                "never clear to repeat an evaluation."
            ),
        }, indent=2) + "\n"
    )

    # Protected case generation occurs after all guards.
    protected = ml.protected_target_cases(held)

    assert len(protected) == 1536

    names, lookup = name_lookup()
    rows = []

    for seed in SEEDS:
        for arm in ml.ARMS:
            model = ml.load_checkpoint(
                ROOT / f"seed_{seed}" / arm
                / "model_step_12800.pt",
                seed,
                arm,
                h,
                device,
            )

            inp = extract(
                model,
                protected,
                device,
                names,
            )

            metrics = measure(
                inp,
                bank[f"{seed}/{arm}"],
                lookup,
                names,
            )

            rows.append({
                "seed": seed,
                "arm": arm,
                "metrics": metrics,
            })

            print(
                f"seed={seed} arm={arm:8s} "
                f"native={metrics['native_choice']:.3f} "
                f"logits={metrics['logit_rehead_choice']:.3f} "
                f"L3={metrics['L3_choice']:.3f} "
                f"L1={metrics['L1_choice']:.3f}",
                flush=True,
            )

    paired = {
        (r["seed"], r["arm"]): r["metrics"]
        for r in rows
    }

    primary = []
    native_gap = []
    probe_gap = []

    for seed in SEEDS:
        familiar = paired[seed, "familiar"]
        target = paired[seed, "target"]

        baseline = max(
            familiar["native_choice"],
            familiar["logit_rehead_choice"],
        )

        primary.append(
            familiar["L3_choice"] - baseline
        )

        native_gap.append(
            target["native_choice"]
            - familiar["native_choice"]
        )

        probe_gap.append(
            target["L3_choice"]
            - familiar["L3_choice"]
        )

    ci = bootstrap(primary)

    summary = {
        "primary_per_seed": {
            str(seed): v
            for seed, v in zip(SEEDS, primary)
        },
        "primary_mean": mean(primary),
        "primary_95pct_seed_bootstrap_ci": ci,
        "primary_positive_seeds": sum(
            v > 0 for v in primary
        ),
        "primary_success_rule_met": (
            mean(primary) > .15
            and sum(v > 0 for v in primary) >= 6
            and ci[0] > 0
        ),
        "native_target_minus_familiar_mean": mean(
            native_gap
        ),
        "L3_probe_target_minus_familiar_mean": mean(
            probe_gap
        ),
        "reduction_of_exposure_gap_mean": mean([
            n - p
            for n, p in zip(native_gap, probe_gap)
        ]),
        "n_seed_pairs": len(SEEDS),
        "warning": (
            "Only supervised recoverability; external "
            "probe may provide new readout supervision. "
            "Not a native-routing or causal-mediation test."
        ),
    }

    report = {
        "experiment": "ML-088",
        "protocol_sha256": h,
        "calibration_sha256": doc["sha256"],
        "protected_opened_once": True,
        "runs": rows,
        "summary": summary,
    }

    PROTECTED_REPORT.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print(
        "PROTECTED SUMMARY\n"
        + json.dumps(summary, indent=2)
    )
    print("Saved", PROTECTED_REPORT)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--phase",
        required=True,
        choices=(
            "plan", "preflight", "train",
            "summarize", "calibrate", "evaluate",
        ),
    )

    ap.add_argument(
        "--until",
        choices=(3200, 12800),
        type=int,
        default=3200,
    )

    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )

    ap.add_argument("--save-every", type=int, default=400)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--arm", choices=ml.ARMS)
    ap.add_argument("--confirm-protected", action="store_true")

    args = ap.parse_args()

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    (
        vocab, tr, ca, held,
        base, cal, dose, fit, valid,
    ) = prepare()

    h = frozen(
        specification(
            vocab, tr, ca, held, fit, valid
        ),
        args.phase == "plan",
    )

    if args.phase == "plan":
        print(
            "ML-088 plan frozen:",
            PROTOCOL,
            "protocol",
            h[:12],
        )
        return

    if args.phase == "preflight":
        torch.manual_seed(88088)
        model = ml.Model().eval()
        name_ids, lookup = name_lookup()

        data = extract(
            model,
            valid[:96],
            "cpu",
            name_ids,
        )

        ids, ro, _, _ = ml.encode(valid[:96])

        with torch.inference_mode():
            direct = model(ids, ro).float()

        parity = float(
            (data["native_logits"] - direct).abs().max()
        )

        assert parity < 1e-3, parity

        probes = train_probes(
            data,
            lookup,
            880880,
        )

        outcome = measure(
            data,
            probes,
            lookup,
            name_ids,
        )

        assert all(
            0 <= v <= 1
            for v in outcome.values()
        )

        assert all(
            torch.isfinite(p["weight"]).all()
            for p in probes.values()
        )

        assert data["states"][2].shape == (
            96,
            ml.WIDTH,
        )

        print(
            "PREFLIGHT PASS: 16 names, 16/8 disjoint "
            f"calibration pairs; forward parity={parity:.2e}"
        )

        print(
            "Preflight includes ridge, native-logit rehead, "
            "label-shuffle control; no trained checkpoints "
            "or protected examples."
        )
        return

    if args.phase == "train":
        if args.save_every <= 0:
            raise ValueError(
                "save-every must be >0"
            )

        device = ml.select_device(args.device)

        for seed in (
            SEEDS
            if args.seed is None
            else (args.seed,)
        ):
            for arm in (
                ml.ARMS
                if args.arm is None
                else (args.arm,)
            ):
                ml.train_one(
                    seed, arm, args.until,
                    args.save_every,
                    ROOT, h, base, cal, dose,
                    device,
                )

        print(
            "Training milestone completed; "
            "protected remains sealed."
        )
        return

    if args.phase == "summarize":
        ml.summarize(ROOT, h)
        return

    if args.phase == "calibrate":
        train_gate_and_calibrate(
            h,
            fit,
            valid,
            ml.select_device(args.device),
        )
        return

    evaluate(
        h,
        held,
        ml.select_device(args.device),
        args.confirm_protected,
    )


if __name__ == "__main__":
    main()
