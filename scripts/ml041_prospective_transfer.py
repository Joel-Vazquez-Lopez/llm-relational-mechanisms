
"""ML-041: prospective transfer across newly trained scratch Transformers.

Phases:
  audit -> train -> freeze -> test

Training pairs determine the shared interaction direction.
Calibration pairs determine frozen numerical predictions.
Protected pairs are evaluated only after freezing.

The vocabulary and sentence templates appeared in earlier experiments.
This is a new split for new models, not a new linguistic task family.
"""

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from ml037_scratch_transformer_pilot import (
    ScratchTransformer,
    Example,
    NAMES,
    VERBS,
    VOICES,
    ROLES,
    VOCAB,
    encode_batch,
)

SEEDS = (101, 103, 107, 109, 113, 127)
SPLIT_SEED = 20261101

STEPS = 300
BATCH = 64
LR = 0.002

BUDGET = 0.20
CONTROL_COUNT = 4

RESULTS = Path("results")
SIGNS = (1.0, -1.0, -1.0, 1.0)


def make_split():
    pairs = [
        (NAMES[i], NAMES[j])
        for i in range(len(NAMES))
        for j in range(i + 1, len(NAMES))
    ]

    random.Random(SPLIT_SEED).shuffle(pairs)

    out = {
        "train": tuple(pairs[:14]),
        "calibration": tuple(pairs[14:21]),
        "protected": tuple(pairs[21:]),
    }

    assert [len(out[k]) for k in out] == [14, 7, 7]

    all_pairs = [
        p for group in out.values() for p in group
    ]
    assert len(set(all_pairs)) == 28

    seen_names = {
        name
        for pair in out["train"]
        for name in pair
    }

    assert set(NAMES) == seen_names

    return out


SPLIT = make_split()

SPEC = {
    "experiment": "ML-041",
    "hypothesis": (
        "Training-derived shared interaction direction "
        "predicts causal interference on new name pairs"
    ),
    "split_seed": SPLIT_SEED,
    "split": {
        k: [list(p) for p in v]
        for k, v in SPLIT.items()
    },
    "seeds": list(SEEDS),
    "steps": STEPS,
    "batch": BATCH,
    "lr": LR,
    "width": 96,
    "heads": 4,
    "layers": 2,
    "budget": BUDGET,
    "control_count": CONTROL_COUNT,
    "prediction": (
        "Protected control-minus-donor margin effect "
        "equals calibration control-minus-donor effect"
    ),
    "competence_gate": (
        "Calibration pairwise accuracy >= 0.95; "
        "all seeds must be reported"
    ),
    "primary_score": (
        "Prediction MAE versus zero-effect baseline"
    ),
}

DESIGN_HASH = hashlib.sha256(
    json.dumps(SPEC, sort_keys=True).encode()
).hexdigest()


def quartets(pairs):
    return [
        (
            pair,
            verb,
            direction,
            [
                Example(
                    pair,
                    verb,
                    direction,
                    voice,
                    role,
                )
                for voice, role in itertools.product(
                    VOICES, ROLES
                )
            ],
        )
        for pair in pairs
        for verb in VERBS
        for direction in (0, 1)
    ]


def all_examples(pairs):
    return [
        example
        for _, _, _, q in quartets(pairs)
        for example in q
    ]


def sha_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def model_path(seed):
    return RESULTS / f"ml041_scratch_seed{seed}.pt"


def load_model(seed, device, expected_digest=None):
    path = model_path(seed)
    digest = sha_file(path)

    if (
        expected_digest is not None
        and digest != expected_digest
    ):
        raise RuntimeError(
            f"Checkpoint changed after freeze: {path}"
        )

    saved = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if (
        saved["design_hash"] != DESIGN_HASH
        or saved["seed"] != seed
    ):
        raise RuntimeError(
            f"Checkpoint design mismatch: {path}"
        )

    if saved["vocab"] != VOCAB:
        raise RuntimeError("Vocabulary mismatch")

    model = ScratchTransformer().to(device)

    model.load_state_dict(
        saved["state_dict"],
        strict=True,
    )

    return model.eval(), digest


@torch.no_grad()
def pairwise_accuracy(model, pairs, device):
    examples = all_examples(pairs)
    correct = 0

    for i in range(0, len(examples), 128):
        batch = examples[i:i + 128]

        ids, positions, labels, choices = (
            encode_batch(batch)
        )

        logits = model(
            ids.to(device),
            positions.to(device),
        ).cpu()

        selected = logits.gather(1, choices)

        predicted = choices.gather(
            1,
            selected.argmax(
                1, keepdim=True
            ),
        ).squeeze(1)

        correct += int(
            (predicted == labels).sum()
        )

    return correct / len(examples)


def train(seed, device):
    path = model_path(seed)

    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite {path}"
        )

    torch.manual_seed(seed)

    model = ScratchTransformer().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01,
    )

    ids, positions, labels, _ = encode_batch(
        all_examples(SPLIT["train"])
    )

    for step in range(1, STEPS + 1):
        model.train()

        indices = torch.randint(
            len(labels),
            (BATCH,),
        )

        logits = model(
            ids[indices].to(device),
            positions[indices].to(device),
        )

        loss = F.cross_entropy(
            logits,
            labels[indices].to(device),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()

    training_acc = pairwise_accuracy(
        model, SPLIT["train"], device
    )

    calibration_acc = pairwise_accuracy(
        model, SPLIT["calibration"], device
    )

    checkpoint = {
        "seed": seed,
        "design_hash": DESIGN_HASH,
        "state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "vocab": VOCAB,
        "train_accuracy": training_acc,
        "calibration_accuracy": calibration_acc,
    }

    RESULTS.mkdir(exist_ok=True)

    torch.save(checkpoint, path)

    print(
        f"seed={seed}: "
        f"train={training_acc:.3f} "
        f"calibration={calibration_acc:.3f} "
        f"checkpoint={path}"
    )


@torch.no_grad()
def donor_from_training(model, device):
    vectors = []

    for _, _, _, q in quartets(SPLIT["train"]):
        ids, positions, _, _ = encode_batch(q)

        _, states = model(
            ids.to(device),
            positions.to(device),
            return_states=True,
        )

        # Block-1 output.
        # Quartet order:
        # active-agent, active-patient,
        # passive-agent, passive-patient.
        h = states[1]

        delta = (
            h[3] - h[2] - h[1] + h[0]
        )

        norm = delta.norm()

        if float(norm) < 1e-8:
            raise RuntimeError(
                "Zero training interaction"
            )

        vectors.append(delta / norm)

    stacked = torch.stack(vectors)

    mean_vector = stacked.mean(dim=0)

    coherence = float(
        mean_vector.norm()
    )

    if coherence < 1e-6:
        raise RuntimeError(
            "Training donor direction ill-defined"
        )

    donor = mean_vector / mean_vector.norm()

    return donor, coherence


def control_directions(donor, seed):
    vectors = []

    for j in range(CONTROL_COUNT):
        payload = f"ML041:{seed}:{j}".encode()

        number = int.from_bytes(
            hashlib.sha256(payload).digest()[:8],
            "big",
        ) % (2**63 - 1)

        generator = torch.Generator(
            device="cpu"
        ).manual_seed(number)

        bits = torch.randint(
            0,
            2,
            (donor.numel(),),
            generator=generator,
        )

        signs = (2 * bits - 1).to(
            device=donor.device,
            dtype=donor.dtype,
        )

        vectors.append(donor * signs)

    return vectors


@torch.no_grad()
def transfer(model, pairs, donor, seed, device):
    controls = control_directions(
        donor, seed
    )

    directions = [donor, *controls]

    signs = torch.tensor(
        SIGNS,
        device=device,
        dtype=donor.dtype,
    )[:, None]

    records = []

    for pair, verb, direction, q in quartets(pairs):
        ids, readout, labels, choices = (
            encode_batch(q)
        )

        ids, readout, labels, choices = (
            x.to(device)
            for x in (ids, readout, labels, choices)
        )

        original_logits, states = model(
            ids,
            readout,
            return_states=True,
        )

        h = states[1]
        hnorm = h.norm(dim=-1).mean()

        replacements = [h]

        for d in directions:
            replacements.append(
                h - signs * BUDGET * hnorm * d
            )

        n = len(replacements)

        patch_vectors = torch.stack(
            replacements
        ).reshape(
            n * 4,
            h.shape[-1],
        )

        big_ids = ids.repeat(n, 1)
        big_pos = readout.repeat(n)

        def patch(_module, _inputs, output):
            result = output.clone()

            batch_indices = torch.arange(
                n * 4,
                device=device,
            )

            result[
                batch_indices,
                big_pos,
            ] = patch_vectors.to(result.dtype)

            return result

        handle = (
            model.blocks[0]
            .register_forward_hook(patch)
        )

        try:
            patch_logits = model(
                big_ids, big_pos
            ).reshape(n, 4, -1)
        finally:
            handle.remove()

        other = torch.where(
            choices[:, 0] == labels,
            choices[:, 1],
            choices[:, 0],
        )

        targets = labels[
            None, :, None
        ].expand(n, -1, 1)

        alternatives = other[
            None, :, None
        ].expand(n, -1, 1)

        margins = (
            patch_logits.gather(2, targets)
            - patch_logits.gather(
                2, alternatives
            )
        ).squeeze(-1)

        original_margin = (
            original_logits.gather(
                1, labels[:, None]
            ).squeeze(1)
            - original_logits.gather(
                1, other[:, None]
            ).squeeze(1)
        )

        self_patch_difference = float(
            (
                margins[0] - original_margin
            ).abs().max()
        )

        if self_patch_difference > 0.05:
            raise RuntimeError(
                "Self-patch check failed"
            )

        control_mean = margins[2:].mean(dim=0)
        donor_margin = margins[1]

        # Positive harm means the donor direction
        # is more damaging than sign-flipped controls.
        harm = float(
            (
                control_mean - donor_margin
            ).mean()
        )

        records.append({
            "pair": list(pair),
            "verb": verb,
            "direction": direction,
            "original_accuracy": float(
                (original_margin > 0)
                .float().mean()
            ),
            "donor_accuracy": float(
                (donor_margin > 0)
                .float().mean()
            ),
            "control_accuracy": float(
                (margins[2:] > 0)
                .float().mean()
            ),
            "harm": harm,
            "donor_margin_change": float(
                (
                    donor_margin - original_margin
                ).mean()
            ),
        })

    return {
        "mean_harm": mean(
            r["harm"] for r in records
        ),
        "original_accuracy": mean(
            r["original_accuracy"]
            for r in records
        ),
        "donor_accuracy": mean(
            r["donor_accuracy"]
            for r in records
        ),
        "control_accuracy": mean(
            r["control_accuracy"]
            for r in records
        ),
        "records": records,
    }


def freeze(device):
    path = RESULTS / "ml041_frozen_predictions.json"

    if path.exists():
        raise RuntimeError(
            "Predictions already frozen; refusing to overwrite"
        )

    manifest = {
        "design_hash": DESIGN_HASH,
        "spec": SPEC,
        "seeds": {},
        "complete": True,
    }

    for seed in SEEDS:
        model, digest = load_model(
            seed, device
        )

        calibration_acc = pairwise_accuracy(
            model,
            SPLIT["calibration"],
            device,
        )

        donor, coherence = donor_from_training(
            model, device
        )

        calibration = transfer(
            model,
            SPLIT["calibration"],
            donor,
            seed,
            device,
        )

        prediction = calibration["mean_harm"]

        manifest["seeds"][str(seed)] = {
            "checkpoint_sha256": digest,
            "calibration_pairwise_accuracy": (
                calibration_acc
            ),
            "eligible": (
                calibration_acc >= 0.95
            ),
            "train_direction_coherence": (
                coherence
            ),
            "donor_unit_vector": (
                donor.detach().cpu().tolist()
            ),
            "prediction_protected_harm": (
                prediction
            ),
            "calibration": calibration,
        }

        print(
            f"seed={seed}: "
            f"cal_acc={calibration_acc:.3f} "
            f"coherence={coherence:.3f} "
            f"frozen_harm={prediction:+.3f}"
        )

    RESULTS.mkdir(exist_ok=True)

    path.write_text(
        json.dumps(manifest, indent=2)
    )

    print(
        "FROZEN PREDICTIONS:",
        path,
        "sha256:",
        sha_file(path),
    )

    print(
        "No protected outcomes have been used."
    )


def test(device):
    predictions_path = (
        RESULTS / "ml041_frozen_predictions.json"
    )

    if not predictions_path.exists():
        raise RuntimeError(
            "Run --phase freeze before protected test"
        )

    frozen = json.loads(
        predictions_path.read_text()
    )

    expected_seeds = {
        str(x) for x in SEEDS
    }

    if (
        frozen["design_hash"] != DESIGN_HASH
        or not frozen["complete"]
        or set(frozen["seeds"]) != expected_seeds
    ):
        raise RuntimeError(
            "Frozen manifest mismatch"
        )

    path = (
        RESULTS / "ml041_protected_results.json"
    )

    if path.exists():
        raise RuntimeError(
            "Refusing to overwrite protected results"
        )

    rows = []

    for seed in SEEDS:
        details = frozen["seeds"][str(seed)]

        model, _ = load_model(
            seed,
            device,
            details["checkpoint_sha256"],
        )

        donor = torch.tensor(
            details["donor_unit_vector"],
            device=device,
            dtype=torch.float32,
        )

        if abs(
            float(donor.norm()) - 1.0
        ) > 1e-4:
            raise RuntimeError(
                "Frozen donor norm invalid"
            )

        measured = transfer(
            model,
            SPLIT["protected"],
            donor,
            seed,
            device,
        )

        prediction = details[
            "prediction_protected_harm"
        ]

        observed = measured["mean_harm"]

        rows.append({
            "seed": seed,
            "eligible": details["eligible"],
            "frozen_predicted_harm": prediction,
            "protected_harm": observed,
            "absolute_error": abs(
                prediction - observed
            ),
            "zero_baseline_absolute_error": abs(
                observed
            ),
            "protected": measured,
        })

        print(
            f"seed={seed} "
            f"eligible={details['eligible']} "
            f"predicted_harm={prediction:+.3f} "
            f"observed={observed:+.3f} "
            f"protected_acc="
            f"{measured['original_accuracy']:.3f}"
        )

    eligible = [
        r for r in rows
        if r["eligible"]
    ]

    print(
        f"Eligible seeds: "
        f"{len(eligible)}/{len(SEEDS)}"
    )

    if eligible:
        predicted_mae = mean(
            r["absolute_error"]
            for r in eligible
        )

        zero_mae = mean(
            r["zero_baseline_absolute_error"]
            for r in eligible
        )

        sign_matches = sum(
            (
                r["frozen_predicted_harm"] > 0
            ) == (
                r["protected_harm"] > 0
            )
            for r in eligible
        )

        print(
            f"Predicted MAE: {predicted_mae:.3f}"
        )

        print(
            f"Zero-effect baseline MAE: "
            f"{zero_mae:.3f}"
        )

        print(
            f"Sign matches: "
            f"{sign_matches}/{len(eligible)}"
        )

    path.write_text(
        json.dumps({
            "experiment": "ML-041",
            "complete": True,
            "design_hash": DESIGN_HASH,
            "frozen_manifest_sha256": (
                sha_file(predictions_path)
            ),
            "records": rows,
        }, indent=2)
    )

    print("Saved protected results:", path)

    print(
        "Interpret at the model-seed level, "
        "not as independent answers."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "audit", "train", "freeze", "test"
        ),
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
    )

    args = parser.parse_args()

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    if args.device == "auto":
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = args.device

    if (
        device == "mps"
        and not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS unavailable")

    print(
        f"ML-041 | {args.phase} "
        f"| device={device} "
        f"| design={DESIGN_HASH}"
    )

    if args.phase == "audit":
        print(
            "Pairs:",
            {
                k: len(v)
                for k, v in SPLIT.items()
            },
        )

        print(
            "Examples:",
            [
                len(all_examples(SPLIT[k]))
                for k in SPLIT
            ],
        )

        print("Seeds:", SEEDS)

        print(
            "Primary prediction: validation harm "
            "predicts protected harm."
        )

    elif args.phase == "train":
        for seed in SEEDS:
            train(seed, device)

    elif args.phase == "freeze":
        freeze(device)

    else:
        test(device)


if __name__ == "__main__":
    main()
