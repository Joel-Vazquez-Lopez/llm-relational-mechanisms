
"""ML-060: Development of causal dependencies across Transformer layers.

Read-only analysis of ML-057 checkpoints.

Interventions:
  - Zero attention/MLP output at the answer readout position.
  - Add random perturbations with the same L2 norm as the removed output.
  - Zero output at a future position (negative control).

This measures causal sensitivity, not semantic-role circuit identity.
Only ML-057 calibration examples are evaluated.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ml037_scratch_transformer_pilot import (
    ScratchTransformer,
    encode_batch,
    VOCAB,
)
from ml057_developmental_checkpoints import make_split, case_key


DEFAULT_STEPS = (0, 50, 75, 100, 150, 200, 300)


def check_manifest(root, calibration):
    manifest = json.loads(
        (root / "manifest.json").read_text()
    )
    digest = manifest.pop("sha256")

    actual = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()

    assert digest == actual, "Manifest hash mismatch"

    assert sorted(
        case_key(e) for e in calibration
    ) == sorted(manifest["calibration_case_keys"])

    return digest


def module_at(model, layer, part):
    block = model.blocks[layer]

    if part == "attn":
        return getattr(
            block, "attn", getattr(block, "att", None)
        )

    if part == "mlp":
        return getattr(
            block, "mlp", getattr(block, "ff", None)
        )

    raise ValueError(part)


def pair_metrics(logits, choices, labels):
    scores = logits.float().gather(1, choices)

    gold_index = (
        choices == labels.unsqueeze(1)
    ).long().argmax(1)

    assert torch.all(
        choices.gather(
            1, gold_index[:, None]
        ).squeeze(1) == labels
    )

    wrong_index = 1 - gold_index

    gold = scores.gather(
        1, gold_index[:, None]
    ).squeeze(1)

    wrong = scores.gather(
        1, wrong_index[:, None]
    ).squeeze(1)

    margin = gold - wrong

    ce = F.cross_entropy(
        scores, gold_index, reduction="none"
    )

    return margin.cpu(), ce.cpu()


def intervene(
    model, ids, readout, layer, part,
    mode, trial, seed
):
    future = mode == "future"

    if future:
        ids = torch.cat([
            ids,
            ids.new_full(
                (ids.shape[0], 1), VOCAB["<pad>"]
            )
        ], dim=1)

    position = (
        ids.shape[1] - 1
        if future else readout
    )

    module = module_at(model, layer, part)
    assert module is not None

    # Keep random directions fixed across checkpoints.
    generator = torch.Generator(device="cpu")

    generator.manual_seed(
        605000
        + 101 * seed
        + 31 * layer
        + (7 if part == "mlp" else 0)
        + trial
    )

    def patch(_module, _inputs, output):
        component = (
            output[0] if part == "attn"
            else output
        )

        modified = component.clone()

        rows = torch.arange(
            ids.shape[0], device=ids.device
        )

        original = component[rows, position]

        if mode in ("zero", "future"):
            replacement = torch.zeros_like(original)

        elif mode == "random":
            noise = torch.randn(
                original.shape,
                generator=generator,
                device="cpu",
            ).to(original.device, original.dtype)

            noise = noise / (
                noise.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
            )

            replacement = (
                original
                + noise * original.norm(
                    dim=-1, keepdim=True
                )
            )
        else:
            raise ValueError(mode)

        modified[rows, position] = replacement

        if part == "attn":
            return (modified, *output[1:])

        return modified

    handle = module.register_forward_hook(patch)

    try:
        return model(ids, readout)
    finally:
        handle.remove()


def summarise(base_m, base_ce, changed_m, changed_ce):
    margin_change = base_m - changed_m

    return {
        "mean_margin_drop": float(
            margin_change.mean()
        ),
        "mean_abs_margin_change": float(
            margin_change.abs().mean()
        ),
        "mean_candidate_ce_increase": float(
            (changed_ce - base_ce).mean()
        ),
        "prediction_flip_rate": float(
            (
                (base_m > 0) != (changed_m > 0)
            ).float().mean()
        ),
        "margin_drop_by_case": [
            round(float(v), 6)
            for v in margin_change
        ],
    }


@torch.inference_mode()
def evaluate_step(
    model, calibration, device,
    seed, step, repeats
):
    model.eval()

    ids, readout, labels, choices = encode_batch(
        calibration
    )

    ids, readout, labels, choices = [
        tensor.to(device)
        for tensor in (
            ids, readout, labels, choices
        )
    ]

    base_m, base_ce = pair_metrics(
        model(ids, readout), choices, labels
    )

    # Verify that future padding does not change
    # predictions at the causal readout position.
    padded = torch.cat([
        ids,
        ids.new_full(
            (ids.shape[0], 1), VOCAB["<pad>"]
        )
    ], dim=1)

    padded_m, _ = pair_metrics(
        model(padded, readout), choices, labels
    )

    padding_error = float(
        (padded_m - base_m).abs().max()
    )

    assert padding_error < 2e-3, (
        f"Causal padding check failed: {padding_error}"
    )

    sites = {}

    for layer in (0, 1):
        for part in ("attn", "mlp"):
            label = f"L{layer+1}_{part}_readout"

            zero_m, zero_ce = pair_metrics(
                intervene(
                    model, ids, readout,
                    layer, part, "zero", 0, seed
                ),
                choices, labels,
            )

            # A future token must not influence
            # an earlier answer in a causal model.
            future_m, _ = pair_metrics(
                intervene(
                    model, ids, readout,
                    layer, part, "future", 0, seed
                ),
                choices, labels,
            )

            future_error = float(
                (padded_m - future_m).abs().max()
            )

            assert future_error < 2e-3, (
                f"Future-token control failed: "
                f"{label}, error={future_error}"
            )

            random_trials = []

            for trial in range(repeats):
                random_m, random_ce = pair_metrics(
                    intervene(
                        model, ids, readout,
                        layer, part, "random",
                        trial, seed
                    ),
                    choices, labels,
                )

                random_trials.append(
                    summarise(
                        base_m, base_ce,
                        random_m, random_ce
                    )
                )

            zero = summarise(
                base_m, base_ce, zero_m, zero_ce
            )

            random_mean = {}

            for metric in (
                "mean_margin_drop",
                "mean_abs_margin_change",
                "mean_candidate_ce_increase",
                "prediction_flip_rate",
            ):
                random_mean[metric] = sum(
                    r[metric] for r in random_trials
                ) / repeats

            sites[label] = {
                "zero": zero,
                "random": {
                    "repeats": repeats,
                    **random_mean,
                },
                "future_max_abs_margin_change": (
                    future_error
                ),
            }

    return {
        "step": step,
        "baseline_pair_accuracy": float(
            (base_m > 0).float().mean()
        ),
        "baseline_mean_margin": float(
            base_m.mean()
        ),
        "baseline_candidate_ce": float(
            base_ce.mean()
        ),
        "sites": sites,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="results/ml057_development"
    )

    parser.add_argument(
        "--out",
        default="results/ml060_layer_dependency.json"
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[7, 17],
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=list(DEFAULT_STEPS),
    )

    parser.add_argument(
        "--random-repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )

    args = parser.parse_args()

    assert args.random_repeats >= 1

    device = args.device

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if device == "mps":
        assert torch.backends.mps.is_available()

    if device == "cuda":
        assert torch.cuda.is_available()

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    _, calibration, _ = make_split()

    root = Path(args.root)
    digest = check_manifest(root, calibration)

    report = {
        "experiment": "ML-060",
        "source": "ML-057 calibration only",
        "manifest_sha256": digest,
        "device": device,
        "seeds": args.seeds,
        "steps": args.steps,
        "calibration_case_keys": [
            case_key(e) for e in calibration
        ],
        "random_repeats": args.random_repeats,
        "limitations": (
            "Exploratory causal sensitivity. "
            "Zeroing is not necessarily on-manifold. "
            "Matched-norm random perturbations do not "
            "guarantee matched functional disruption. "
            "Four independent calibration name pairs. "
            "No semantic role circuit is identified."
        ),
        "runs": [],
    }

    output = Path(args.out)
    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    for seed in args.seeds:
        for step in args.steps:
            path = (
                root
                / f"seed_{seed}"
                / f"checkpoint_{step:04d}.pt"
            )

            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

            assert checkpoint["split_hash"] == digest
            assert checkpoint["seed"] == seed
            assert checkpoint["step"] == step

            model = ScratchTransformer(
                width=96, heads=4, layers=2
            ).to(device)

            model.load_state_dict(
                checkpoint["model"],
                strict=True,
            )

            result = evaluate_step(
                model, calibration,
                device, seed, step,
                args.random_repeats,
            )

            result["seed"] = seed
            report["runs"].append(result)

            output.write_text(
                json.dumps(report, indent=2) + "\n"
            )

            summary = ", ".join(
                f"{name}: "
                f"zero={site['zero']['mean_candidate_ce_increase']:+.3f} "
                f"rnd={site['random']['mean_candidate_ce_increase']:+.3f}"
                for name, site in result["sites"].items()
            )

            print(
                f"seed={seed} "
                f"step={step:3d} "
                f"acc={result['baseline_pair_accuracy']:.3f} "
                f"CE0={result['baseline_candidate_ce']:.3f} "
                f"| {summary}",
                flush=True,
            )

    print("\nSaved:", output)
    print(
        "Future-token controls passed. "
        "No protected examples evaluated."
    )


if __name__ == "__main__":
    main()
