
"""ML-061: Does layer-1 attention support answers through earlier tokens?

Read-only checkpoint intervention study on ML-057 calibration data.
No training, protected evaluation, or semantic-circuit claims.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ml037_scratch_transformer_pilot import (
    ScratchTransformer, VOCAB, encode_batch,
)
from ml057_developmental_checkpoints import (
    case_key, make_split,
)

STEPS = (0, 50, 75, 100, 150, 200, 300)
SITES = (
    "L1_event",
    "L1_question",
    "L1_readout",
    "L1_prefix",
    "L1_all",
    "L2_readout",
)


def verify_manifest(root, examples):
    manifest = json.loads(
        (root / "manifest.json").read_text()
    )
    digest = manifest.pop("sha256")
    actual = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()

    assert digest == actual, "Manifest hash mismatch"
    assert sorted(case_key(e) for e in examples) == sorted(
        manifest["calibration_case_keys"]
    )
    return digest


def token_mask(ids, readout, group):
    positions = torch.arange(
        ids.shape[1], device=ids.device
    )[None, :].expand_as(ids)

    sep_matches = (
        (ids == VOCAB["<sep>"])
        & (positions < readout[:, None])
    )
    assert torch.all(sep_matches.sum(1) == 1)

    sep = sep_matches.long().argmax(1)

    masks = {
        "event": (
            (positions > 0)
            & (positions < sep[:, None])
        ),
        "question": (
            (positions >= sep[:, None])
            & (positions < readout[:, None])
        ),
        "readout": positions == readout[:, None],
        "prefix": positions < readout[:, None],
        "all": positions <= readout[:, None],
    }

    mask = masks[group]

    assert torch.all(mask.sum(1) > 0)
    assert not (
        mask & (positions > readout[:, None])
    ).any()

    return mask


def scores(logits, choices, labels):
    candidate_logits = logits.float().gather(
        1, choices
    )

    gold_slot = (
        choices == labels[:, None]
    ).long().argmax(1)

    assert torch.all(
        choices.gather(
            1, gold_slot[:, None]
        ).flatten() == labels
    )

    other_slot = 1 - gold_slot

    gold_score = candidate_logits.gather(
        1, gold_slot[:, None]
    ).flatten()

    other_score = candidate_logits.gather(
        1, other_slot[:, None]
    ).flatten()

    margins = gold_score - other_score

    ce = F.cross_entropy(
        candidate_logits,
        gold_slot,
        reduction="none",
    )

    return margins.cpu(), ce.cpu()


def patched_forward(
    model, ids, readout,
    layer, mask, mode,
    random_seed=0,
):
    block = model.blocks[layer]
    attn = getattr(block, "attn", None)

    if attn is None:
        attn = getattr(block, "att", None)

    assert attn is not None

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(random_seed)

    perturbation_norms = []

    def hook(_module, _inputs, result):
        original = result[0]
        assert original.shape[:2] == mask.shape

        if mode == "zero":
            modified = original.masked_fill(
                mask[..., None], 0
            )

        elif mode == "random":
            noise = torch.randn(
                original.shape,
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            ).to(original.device, original.dtype)

            noise = noise / noise.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)

            strength = original.norm(
                dim=-1, keepdim=True
            )

            modified = original + torch.where(
                mask[..., None],
                noise * strength,
                torch.zeros_like(original),
            )
        else:
            raise ValueError(mode)

        difference = (
            modified - original
        ).float().flatten(1)

        perturbation_norms.extend(
            difference.norm(dim=1)
            .detach().cpu().tolist()
        )

        return (modified, *result[1:])

    handle = attn.register_forward_hook(hook)

    try:
        logits = model(ids, readout)
    finally:
        handle.remove()

    assert len(perturbation_norms) == len(ids)

    return (
        logits,
        sum(perturbation_norms)
        / len(perturbation_norms),
    )


def summarize_effect(
    base_margin, base_ce,
    new_margin, new_ce,
    examples, perturbation_norm,
):
    delta_ce = new_ce - base_ce
    delta_margin = base_margin - new_margin

    by_pair = {}

    for pair in sorted({
        e.pair for e in examples
    }):
        indices = [
            i for i, e in enumerate(examples)
            if e.pair == pair
        ]

        by_pair["|".join(pair)] = float(
            delta_ce[indices].mean()
        )

    originally_correct = base_margin > 0
    originally_wrong = ~originally_correct

    return {
        "mean_ce_increase": float(
            delta_ce.mean()
        ),
        "mean_margin_drop": float(
            delta_margin.mean()
        ),
        "prediction_flip_rate": float(
            (
                (base_margin > 0)
                != (new_margin > 0)
            ).float().mean()
        ),
        "mean_perturbation_norm": (
            perturbation_norm
        ),
        "ce_increase_on_correct_cases": (
            float(
                delta_ce[
                    originally_correct
                ].mean()
            )
            if originally_correct.any()
            else None
        ),
        "ce_increase_on_wrong_cases": (
            float(
                delta_ce[
                    originally_wrong
                ].mean()
            )
            if originally_wrong.any()
            else None
        ),
        "ce_increase_by_name_pair": by_pair,
    }


@torch.inference_mode()
def evaluate(
    model, examples, device,
    seed, step, repeats,
):
    model.eval()

    ids, readout, labels, choices = (
        encode_batch(examples)
    )

    ids, readout, labels, choices = [
        tensor.to(device)
        for tensor in (
            ids, readout, labels, choices
        )
    ]

    base_m, base_ce = scores(
        model(ids, readout),
        choices, labels,
    )

    # Appending a future token must not
    # alter the answer of a causal model.
    future_ids = torch.cat([
        ids,
        ids.new_full(
            (len(ids), 1),
            VOCAB["<pad>"],
        ),
    ], dim=1)

    future_m, _ = scores(
        model(future_ids, readout),
        choices, labels,
    )

    future_error = float(
        (future_m - base_m).abs().max()
    )

    assert future_error < 2e-3, (
        "Future-token invariance failed"
    )

    results = {}

    for index, site in enumerate(SITES):
        layer = (
            0 if site.startswith("L1")
            else 1
        )
        group = site.split("_", 1)[1]

        mask = token_mask(
            ids, readout, group
        )

        zero_logits, zero_norm = (
            patched_forward(
                model, ids, readout,
                layer, mask, "zero",
            )
        )

        zero_m, zero_ce = scores(
            zero_logits, choices, labels
        )

        zero_stats = summarize_effect(
            base_m, base_ce,
            zero_m, zero_ce,
            examples, zero_norm,
        )

        random_stats = []

        for trial in range(repeats):
            random_logits, random_norm = (
                patched_forward(
                    model, ids, readout,
                    layer, mask, "random",
                    random_seed=(
                        610000
                        + seed * 101
                        + step * 7
                        + layer * 13
                        + index * 5
                        + trial
                    ),
                )
            )

            random_m, random_ce = scores(
                random_logits,
                choices, labels,
            )

            random_stats.append(
                summarize_effect(
                    base_m, base_ce,
                    random_m, random_ce,
                    examples, random_norm,
                )
            )

        results[site] = {
            "mean_tokens_modified": float(
                mask.sum(1).float().mean()
            ),
            "zero": zero_stats,
            "random_mean_ce_increase": (
                sum(
                    r["mean_ce_increase"]
                    for r in random_stats
                ) / repeats
            ),
            "random_mean_perturbation_norm": (
                sum(
                    r["mean_perturbation_norm"]
                    for r in random_stats
                ) / repeats
            ),
        }

    return {
        "seed": seed,
        "step": step,
        "baseline_accuracy": float(
            (base_m > 0).float().mean()
        ),
        "baseline_ce": float(
            base_ce.mean()
        ),
        "future_token_control_error": (
            future_error
        ),
        "sites": results,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="results/ml057_development",
    )
    parser.add_argument(
        "--out",
        default="results/ml061_token_routing.json",
    )
    parser.add_argument(
        "--seeds",
        nargs="+", type=int,
        default=[7, 17],
    )
    parser.add_argument(
        "--steps",
        nargs="+", type=int,
        default=list(STEPS),
    )
    parser.add_argument(
        "--random-repeats",
        type=int, default=3,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )

    args = parser.parse_args()

    assert args.random_repeats >= 1

    if args.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = args.device

    if device == "mps":
        assert torch.backends.mps.is_available()

    if device == "cuda":
        assert torch.cuda.is_available()

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    _, calibration, _ = make_split()

    root = Path(args.root)
    digest = verify_manifest(
        root, calibration
    )

    report = {
        "experiment": "ML-061",
        "source": "ML-057 calibration only",
        "manifest_sha256": digest,
        "intervention": (
            "Zero or randomly perturb attention "
            "outputs at specified token groups."
        ),
        "limitations": [
            "Off-distribution intervention effects",
            "Not a semantic-circuit identification",
            "Only four independent name-pair clusters",
            "Exploratory site comparison",
        ],
        "runs": [],
    }

    output = Path(args.out)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
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

            assert checkpoint["seed"] == seed
            assert checkpoint["step"] == step
            assert (
                checkpoint["split_hash"]
                == digest
            )

            model = ScratchTransformer(
                width=96, heads=4, layers=2
            ).to(device)

            model.load_state_dict(
                checkpoint["model"],
                strict=True,
            )

            result = evaluate(
                model, calibration,
                device, seed, step,
                args.random_repeats,
            )

            report["runs"].append(result)

            output.write_text(
                json.dumps(
                    report, indent=2
                ) + "\n"
            )

            summary = " ".join(
                f"{site}="
                f"{info['zero']['mean_ce_increase']:+.3f}"
                for site, info
                in result["sites"].items()
            )

            print(
                f"seed={seed} "
                f"step={step:3d} "
                f"acc={result['baseline_accuracy']:.3f} "
                f"CE0={result['baseline_ce']:.3f} "
                f"| {summary}",
                flush=True,
            )

    print("\nSaved:", output)
    print(
        "Calibration only. "
        "Historical protected cases untouched."
    )


if __name__ == "__main__":
    main()
