
"""ML-062: Source-resolved L2 attention edges during learning.

Uses frozen ML-057 checkpoints and calibration examples only.

Intervention:
Subtract the original weighted value contribution of selected
source tokens to the layer-2 answer-position attention output.

Attention weights are NOT renormalized.

This is causal sensitivity to attention edges, not evidence
of a unique semantic-role circuit.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ml037_scratch_transformer_pilot import (
    ScratchTransformer,
    VOCAB,
    encode_batch,
)
from ml057_developmental_checkpoints import (
    case_key,
    make_split,
)

STEPS = (0, 50, 75, 100, 150, 200, 300)

GROUPS = (
    "event",
    "role_cue",
    "question_other",
    "question_all",
    "self",
    "bos",
    "future",
)


def verify(root, examples):
    manifest = json.loads(
        (root / "manifest.json").read_text()
    )

    digest = manifest.pop("sha256")

    actual = hashlib.sha256(
        json.dumps(
            manifest, sort_keys=True
        ).encode()
    ).hexdigest()

    assert digest == actual, "Manifest mismatch"

    assert sorted(
        case_key(e) for e in examples
    ) == sorted(
        manifest["calibration_case_keys"]
    )

    return digest


def make_masks(ids, readout):
    B, T = ids.shape

    positions = torch.arange(
        T, device=ids.device
    )[None, :].expand_as(ids)

    sep_mask = (
        ids.eq(VOCAB["<sep>"])
        & (positions < readout[:, None])
    )

    assert torch.all(sep_mask.sum(1) == 1)

    sep = sep_mask.long().argmax(1)

    role = (
        ids.eq(VOCAB["acted"])
        | ids.eq(VOCAB["received"])
    ) & (positions < readout[:, None])

    assert torch.all(role.sum(1) == 1)

    question = (
        (positions >= sep[:, None])
        & (positions < readout[:, None])
    )

    masks = {
        "event": (
            (positions > 0)
            & (positions < sep[:, None])
        ),
        "role_cue": role,
        "question_other": question & ~role,
        "question_all": question,
        "self": positions == readout[:, None],
        "bos": positions == 0,
        "future": positions > readout[:, None],
    }

    for name, mask in masks.items():
        if name != "future":
            assert torch.all(mask.sum(1) > 0)

    return masks


def paired_stats(logits, gold, choices):
    candidate_scores = logits.float().gather(
        1, choices
    )

    gold_slot = (
        choices == gold[:, None]
    ).long().argmax(1)

    assert torch.all(
        choices.gather(
            1, gold_slot[:, None]
        ).flatten() == gold
    )

    correct = candidate_scores.gather(
        1, gold_slot[:, None]
    ).flatten()

    incorrect = candidate_scores.gather(
        1, 1 - gold_slot[:, None]
    ).flatten()

    margin = correct - incorrect

    ce = F.cross_entropy(
        candidate_scores,
        gold_slot,
        reduction="none",
    )

    return margin.cpu(), ce.cpu()


def edge_intervention(
    model, ids, readout, source_mask
):
    """Subtract weighted L2 value messages from selected sources."""

    attn = getattr(model.blocks[1], "attn", None)

    if attn is None:
        attn = getattr(model.blocks[1], "att", None)

    assert attn is not None

    diagnostics = {}

    def before(_module, args, kwargs):
        kwargs = dict(kwargs)

        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return args, kwargs

    def after(_module, args, _kwargs, output):
        attn_output, weights = output

        B, T, D = attn_output.shape
        H = attn.num_heads

        assert D % H == 0
        assert weights.shape == (B, H, T, T)
        assert source_mask.shape == (B, T)

        # Reconstruct the original per-head value vectors.
        inputs = args[2]

        value_weight = attn.in_proj_weight[
            2 * D:3 * D
        ]

        value_bias = (
            None
            if attn.in_proj_bias is None
            else attn.in_proj_bias[2 * D:3 * D]
        )

        values = F.linear(
            inputs,
            value_weight,
            value_bias,
        )

        values = values.reshape(
            B, T, H, D // H
        ).permute(0, 2, 1, 3).float()

        batch = torch.arange(
            B, device=ids.device
        )

        # Original attention weights from answer to sources.
        row_weights = weights[
            batch, :, readout, :
        ].float()

        # Numerical check: reconstruct unmodified attention.
        all_values = torch.einsum(
            "bht,bhtd->bhd",
            row_weights,
            values,
        ).reshape(B, D)

        rebuilt = F.linear(
            all_values,
            attn.out_proj.weight.float(),
            (
                None
                if attn.out_proj.bias is None
                else attn.out_proj.bias.float()
            ),
        )

        original = attn_output[
            batch, readout
        ].float()

        error = float(
            (rebuilt - original).abs().max()
        )

        assert error < 0.003, (
            f"Attention reconstruction failed: {error}"
        )

        # Remove only the chosen source contributions.
        # Crucially, do not change attention weights.
        selected_weights = (
            row_weights
            * source_mask[:, None, :].float()
        )

        selected_values = torch.einsum(
            "bht,bhtd->bhd",
            selected_weights,
            values,
        ).reshape(B, D)

        removed = F.linear(
            selected_values,
            attn.out_proj.weight.float(),
            None,
        )

        patched = attn_output.clone()

        patched[batch, readout] = (
            original - removed
        ).to(attn_output.dtype)

        diagnostics.update({
            "reconstruction_error": error,
            "removed_l2_norm": float(
                removed.norm(dim=1).mean()
            ),
            "source_attention_mass": float(
                selected_weights.sum(-1).mean()
            ),
        })

        return (patched, weights)

    pre_handle = attn.register_forward_pre_hook(
        before, with_kwargs=True
    )

    post_handle = attn.register_forward_hook(
        after, with_kwargs=True
    )

    try:
        logits = model(ids, readout)
    finally:
        post_handle.remove()
        pre_handle.remove()

    assert "reconstruction_error" in diagnostics

    return logits, diagnostics


@torch.inference_mode()
def evaluate(model, examples, device):
    model.eval()

    ids, readout, gold, choices = encode_batch(
        examples
    )

    ids, readout, gold, choices = [
        tensor.to(device)
        for tensor in (
            ids, readout, gold, choices
        )
    ]

    original_logits = model(ids, readout)

    original_margin, original_ce = paired_stats(
        original_logits,
        gold,
        choices,
    )

    masks = make_masks(ids, readout)

    outcomes = {}

    for name, source_mask in masks.items():
        new_logits, info = edge_intervention(
            model,
            ids,
            readout,
            source_mask,
        )

        new_margin, new_ce = paired_stats(
            new_logits,
            gold,
            choices,
        )

        ce_change = new_ce - original_ce

        by_pair = {}

        for pair in sorted({
            e.pair for e in examples
        }):
            indices = [
                i for i, e in enumerate(examples)
                if e.pair == pair
            ]

            by_pair["|".join(pair)] = float(
                ce_change[indices].mean()
            )

        outcomes[name] = {
            "mean_ce_increase": float(
                ce_change.mean()
            ),
            "mean_margin_drop": float(
                (original_margin - new_margin).mean()
            ),
            "flip_rate": float(
                (
                    (original_margin > 0)
                    != (new_margin > 0)
                ).float().mean()
            ),
            "mean_source_tokens": float(
                source_mask.sum(1).float().mean()
            ),
            "by_pair_ce_increase": by_pair,
            **info,
        }

        # Future source positions must have zero influence.
        if name == "future":
            assert (
                info["source_attention_mass"] < 1e-6
            )

            assert float(
                (original_margin - new_margin)
                .abs().max()
            ) < 0.003

    return {
        "baseline_accuracy": float(
            (original_margin > 0).float().mean()
        ),
        "baseline_ce": float(
            original_ce.mean()
        ),
        "outcomes": outcomes,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="results/ml057_development",
    )

    parser.add_argument(
        "--out",
        default="results/ml062_l2_source_edges.json",
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
        default=list(STEPS),
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )

    args = parser.parse_args()

    torch.set_num_threads(
        min(torch.get_num_threads(), 4)
    )

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

    _, calibration, _ = make_split()

    root = Path(args.root)

    digest = verify(
        root, calibration
    )

    report = {
        "experiment": "ML-062",
        "source": "ML-057 calibration",
        "manifest_sha256": digest,
        "method": (
            "Subtract original weighted L2 value-to-readout "
            "contribution without softmax renormalization."
        ),
        "limitations": (
            "Exploratory causal edge ablation. "
            "Off-manifold effects remain possible. "
            "Four calibration name-pair clusters. "
            "No unique semantic mechanism established."
        ),
        "runs": [],
    }

    output = Path(args.out)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for seed in args.seeds:
        for step in args.steps:
            checkpoint_path = (
                root
                / f"seed_{seed}"
                / f"checkpoint_{step:04d}.pt"
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            assert checkpoint["seed"] == seed
            assert checkpoint["step"] == step
            assert checkpoint["split_hash"] == digest

            model = ScratchTransformer(
                width=96,
                heads=4,
                layers=2,
            ).to(device)

            model.load_state_dict(
                checkpoint["model"],
                strict=True,
            )

            result = evaluate(
                model,
                calibration,
                device,
            )

            report["runs"].append({
                "seed": seed,
                "step": step,
                **result,
            })

            output.write_text(
                json.dumps(report, indent=2) + "\n"
            )

            effects = " ".join(
                f"{group}="
                f"{result['outcomes'][group]['mean_ce_increase']:+.3f}"
                for group in (
                    "event",
                    "role_cue",
                    "question_other",
                    "self",
                    "future",
                )
            )

            print(
                f"seed={seed} "
                f"step={step:3d} "
                f"acc={result['baseline_accuracy']:.3f} "
                f"| {effects}",
                flush=True,
            )

    print("\nSaved:", output)
    print(
        "ML-057 calibration only. "
        "No protected examples evaluated."
    )


if __name__ == "__main__":
    main()
