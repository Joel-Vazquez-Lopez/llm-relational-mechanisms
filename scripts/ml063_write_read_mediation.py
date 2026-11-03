
"""ML-063: causal mediation of L1 attention updates through L2 Q/K/V.

Read-only examination of ML-057 calibration checkpoints.
No training or protected evaluation.

Restoring projected inputs is a hybrid-state intervention,
not proof of unique information content.
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

STEPS = (50, 75, 100, 150, 200, 300)
SITES = ("event", "role_cue", "question_other", "readout")
REPAIRS = ("Q", "K", "V", "KV")


def validate(root, examples):
    manifest = json.loads(
        (root / "manifest.json").read_text()
    )

    digest = manifest.pop("sha256")
    expected = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()

    assert digest == expected, "ML-057 manifest mismatch"

    assert sorted(case_key(e) for e in examples) == sorted(
        manifest["calibration_case_keys"]
    )

    return digest


def masks_for(ids, readout):
    positions = torch.arange(
        ids.size(1), device=ids.device
    )[None, :].expand_as(ids)

    sep_mask = (
        (ids == VOCAB["<sep>"])
        & (positions < readout[:, None])
    )

    role_mask = (
        (ids == VOCAB["acted"])
        | (ids == VOCAB["received"])
    ) & (positions < readout[:, None])

    assert (sep_mask.sum(1) == 1).all()
    assert (role_mask.sum(1) == 1).all()

    sep = sep_mask.long().argmax(1)

    question = (
        (positions >= sep[:, None])
        & (positions < readout[:, None])
    )

    masks = {
        "event": (
            (positions > 0)
            & (positions < sep[:, None])
        ),
        "role_cue": role_mask,
        "question_other": question & ~role_mask,
        "readout": positions == readout[:, None],
    }

    assert all(
        mask.any(dim=1).all()
        for mask in masks.values()
    )

    assert all(
        not (
            mask & (positions > readout[:, None])
        ).any()
        for mask in masks.values()
    )

    return masks


def per_case_ce(logits, labels, choices):
    candidates = logits.float().gather(1, choices)

    is_gold = choices == labels[:, None]

    assert (is_gold.sum(1) == 1).all()

    targets = is_gold.long().argmax(1)

    return F.cross_entropy(
        candidates, targets, reduction="none"
    ).detach().cpu()


def evaluate(model, examples, device):
    model.eval()

    ids, readout, labels, choices = [
        x.to(device)
        for x in encode_batch(examples)
    ]

    masks = masks_for(ids, readout)

    attn1 = model.blocks[0].attn
    attn2 = model.blocks[1].attn

    clean_inputs = {}

    # Record the original inputs reaching layer-2 attention.
    def capture_l2(_module, args):
        assert len(args) >= 3

        clean_inputs["QKV"] = tuple(
            tensor.detach().clone()
            for tensor in args[:3]
        )

    handle = attn2.register_forward_pre_hook(capture_l2)

    try:
        clean_logits = model(ids, readout)
    finally:
        handle.remove()

    assert "QKV" in clean_inputs

    clean_ce = per_case_ce(
        clean_logits, labels, choices
    )

    predicted = clean_logits.gather(
        1, choices
    ).argmax(1)

    gold_slot = (
        choices == labels[:, None]
    ).long().argmax(1)

    answer = {
        "baseline_accuracy": float(
            (predicted == gold_slot).float().mean()
        ),
        "baseline_ce": float(clean_ce.mean()),
        "sites": {},
    }

    def corrupted_run(mask, repair="none"):

        # Remove the selected layer-1 attention updates.
        def ablate_l1(_module, _args, result):
            original = result[0]

            altered = original.masked_fill(
                mask[..., None], 0
            )

            return (altered, *result[1:])

        # Selectively repair layer-2 Q, K or V inputs
        # using the corresponding clean computation.
        def restore_l2(_module, args):
            q, k, v = args[:3]

            clean_q, clean_k, clean_v = (
                clean_inputs["QKV"]
            )

            projected = [q, k, v]

            if "Q" in repair:
                projected[0] = q.clone()
                projected[0][mask] = clean_q[mask]

            if "K" in repair:
                projected[1] = k.clone()
                projected[1][mask] = clean_k[mask]

            if "V" in repair:
                projected[2] = v.clone()
                projected[2][mask] = clean_v[mask]

            return tuple(projected) + tuple(args[3:])

        h1 = attn1.register_forward_hook(ablate_l1)
        h2 = attn2.register_forward_pre_hook(restore_l2)

        try:
            logits = model(ids, readout)
        finally:
            h2.remove()
            h1.remove()

        return per_case_ce(
            logits, labels, choices
        )

    for site in SITES:
        mask = masks[site]

        corrupt = corrupted_run(mask)
        harm = corrupt - clean_ce

        site_result = {
            "mean_ce_increase": float(harm.mean()),
            "mean_affected_tokens": float(
                mask.sum(1).float().mean()
            ),
            "by_pair_ce_increase": {},
            "repairs": {},
        }

        pair_names = sorted({
            "|".join(e.pair)
            for e in examples
        })

        for name in pair_names:
            selection = [
                i for i, e in enumerate(examples)
                if "|".join(e.pair) == name
            ]

            site_result["by_pair_ce_increase"][name] = (
                float(harm[selection].mean())
            )

        for repair in REPAIRS:
            restored = corrupted_run(mask, repair)

            repaired_harm = restored - clean_ce
            rescued = corrupt - restored

            average_harm = float(harm.mean())

            site_result["repairs"][repair] = {
                "mean_ce_increase_after_repair": float(
                    repaired_harm.mean()
                ),
                "mean_ce_rescued": float(
                    rescued.mean()
                ),
                "fraction_of_harm_rescued": (
                    float(
                        rescued.mean() / harm.mean()
                    )
                    if average_harm > 0.05
                    else None
                ),
            }

            # Earlier-token queries cannot change the
            # final answer in this same attention layer.
            if repair == "Q" and site != "readout":
                error = float(
                    (restored - corrupt).abs().max()
                )

                assert error < 2e-3, (
                    f"Earlier-query control failed: "
                    f"{site}, {error}"
                )

                site_result["repairs"][repair][
                    "negative_control_max_abs_ce_error"
                ] = error

        # A full restoration of earlier-token L2 inputs
        # should recover the clean answer computation.
        if site != "readout":
            full_ce = corrupted_run(mask, "QKV")

            error = float(
                (full_ce - clean_ce).abs().max()
            )

            assert error < 2e-3, (
                f"Full restoration failed: "
                f"{site}, {error}"
            )

            site_result[
                "full_QKV_repair_max_ce_error"
            ] = error

        answer["sites"][site] = site_result

    return answer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="results/ml057_development",
    )

    parser.add_argument(
        "--out",
        default="results/ml063_write_read_mediation.json",
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

    if args.device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = args.device

    if (
        device == "mps"
        and not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS unavailable")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    _, calibration, _ = make_split()

    root = Path(args.root)
    digest = validate(root, calibration)

    output = Path(args.out)
    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    report = {
        "experiment": "ML-063",
        "manifest_sha256": digest,
        "source": "ML-057 calibration only",
        "interpretation": (
            "Exploratory L1-to-L2 Q/K/V restoration; "
            "not proof of neural algorithm identity."
        ),
        "runs": [],
    }

    with torch.inference_mode():

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
                    model, calibration, device
                )

                report["runs"].append({
                    "seed": seed,
                    "step": step,
                    **result,
                })

                output.write_text(
                    json.dumps(report, indent=2) + "\n"
                )

                parts = []

                for site in SITES:
                    cell = result["sites"][site]

                    rescuers = ",".join(
                        f"{r}:"
                        f"{cell['repairs'][r]['mean_ce_rescued']:+.2f}"
                        for r in REPAIRS
                    )

                    parts.append(
                        f"{site} "
                        f"harm={cell['mean_ce_increase']:+.2f} "
                        f"rescue[{rescuers}]"
                    )

                print(
                    f"seed={seed} step={step:3d} | "
                    + " | ".join(parts),
                    flush=True,
                )

    print(f"\nSaved: {output}")
    print(
        "Negative controls passed; "
        "no protected evaluation."
    )


if __name__ == "__main__":
    main()
