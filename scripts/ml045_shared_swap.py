
"""ML-045: Same attention perturbation, different linguistic question.

Exploratory analysis using ML-041 calibration examples.
One swap-derived vector is injected identically into agent and
patient questions for the same event and grammatical voice.

Compare live MLP computation with replay of the natural MLP output.
No formerly protected examples are accessed.
"""

import argparse
import hashlib
import json
from statistics import mean

import torch

from ml041_prospective_transfer import (
    DESIGN_HASH as SOURCE_HASH,
    RESULTS,
    SEEDS,
    SPLIT,
    quartets,
    load_model,
)
from ml044_role_routing import capture

CONTROLS = 4

DESIGN = {
    "experiment": "ML-045",
    "source_design": SOURCE_HASH,
    "scope": "exploratory calibration only",
    "test": "same within-voice donor vector across both question roles",
    "donors": ("agent_question", "patient_question"),
    "heads": "all four block-2 heads",
    "mlp": ("live", "natural_replay"),
    "controls": CONTROLS,
}

DESIGN_HASH = hashlib.sha256(
    json.dumps(DESIGN, sort_keys=True).encode()
).hexdigest()


def random_signs(
    seed, key, head, donor, control,
    width, device, dtype
):
    payload = json.dumps([
        "ML045", seed, list(key[0]),
        key[1], key[2],
        head, donor, control,
    ])

    number = int.from_bytes(
        hashlib.sha256(
            payload.encode()
        ).digest()[:8],
        "big",
    ) % (2**63 - 1)

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(number)

    return (
        2 * torch.randint(
            0, 2, (2, width),
            generator=generator
        ) - 1
    ).to(device=device, dtype=dtype)


@torch.no_grad()
def intervene(model, c, seed, key, device):
    swap = c["layers"][1]["swap_deltas"]

    heads = swap.shape[1]
    width = swap.shape[2]

    natural_mlp = {}

    idx4 = torch.arange(
        4, device=device
    )

    def save_mlp(_module, _args, output):
        natural_mlp["value"] = (
            output[idx4, c["pos"]]
            .detach()
            .clone()
        )

    handle = (
        model.blocks[1].mlp
        .register_forward_hook(save_mlp)
    )

    try:
        baseline_logits = model(
            c["ids"], c["pos"]
        )
    finally:
        handle.remove()

    if float(
        (
            baseline_logits - c["logits"]
        ).abs().max()
    ) > 0.05:
        raise RuntimeError(
            "Natural logits changed"
        )

    variants = [
        (
            "baseline", "live",
            -1, -1, -1,
            torch.zeros(4, width, device=device)
        ),
        (
            "baseline", "natural_replay",
            -1, -1, -1,
            torch.zeros(4, width, device=device)
        ),
    ]

    for head in range(heads):
        for donor in (0, 1):

            # Active: one donor vector for both questions.
            # Passive: one donor vector for both questions.
            base_offset = torch.stack([
                swap[donor, head],
                swap[donor, head],
                swap[2 + donor, head],
                swap[2 + donor, head],
            ])

            assert torch.equal(
                base_offset[0],
                base_offset[1]
            )

            assert torch.equal(
                base_offset[2],
                base_offset[3]
            )

            for mode in (
                "live", "natural_replay"
            ):
                variants.append((
                    "actual",
                    mode, head, donor, -1,
                    base_offset
                ))

                for j in range(CONTROLS):
                    signs = random_signs(
                        seed, key, head,
                        donor, j, width,
                        device, base_offset.dtype
                    )

                    control_offset = (
                        base_offset * torch.stack([
                            signs[0], signs[0],
                            signs[1], signs[1]
                        ])
                    )

                    variants.append((
                        "control",
                        mode, head, donor, j,
                        control_offset
                    ))

    n = len(variants)

    ids = c["ids"].repeat(n, 1)
    positions = c["pos"].repeat(n)

    batch_indices = torch.arange(
        4 * n, device=device
    )

    offsets = torch.stack([
        item[5] for item in variants
    ]).reshape(4 * n, width)

    replay = torch.tensor([
        item[1] == "natural_replay"
        for item in variants
    ], device=device).repeat_interleave(4)

    natural_outputs = (
        natural_mlp["value"]
        .repeat(n, 1)
    )

    def patch_attention(_module, _args, output):
        values, *rest = output
        updated = values.clone()

        updated[
            batch_indices, positions
        ] = (
            updated[batch_indices, positions]
            + offsets.to(updated.dtype)
        )

        return (updated, *rest)

    def patch_mlp(_module, _args, output):
        updated = output.clone()

        updated[
            batch_indices, positions
        ] = torch.where(
            replay[:, None],
            natural_outputs.to(updated.dtype),
            updated[batch_indices, positions],
        )

        return updated

    handles = [
        model.blocks[1].attn.register_forward_hook(
            patch_attention
        ),
        model.blocks[1].mlp.register_forward_hook(
            patch_mlp
        ),
    ]

    try:
        logits = model(
            ids, positions
        ).reshape(n, 4, -1)
    finally:
        for h in handles:
            h.remove()

    for baseline_index in (0, 1):
        error = float(
            (
                logits[baseline_index]
                - c["logits"]
            ).abs().max()
        )

        if error > 0.05:
            raise RuntimeError(
                f"Baseline replay failed: {error}"
            )

    # For each voice, compare the same ordered pair
    # of agent/patient name logits under both questions.
    agents = torch.stack([
        c["labels"][0], c["labels"][0],
        c["labels"][2], c["labels"][2],
    ])

    patients = torch.stack([
        c["labels"][1], c["labels"][1],
        c["labels"][3], c["labels"][3],
    ])

    rows = torch.arange(
        4, device=device
    )

    contrast = (
        logits[:, rows, agents]
        - logits[:, rows, patients]
    )

    changes = contrast - contrast[0]

    def locate(kind, mode, head, donor, control=-1):
        return next(
            i for i, v in enumerate(variants)
            if v[:5] == (
                kind, mode, head,
                donor, control
            )
        )

    output = []

    for head in range(heads):
        for donor in (0, 1):

            for voice_idx, voice in enumerate(
                ("active", "passive")
            ):
                j = 2 * voice_idx

                va = swap[j, head]
                vp = swap[j + 1, head]

                na = float(va.norm())
                np = float(vp.norm())

                cosine = (
                    float(
                        torch.dot(va, vp)
                        / (na * np)
                    )
                    if min(na, np) > 1e-8
                    else None
                )

                for mode in (
                    "live", "natural_replay"
                ):
                    ai = locate(
                        "actual", mode,
                        head, donor
                    )

                    ci = [
                        locate(
                            "control", mode,
                            head, donor, k
                        )
                        for k in range(CONTROLS)
                    ]

                    actual = changes[
                        ai, j:j + 2
                    ]

                    control = changes[
                        ci, j:j + 2
                    ].mean(dim=0)

                    output.append({
                        "head": head,
                        "donor_question": donor,
                        "voice": voice,
                        "mlp": mode,
                        "donor_cosine_between_questions": cosine,
                        "agent_donor_norm": na,
                        "patient_donor_norm": np,
                        "agent_question_contrast_change": float(
                            actual[0]
                        ),
                        "patient_question_contrast_change": float(
                            actual[1]
                        ),
                        "recipient_conditioning": float(
                            actual[0] - actual[1]
                        ),
                        "control_conditioning": float(
                            control[0] - control[1]
                        ),
                        "agent_harm": float(
                            control[0] - actual[0]
                        ),
                        "patient_harm": float(
                            actual[1] - control[1]
                        ),
                    })

    return output


def run(seed, phase, device):
    model, digest = load_model(
        seed, device
    )

    frozen = json.loads(
        (
            RESULTS / "ml041_frozen_predictions.json"
        ).read_text()
    )

    if (
        frozen.get("design_hash") != SOURCE_HASH
        or digest != frozen["seeds"][
            str(seed)
        ]["checkpoint_sha256"]
    ):
        raise RuntimeError(
            "ML-041 checkpoint provenance mismatch"
        )

    cases = quartets(
        SPLIT[
            "train"
            if phase == "audit"
            else "calibration"
        ]
    )

    if phase == "audit":
        c = capture(
            model, cases[0][3], device
        )

        records = intervene(
            model, c, seed,
            cases[0][:3], device
        )

        assert len(records) == 32

        print(
            "AUDIT PASS: identical donor vector "
            "across questions, matched controls, "
            "live MLP and natural MLP replay"
        )

        return

    path = RESULTS / (
        f"ml045_shared_swap_seed{seed}.json"
    )

    if path.exists():
        raise RuntimeError(
            f"Refusing overwrite: {path}"
        )

    records = []

    for i, (
        pair, verb, direction, examples
    ) in enumerate(cases, 1):

        c = capture(
            model, examples, device
        )

        for row in intervene(
            model, c, seed,
            (pair, verb, direction),
            device
        ):
            records.append({
                "pair": list(pair),
                "verb": verb,
                "direction": direction,
                **row,
            })

        if i % 8 == 0:
            print(
                f"seed {seed}: "
                f"{i}/56 quartets"
            )

    summary = {}

    for head in range(4):
        live = [
            r for r in records
            if r["head"] == head
            and r["donor_question"] == 0
            and r["mlp"] == "live"
        ]

        fixed = [
            r for r in records
            if r["head"] == head
            and r["donor_question"] == 0
            and r["mlp"] == "natural_replay"
        ]

        summary[str(head)] = {
            "agent_harm": mean(
                r["agent_harm"] for r in live
            ),
            "patient_harm": mean(
                r["patient_harm"] for r in live
            ),
            "recipient_conditioning": mean(
                r["recipient_conditioning"]
                for r in live
            ),
            "control_conditioning": mean(
                r["control_conditioning"]
                for r in live
            ),
            "mlp_contribution": (
                mean(
                    r["recipient_conditioning"]
                    for r in live
                )
                - mean(
                    r["recipient_conditioning"]
                    for r in fixed
                )
            ),
        }

    result = {
        "experiment": "ML-045",
        "complete": True,
        "design_hash": DESIGN_HASH,
        "seed": seed,
        "eligible_ml041": frozen[
            "seeds"
        ][str(seed)]["eligible"],
        "checkpoint_sha256": digest,
        "summary": summary,
        "records": records,
    }

    path.write_text(
        json.dumps(result, indent=2)
    )

    print(
        f"\nML-045 seed={seed} "
        f"eligible={result['eligible_ml041']}"
    )

    for head, s in summary.items():
        print(
            f"H{head}: "
            f"agent_harm={s['agent_harm']:+.3f} "
            f"patient_harm={s['patient_harm']:+.3f} "
            f"conditioned_effect="
            f"{s['recipient_conditioning']:+.3f} "
            f"control_effect="
            f"{s['control_conditioning']:+.3f} "
            f"MLP_contribution="
            f"{s['mlp_contribution']:+.3f}"
        )

    print("Saved:", path)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--phase",
        required=True,
        choices=("audit", "collect"),
    )

    p.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=SEEDS,
    )

    p.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
    )

    args = p.parse_args()

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    ) if args.device == "auto" else args.device

    if (
        device == "mps"
        and not torch.backends.mps.is_available()
    ):
        raise RuntimeError(
            "MPS unavailable"
        )

    print(
        f"ML-045 {args.phase} "
        f"seed={args.seed} device={device} "
        f"hash={DESIGN_HASH}"
    )

    run(seed=args.seed, phase=args.phase, device=device)


if __name__ == "__main__":
    main()
