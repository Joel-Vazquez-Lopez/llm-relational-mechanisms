
"""ML-068: Construction-stratified audit of ML-067.

Uses frozen ML-057 checkpoints and the historical ML-037 holdout.
No training, new examples, or checkpoint changes.

Reuses ML-067's exact factorial intervention, but separates results
by grammatical voice, requested semantic role, and participant pair.

Exploratory: the historical holdout has been studied repeatedly.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ml037_scratch_transformer_pilot import (
    ScratchTransformer,
    encode_batch,
    split_examples,
)
from ml057_developmental_checkpoints import case_key
from ml066_role_swap import (
    swap_role,
    swap_verb,
    answer_margins,
)
from ml067_path_factorial import snapshot, hybrid


SEEDS = (7, 17, 23, 29, 31, 37, 41, 43)
STEP = 300


def get_margins(
    model, ids, readout, labels,
    choices, clean, shifted,
):
    """Reproduce the four ML-067 conditions."""

    output = {}

    conditions = (
        ("clean", "clean"),
        ("clean", "swapped"),
        ("swapped", "clean"),
        ("swapped", "swapped"),
    )

    for x_source, a_source in conditions:
        logits = hybrid(
            model,
            ids,
            readout,
            clean,
            shifted,
            x_source,
            a_source,
        )

        output[(x_source, a_source)] = (
            answer_margins(
                logits, labels, choices
            )
        )

    return output


def calculate(margins, control_margins, selected):
    n = int(selected.sum())

    if n == 0:
        return None

    m00 = margins[
        ("clean", "clean")
    ][selected]

    m01 = margins[
        ("clean", "swapped")
    ][selected]

    m10 = margins[
        ("swapped", "clean")
    ][selected]

    m11 = margins[
        ("swapped", "swapped")
    ][selected]

    control = control_margins[
        ("swapped", "swapped")
    ][selected]

    attention = 0.5 * (
        (m00 - m01) + (m10 - m11)
    )

    input_state = 0.5 * (
        (m00 - m10) + (m01 - m11)
    )

    total = m00 - m11

    interaction = (
        m00 - m01 - m10 + m11
    )

    assert torch.allclose(
        attention + input_state,
        total,
        atol=2e-4,
    )

    def avg(x):
        return float(x.mean())

    attn_mean = avg(attention)
    input_mean = avg(input_state)

    return {
        "n": n,
        "attention_margin_contribution": (
            attn_mean
        ),
        "input_margin_contribution": (
            input_mean
        ),
        "attention_minus_input": (
            attn_mean - input_mean
        ),
        "total_margin_drop": avg(total),
        "factorial_interaction": (
            avg(interaction)
        ),
        "combined_flip_rate": avg(
            (m11 < 0).float()
        ),
        "attention_only_flip_rate": avg(
            (m01 < 0).float()
        ),
        "input_only_flip_rate": avg(
            (m10 < 0).float()
        ),
        "verb_control_flip_rate": avg(
            (control < 0).float()
        ),
    }


@torch.inference_mode()
def evaluate(model, examples, device):
    ids, readout, labels, choices = [
        tensor.to(device)
        for tensor in encode_batch(examples)
    ]

    clean_logits, clean = snapshot(
        model, ids, readout
    )

    clean_margin = answer_margins(
        clean_logits,
        labels,
        choices,
    )

    donor_results = {}

    for name, donor_examples in (
        (
            "role",
            [swap_role(e) for e in examples],
        ),
        (
            "verb",
            [swap_verb(e) for e in examples],
        ),
    ):
        donor_ids, donor_readout, _, _ = [
            tensor.to(device)
            for tensor in encode_batch(
                donor_examples
            )
        ]

        assert donor_ids.shape == ids.shape
        assert torch.equal(
            donor_readout, readout
        )

        donor_logits, donor_state = snapshot(
            model,
            donor_ids,
            donor_readout,
        )

        shifted_logits, shifted_state = snapshot(
            model,
            ids,
            readout,
            donor_update=donor_state["l1"],
        )

        conditions = get_margins(
            model,
            ids,
            readout,
            labels,
            choices,
            clean,
            shifted_state,
        )

        clean_error = float(
            (
                conditions[("clean", "clean")]
                - clean_margin
            ).abs().max()
        )

        swapped_margin = answer_margins(
            shifted_logits,
            labels,
            choices,
        )

        swapped_error = float(
            (
                conditions[
                    ("swapped", "swapped")
                ]
                - swapped_margin
            ).abs().max()
        )

        assert clean_error < 0.003
        assert swapped_error < 0.003

        donor_results[name] = {
            "margins": conditions,
            "donor_margin": answer_margins(
                donor_logits,
                labels,
                choices,
            ),
        }

    # Use exactly the ML-067 eligible subset:
    # clean and opposite-role donor both correct.
    eligible = (
        (clean_margin > 0)
        & (
            donor_results["role"][
                "donor_margin"
            ] < 0
        )
    )

    role_margins = donor_results[
        "role"
    ]["margins"]

    verb_margins = donor_results[
        "verb"
    ]["margins"]

    pairs = [
        "|".join(e.pair)
        for e in examples
    ]

    voices = [
        e.voice for e in examples
    ]

    roles = [
        e.role for e in examples
    ]

    def mask_for(
        pair=None,
        voice=None,
        requested_role=None,
    ):
        membership = torch.tensor([
            (
                pair is None
                or pairs[i] == pair
            )
            and (
                voice is None
                or voices[i] == voice
            )
            and (
                requested_role is None
                or roles[i] == requested_role
            )
            for i in range(len(examples))
        ], dtype=torch.bool)

        return eligible & membership

    result = {
        "baseline_accuracy": float(
            (clean_margin > 0)
            .float().mean()
        ),
        "eligible": int(eligible.sum()),
        "aggregate": calculate(
            role_margins,
            verb_margins,
            eligible,
        ),
        "by_voice_role": {},
        "by_pair": {},
        "by_pair_voice_role": {},
    }

    for voice in ("active", "passive"):
        for requested_role in (
            "agent",
            "patient",
        ):
            name = (
                f"{voice}/{requested_role}"
            )

            result["by_voice_role"][
                name
            ] = calculate(
                role_margins,
                verb_margins,
                mask_for(
                    voice=voice,
                    requested_role=requested_role,
                ),
            )

    for pair in sorted(set(pairs)):
        result["by_pair"][pair] = calculate(
            role_margins,
            verb_margins,
            mask_for(pair=pair),
        )

        result["by_pair_voice_role"][
            pair
        ] = {}

        for voice in ("active", "passive"):
            for requested_role in (
                "agent",
                "patient",
            ):
                name = (
                    f"{voice}/{requested_role}"
                )

                result[
                    "by_pair_voice_role"
                ][pair][name] = calculate(
                    role_margins,
                    verb_margins,
                    mask_for(
                        pair=pair,
                        voice=voice,
                        requested_role=requested_role,
                    ),
                )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="results/ml057_development",
    )

    parser.add_argument(
        "--reference",
        default="results/ml067_path_factorial.json",
    )

    parser.add_argument(
        "--out",
        default="results/ml068_stratified_pathways.json",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SEEDS),
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )

    args = parser.parse_args()

    output = Path(args.out)

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {output}"
        )

    device = args.device

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if device == "cuda":
        assert torch.cuda.is_available()

    if device == "mps":
        assert torch.backends.mps.is_available()

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    _, holdout, pairs = split_examples()

    assert len(holdout) == 224
    assert len(pairs) == 7

    manifest_path = (
        Path(args.root) / "manifest.json"
    )

    manifest = json.loads(
        manifest_path.read_text()
    )

    digest = manifest.pop("sha256")

    calculated = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
        ).encode()
    ).hexdigest()

    assert calculated == digest

    reference_path = Path(args.reference)

    reference = json.loads(
        reference_path.read_text()
    )

    assert (
        reference["manifest_sha256"]
        == digest
    )

    reference_runs = {
        row["seed"]: row
        for row in reference["runs"]
    }

    assert set(args.seeds).issubset(
        reference_runs
    )

    report = {
        "experiment": "ML-068",
        "method": (
            "ML-067 factorial intervention, "
            "stratified by construction and pair"
        ),
        "evaluation_status": (
            "Historical ML-037 holdout; "
            "not independent confirmation"
        ),
        "source_manifest_sha256": digest,
        "reference_sha256": hashlib.sha256(
            reference_path.read_bytes()
        ).hexdigest(),
        "holdout_case_keys_sha256": (
            hashlib.sha256(
                json.dumps(
                    sorted(
                        case_key(e)
                        for e in holdout
                    )
                ).encode()
            ).hexdigest()
        ),
        "step": STEP,
        "seeds": args.seeds,
        "runs": [],
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for seed in args.seeds:
        path = (
            Path(args.root)
            / f"seed_{seed}"
            / "checkpoint_0300.pt"
        )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        assert checkpoint["seed"] == seed
        assert checkpoint["step"] == STEP
        assert (
            checkpoint["split_hash"]
            == digest
        )

        model = ScratchTransformer(
            width=96,
            heads=4,
            layers=2,
        ).to(device)

        model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        model.eval()

        result = evaluate(
            model,
            holdout,
            device,
        )

        # Validate that our aggregate reproduces ML-067.
        previous = reference_runs[
            seed
        ]["conditions"][
            "opposite_role"
        ]["aggregate"]

        current = result["aggregate"]

        assert (
            result["eligible"]
            == reference_runs[seed][
                "jointly_correct"
            ]
        )

        comparisons = (
            (
                "attn_path_margin_drop",
                "attention_margin_contribution",
            ),
            (
                "input_path_margin_drop",
                "input_margin_contribution",
            ),
            (
                "both_flip_rate",
                "combined_flip_rate",
            ),
        )

        for old_name, new_name in comparisons:
            assert abs(
                previous[old_name]
                - current[new_name]
            ) < 0.003, (
                f"ML-067 reproduction mismatch: "
                f"seed={seed}, {old_name}"
            )

        report["runs"].append({
            "seed": seed,
            **result,
        })

        output.write_text(
            json.dumps(
                report,
                indent=2,
            ) + "\n"
        )

        contrast = current[
            "attention_minus_input"
        ]

        if contrast > 0.1:
            dominant = "attention"
        elif contrast < -0.1:
            dominant = "input"
        else:
            dominant = "unclear"

        cells = result["by_voice_role"]

        informative_cells = [
            cell
            for cell in cells.values()
            if cell is not None
        ]

        informative_pairs = [
            cell
            for cell in result[
                "by_pair"
            ].values()
            if cell is not None
        ]

        if dominant == "unclear":
            voice_summary = "n/a"
            pair_summary = "n/a"

        else:
            expected_positive = (
                dominant == "attention"
            )

            voice_matches = sum(
                (
                    cell[
                        "attention_minus_input"
                    ] > 0
                ) == expected_positive
                for cell in informative_cells
            )

            pair_matches = sum(
                (
                    cell[
                        "attention_minus_input"
                    ] > 0
                ) == expected_positive
                for cell in informative_pairs
            )

            voice_summary = (
                f"{voice_matches}/"
                f"{len(informative_cells)}"
            )

            pair_summary = (
                f"{pair_matches}/"
                f"{len(informative_pairs)}"
            )

        print(
            f"seed={seed:2} "
            f"dominant={dominant:9s} "
            f"voice_role={voice_summary} "
            f"pairs={pair_summary} "
            f"overall_D={contrast:+.2f}",
            flush=True,
        )

        for name, cell in cells.items():
            if cell is None:
                print(
                    f"  {name:15s} n=0"
                )
                continue

            print(
                f"  {name:15s} "
                f"n={cell['n']:2d} "
                f"D={cell['attention_minus_input']:+.2f} "
                f"role_flip={cell['combined_flip_rate']:.3f} "
                f"verb_ctrl={cell['verb_control_flip_rate']:.3f}"
            )

    print("\nSaved:", output)

    print(
        "Historical holdout reused. "
        "Descriptive stability audit only."
    )


if __name__ == "__main__":
    main()
