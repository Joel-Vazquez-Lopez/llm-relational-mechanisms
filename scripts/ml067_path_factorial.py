
"""ML-067: Factorial test of L2 attention vs input-state transfer.

Evaluates frozen ML-057 checkpoints on historically examined ML-037
held-out name pairs. No training. Not pristine confirmation.

Causal hybrid: combine layer-2 block-input state at the answer token
and layer-2 attention output at that token from clean versus
opposite-role layer-1 answer-update interchange.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ml037_scratch_transformer_pilot import (
    ScratchTransformer, encode_batch, split_examples,
)
from ml066_role_swap import swap_role, swap_verb, answer_margins
from ml057_developmental_checkpoints import case_key

SEEDS = (7, 17, 23, 29, 31, 37, 41, 43)


def snapshot(model, ids, readout, donor_update=None):
    """Capture layer-2 input and attention output, optionally L1-patched."""
    state = {}
    batch = torch.arange(len(ids), device=ids.device)

    def l1_patch(_module, _args, result):
        modified = result[0].clone()
        modified[batch, readout] = donor_update[batch, readout]
        return (modified, *result[1:])

    def record_block(_module, args):
        state["x"] = args[0].detach().clone()

    def record_attn(_module, _args, result):
        state["a"] = result[0].detach().clone()

    def record_l1(_module, _args, result):
        state["l1"] = result[0].detach().clone()

    handles = [
        model.blocks[1].register_forward_pre_hook(record_block),
        model.blocks[1].attn.register_forward_hook(record_attn),
        model.blocks[0].attn.register_forward_hook(record_l1),
    ]

    if donor_update is not None:
        handles.append(
            model.blocks[0].attn.register_forward_hook(l1_patch)
        )

    try:
        logits = model(ids, readout)
    finally:
        for handle in reversed(handles):
            handle.remove()

    assert set(state) == {"x", "a", "l1"}
    return logits, state


def hybrid(model, ids, readout, clean, swapped, x_source, a_source):
    """Mix frozen inputs at layer-2 answer position."""
    assert x_source in ("clean", "swapped")
    assert a_source in ("clean", "swapped")

    batch = torch.arange(len(ids), device=ids.device)

    source_x = clean if x_source == "clean" else swapped
    source_a = clean if a_source == "clean" else swapped

    def inject_x(_module, args):
        x = args[0].clone()
        x[batch, readout] = source_x["x"][batch, readout]
        return (x, *args[1:])

    def inject_a(_module, _args, result):
        attention = result[0].clone()
        attention[batch, readout] = source_a["a"][batch, readout]
        return (attention, *result[1:])

    h1 = model.blocks[1].register_forward_pre_hook(inject_x)
    h2 = model.blocks[1].attn.register_forward_hook(inject_a)

    try:
        return model(ids, readout)
    finally:
        h2.remove()
        h1.remove()


def summarize(m00, m01, m10, m11, eligible, pair_names):
    """m00 clean; m01 attention-only; m10 input-only; m11 both."""

    drop_total = m00 - m11

    # Symmetric attribution, accounting for both hybrid contexts.
    drop_attn = 0.5 * ((m00 - m01) + (m10 - m11))
    drop_input = 0.5 * ((m00 - m10) + (m01 - m11))

    interaction = m00 - m01 - m10 + m11

    assert torch.allclose(
        drop_attn + drop_input,
        drop_total,
        rtol=1e-4,
        atol=1e-4,
    )

    def stats(mask):
        if int(mask.sum()) == 0:
            return None

        return {
            "cases": int(mask.sum()),
            "total_margin_drop": float(drop_total[mask].mean()),
            "attn_path_margin_drop": float(drop_attn[mask].mean()),
            "input_path_margin_drop": float(drop_input[mask].mean()),
            "factorial_interaction": float(interaction[mask].mean()),
            "both_flip_rate": float((m11[mask] < 0).float().mean()),
            "attn_only_flip_rate": float((m01[mask] < 0).float().mean()),
            "input_only_flip_rate": float((m10[mask] < 0).float().mean()),
        }

    by_pair = {}

    for name in sorted(set(pair_names)):
        selected = eligible & torch.tensor(
            [n == name for n in pair_names],
            dtype=torch.bool,
        )
        by_pair[name] = stats(selected)

    return {
        "aggregate": stats(eligible),
        "by_pair": by_pair,
    }


@torch.inference_mode()
def evaluate(model, examples, device):
    ids, readout, labels, choices = [
        t.to(device) for t in encode_batch(examples)
    ]

    opposite = [swap_role(e) for e in examples]
    control = [swap_verb(e) for e in examples]

    donor_sets = {}

    clean_logits, clean = snapshot(model, ids, readout)
    m00 = answer_margins(clean_logits, labels, choices)

    for name, donor_examples in (
        ("opposite_role", opposite),
        ("different_verb", control),
    ):
        donor_ids, donor_readout, _, _ = [
            t.to(device) for t in encode_batch(donor_examples)
        ]

        assert donor_ids.shape == ids.shape
        assert torch.equal(donor_readout, readout)

        donor_logits, donor = snapshot(
            model, donor_ids, donor_readout
        )

        donor_margin = answer_margins(
            donor_logits, labels, choices
        )

        donor_sets[name] = (donor, donor_margin)

    eligible = (
        (m00 > 0)
        & (donor_sets["opposite_role"][1] < 0)
    )

    assert int(eligible.sum()) > 0, (
        "No eligible opposite-role cases"
    )

    pair_names = [
        "|".join(e.pair) for e in examples
    ]

    results = {}

    for name, (donor, _) in donor_sets.items():
        swapped_logits, swapped = snapshot(
            model,
            ids,
            readout,
            donor_update=donor["l1"],
        )

        predicted = {}

        for x_source, a_source in (
            ("clean", "clean"),
            ("clean", "swapped"),
            ("swapped", "clean"),
            ("swapped", "swapped"),
        ):
            logits = hybrid(
                model,
                ids,
                readout,
                clean,
                swapped,
                x_source,
                a_source,
            )

            predicted[(x_source, a_source)] = logits

        clean_error = float(
            (
                predicted[("clean", "clean")]
                - clean_logits
            ).abs().max()
        )

        swap_error = float(
            (
                predicted[("swapped", "swapped")]
                - swapped_logits
            ).abs().max()
        )

        assert clean_error < 0.003, (
            f"Clean reconstruction failed: {clean_error}"
        )

        assert swap_error < 0.003, (
            f"Swapped reconstruction failed: {swap_error}"
        )

        def margin(x, a):
            return answer_margins(
                predicted[(x, a)],
                labels,
                choices,
            )

        results[name] = {
            "reconstruction_errors": {
                "clean": clean_error,
                "swapped": swap_error,
            },
            **summarize(
                margin("clean", "clean"),
                margin("clean", "swapped"),
                margin("swapped", "clean"),
                margin("swapped", "swapped"),
                eligible,
                pair_names,
            ),
        }

    return {
        "baseline_accuracy": float(
            (m00 > 0).float().mean()
        ),
        "jointly_correct": int(eligible.sum()),
        "conditions": results,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="results/ml057_development",
    )
    parser.add_argument(
        "--out",
        default="results/ml067_path_factorial.json",
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

    device = args.device

    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    if device == "cuda":
        assert torch.cuda.is_available()

    if device == "mps":
        assert torch.backends.mps.is_available()

    torch.set_num_threads(
        min(torch.get_num_threads(), 4)
    )

    _, holdout, pairs = split_examples()

    assert len(holdout) == 224
    assert len(pairs) == 7

    manifest = json.loads(
        (
            Path(args.root) / "manifest.json"
        ).read_text()
    )

    manifest_digest = manifest.pop("sha256")

    assert hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
        ).encode()
    ).hexdigest() == manifest_digest

    holdout_digest = hashlib.sha256(
        json.dumps(
            sorted(case_key(e) for e in holdout)
        ).encode()
    ).hexdigest()

    output = Path(args.out)

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite results: {output}"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "experiment": "ML-067",
        "evaluation_status": (
            "Historical ML-037 holdout, NOT pristine"
        ),
        "step": 300,
        "manifest_sha256": manifest_digest,
        "historical_case_keys_sha256": holdout_digest,
        "factorial": (
            "L2 answer-token input-state x "
            "versus L2 attention output a"
        ),
        "seeds": args.seeds,
        "runs": [],
    }

    for seed in args.seeds:
        checkpoint = torch.load(
            Path(args.root)
            / f"seed_{seed}"
            / "checkpoint_0300.pt",
            map_location="cpu",
            weights_only=True,
        )

        assert checkpoint["seed"] == seed
        assert checkpoint["step"] == 300
        assert (
            checkpoint["split_hash"]
            == manifest_digest
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

        report["runs"].append({
            "seed": seed,
            **result,
        })

        output.write_text(
            json.dumps(report, indent=2) + "\n"
        )

        observed = result[
            "conditions"
        ]["opposite_role"]["aggregate"]

        ctrl = result[
            "conditions"
        ]["different_verb"]["aggregate"]

        print(
            f"seed={seed} "
            f"eligible={result['jointly_correct']:3d} "
            f"role_flip={observed['both_flip_rate']:.3f} "
            f"attn_only={observed['attn_only_flip_rate']:.3f} "
            f"input_only={observed['input_only_flip_rate']:.3f} "
            f"margin_drop={observed['total_margin_drop']:+.2f} "
            f"attn={observed['attn_path_margin_drop']:+.2f} "
            f"input={observed['input_path_margin_drop']:+.2f} "
            f"verb_ctrl_flip={ctrl['both_flip_rate']:.3f}",
            flush=True,
        )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
