
import argparse
import json
from datetime import datetime
from pathlib import Path

from mlx_lm import load
from ml003_role_margins import MODELS
from ml012_cue_patch import (
    capture,
    encode_prompt,
    find_single_change,
    find_record,
    margin,
    patched_margin,
)


def locate_name(tokenizer, token_ids, name):
    pattern = tokenizer.encode(
        " " + name,
        add_special_tokens=False,
    )

    matches = [
        i
        for i in range(len(token_ids) - len(pattern) + 1)
        if token_ids[i:i + len(pattern)] == pattern
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Cannot uniquely locate {name}: "
            f"{len(matches)} matches. Inspect tokenization."
        )

    if len(pattern) != 1:
        print(
            f"Skipping {name}: tokenized as "
            f"{len(pattern)} tokens"
        )
        return None

    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer", type=int, default=14
    )
    parser.add_argument(
        "--probe",
        choices=("event", "writer"),
        default="event",
    )
    args = parser.parse_args()

    paths = sorted(
        Path("results").glob("ml011_llama3b_*.json")
    )
    if not paths:
        raise FileNotFoundError(
            "No saved ML-011 results found"
        )

    source = paths[-1]
    data = json.loads(source.read_text())

    record_a = find_record(
        data["records"], "wrote", args.probe
    )
    record_b = find_record(
        data["records"], "carried", args.probe
    )

    prompt_a = record_a["prompt"]
    prompt_b = record_b["prompt"]

    print("Source:", source)
    print("Loading Llama 3B")
    model, tokenizer = load(
        str(MODELS / "llama3b")
    )

    if not 0 <= args.layer < len(model.model.layers):
        raise ValueError("Invalid layer index")

    ids_a = encode_prompt(tokenizer, prompt_a)
    ids_b = encode_prompt(tokenizer, prompt_b)

    cue_pos = find_single_change(
        tokenizer, prompt_a, prompt_b
    )

    assert len(ids_a) == len(ids_b)

    sites = [("relation_word", cue_pos)]

    for name in ("Leo", "Felix", "Nora", "Clara"):
        position = locate_name(tokenizer, ids_a, name)
        if position is not None:
            sites.append((name, position))

    sites.append(("final_prompt", len(ids_a) - 1))

    baseline = float(
        margin(model, tokenizer, prompt_a)
    )
    counterfactual = float(
        margin(model, tokenizer, prompt_b)
    )

    print("\n=== BASELINES ===")
    print("Layer:", args.layer)
    print("Question:", args.probe)
    print("Original:", f"{baseline:+.4f}")
    print("Counterfactual:", f"{counterfactual:+.4f}")
    print(
        "Full input-edit effect:",
        f"{counterfactual - baseline:+.4f}"
    )

    records = []

    print("\n=== SITE INTERVENTIONS ===")

    for label, pos in sites:
        donor = capture(
            model, tokenizer, prompt_b,
            args.layer, pos
        )
        own = capture(
            model, tokenizer, prompt_a,
            args.layer, pos
        )

        self_value = float(
            patched_margin(
                model, tokenizer, prompt_a,
                args.layer, pos, own
            )
        )

        patched = float(
            patched_margin(
                model, tokenizer, prompt_a,
                args.layer, pos, donor
            )
        )

        self_error = self_value - baseline
        effect = patched - baseline

        print(
            f"{label:15s} "
            f"pos={pos:3d} "
            f"effect={effect:+.4f} "
            f"self_error={self_error:+.4f}"
        )

        if abs(self_error) > 0.05:
            raise RuntimeError(
                f"Self-patch parity failed at {label}"
            )

        records.append({
            "site": label,
            "position": pos,
            "baseline": baseline,
            "counterfactual": counterfactual,
            "self_patch": self_value,
            "patched_margin": patched,
            "self_error": self_error,
            "patch_effect": effect,
        })

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = output_dir / (
        f"ml013_{args.probe}_layer{args.layer}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    output.write_text(
        json.dumps({
            "experiment": "ML-013",
            "source": str(source),
            "model": "llama3b",
            "probe": args.probe,
            "layer": args.layer,
            "records": records,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
