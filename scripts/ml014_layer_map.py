
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from ml003_role_margins import MODELS
from ml012_cue_patch import (
    encode_prompt,
    find_record,
    find_single_change,
    margin,
    patched_margin,
)
from ml013_position_probe import locate_name


class CaptureAllLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.use_sliding = layer.use_sliding
        self.states = None

    def __call__(self, *args, **kwargs):
        output = self.layer(*args, **kwargs)
        self.states = output[0, :, :]
        return output


def capture_states(model, tokenizer, prompt, layer_index):
    layers = model.model.layers
    original = layers[layer_index]
    wrapper = CaptureAllLayer(original)

    layers[layer_index] = wrapper

    try:
        ids = encode_prompt(tokenizer, prompt)
        tokens = mx.array([ids], dtype=mx.int32)
        output = model(tokens)
        mx.eval(output)

        if wrapper.states is None:
            raise RuntimeError("Capture failed")

        mx.eval(wrapper.states)
        states = wrapper.states

    finally:
        layers[layer_index] = original

    return states


def get_sites(tokenizer, prompt_a, prompt_b):
    a = encode_prompt(tokenizer, prompt_a)
    b = encode_prompt(tokenizer, prompt_b)

    cue = find_single_change(tokenizer, prompt_a, prompt_b)
    assert len(a) == len(b)

    sites = [("relation_word", cue)]

    for name in ("Leo", "Felix", "Nora", "Clara"):
        pos = locate_name(tokenizer, a, name)
        if pos is not None:
            sites.append((name, pos))

    sites.append(("final_prompt", len(a) - 1))
    return sites


def save_csv(path, records):
    fields = [
        "layer", "site", "position",
        "baseline", "counterfactual",
        "patched_margin", "effect",
        "full_effect_fraction", "self_error",
    ]

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def make_plot(path, records, sites, n_layers):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Matplotlib unavailable; CSV and JSON saved.")
        return

    site_names = [name for name, _ in sites]
    matrix = np.full((len(sites), n_layers), np.nan)

    for record in records:
        row = site_names.index(record["site"])
        matrix[row, record["layer"]] = record["effect"]

    finite = matrix[np.isfinite(matrix)]
    scale = max(float(np.max(np.abs(finite))), 0.01)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-scale,
        vmax=scale,
        interpolation="nearest",
    )

    ax.set_yticks(range(len(site_names)))
    ax.set_yticklabels(site_names)
    ax.set_xticks(range(0, n_layers, 2))
    ax.set_xlabel("Transformer layer (0-indexed)")
    ax.set_title("Causal patch effect: Leo minus Felix margin")

    fig.colorbar(
        im,
        ax=ax,
        label="Margin change (negative = toward Felix)",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("Heatmap saved:", path)


def main():
    parser = argparse.ArgumentParser()
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
        raise FileNotFoundError("ML-011 results missing")

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
    print("Probe:", args.probe)
    print("Loading Llama 3B")

    model, tokenizer = load(str(MODELS / "llama3b"))
    n_layers = len(model.model.layers)

    sites = get_sites(tokenizer, prompt_a, prompt_b)

    baseline = float(margin(model, tokenizer, prompt_a))
    counterfactual = float(margin(model, tokenizer, prompt_b))
    full_effect = counterfactual - baseline

    print("Layers:", n_layers)
    print("Sites:", sites)
    print("Original margin:", baseline)
    print("Counterfactual margin:", counterfactual)
    print("Full input-edit effect:", full_effect)

    if abs(full_effect) < 0.01:
        raise RuntimeError("Input edit has negligible effect")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    stem = (
        f"ml014_{args.probe}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}"
    )

    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}.png"

    records = []
    check_layers = {0, 7, 14, n_layers - 1}

    for layer_index in range(n_layers):
        donor_states = capture_states(
            model, tokenizer, prompt_b, layer_index
        )

        own_states = None
        if layer_index in check_layers:
            own_states = capture_states(
                model, tokenizer, prompt_a, layer_index
            )

        print(f"\n=== LAYER {layer_index} ===")

        for label, position in sites:
            donor = donor_states[position, :]

            patched = float(
                patched_margin(
                    model, tokenizer, prompt_a,
                    layer_index, position, donor
                )
            )

            effect = patched - baseline
            self_error = None

            if own_states is not None:
                own = own_states[position, :]
                self_value = float(
                    patched_margin(
                        model, tokenizer, prompt_a,
                        layer_index, position, own
                    )
                )
                self_error = self_value - baseline

                if abs(self_error) > 0.05:
                    raise RuntimeError(
                        f"Self-patch failed: "
                        f"layer={layer_index}, site={label}"
                    )

            record = {
                "layer": layer_index,
                "site": label,
                "position": position,
                "baseline": baseline,
                "counterfactual": counterfactual,
                "patched_margin": patched,
                "effect": effect,
                "full_effect_fraction": effect / full_effect,
                "self_error": self_error,
            }
            records.append(record)

            print(
                f"{label:15s} "
                f"effect={effect:+.4f} "
                f"fraction={effect/full_effect:+.3f}"
            )

        # Save completed layers to preserve partial results.
        json_path.write_text(
            json.dumps({
                "experiment": "ML-014",
                "model": "llama3b",
                "probe": args.probe,
                "source": str(source),
                "baseline": baseline,
                "counterfactual": counterfactual,
                "full_effect": full_effect,
                "records": records,
            }, indent=2),
            encoding="utf-8",
        )
        save_csv(csv_path, records)

    make_plot(png_path, records, sites, n_layers)

    print("\nFinished all layers.")
    print("JSON:", json_path)
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
