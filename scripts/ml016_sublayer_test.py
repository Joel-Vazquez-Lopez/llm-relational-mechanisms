
import argparse
import json
from datetime import datetime
from itertools import product
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from ml003_role_margins import MODELS
from ml012_cue_patch import (
    capture,
    encode_prompt,
    find_single_change,
    margin,
    patched_margin,
)


class CaptureModule(nn.Module):
    def __init__(self, inner, position):
        super().__init__()
        self.inner = inner
        self.position = position
        self.value = None

    def __call__(self, *args, **kwargs):
        output = self.inner(*args, **kwargs)
        self.value = output[0, self.position, :]
        return output


class PatchModule(nn.Module):
    def __init__(self, inner, position, replacement):
        super().__init__()
        self.inner = inner
        self.position = position
        self.replacement = replacement

    def __call__(self, *args, **kwargs):
        output = self.inner(*args, **kwargs)
        pos = self.position

        return mx.concatenate(
            [
                output[:, :pos, :],
                self.replacement[None, None, :].astype(
                    output.dtype
                ),
                output[:, pos + 1:, :],
            ],
            axis=1,
        )


def capture_component(
    model, tokenizer, prompt, layer_index,
    component, position
):
    layer = model.model.layers[layer_index]
    original = getattr(layer, component)
    wrapper = CaptureModule(original, position)

    setattr(layer, component, wrapper)

    try:
        ids = encode_prompt(tokenizer, prompt)
        tokens = mx.array([ids], dtype=mx.int32)

        output = model(tokens)

        if wrapper.value is None:
            raise RuntimeError("Capture failed")

        mx.eval(output, wrapper.value)
        state = wrapper.value

    finally:
        setattr(layer, component, original)

    return state


def patch_component(
    model, tokenizer, prompt, layer_index,
    component, position, replacement
):
    layer = model.model.layers[layer_index]
    original = getattr(layer, component)

    wrapper = PatchModule(
        original, position, replacement
    )

    setattr(layer, component, wrapper)

    try:
        result = float(margin(model, tokenizer, prompt))
    finally:
        setattr(layer, component, original)

    return result


def record_for(
    records, writer, carrier, order, link, probe
):
    matches = [
        record for record in records
        if record["writer"] == writer
        and record["carrier"] == carrier
        and record["order"] == order
        and record["link"] == link
        and record["probe"] == probe
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one record: "
            f"{writer}/{carrier}/{order}/{link}/{probe}"
        )

    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    paths = sorted(
        Path("results").glob("ml011_llama3b_*.json")
    )
    if not paths:
        raise FileNotFoundError("ML-011 results missing")

    source = paths[-1]
    records = json.loads(source.read_text())["records"]

    model, tokenizer = load(str(MODELS / "llama3b"))

    layers = [12, 13, 16] if args.full else [13]

    assignments = (
        [("Leo", "Felix"), ("Felix", "Leo")]
        if args.full else [("Leo", "Felix")]
    )

    orders = (
        ["bridge_first", "facts_first"]
        if args.full else ["bridge_first"]
    )

    probes = ["event", "writer"]

    output_path = Path("results") / (
        "ml016_sublayer_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    results = []

    for (writer, carrier), order, probe, layer_index in product(
        assignments, orders, probes, layers
    ):
        a = record_for(
            records, writer, carrier,
            order, "wrote", probe
        )
        b = record_for(
            records, writer, carrier,
            order, "carried", probe
        )

        prompt_a = a["prompt"]
        prompt_b = b["prompt"]

        find_single_change(
            tokenizer, prompt_a, prompt_b
        )

        ids_a = encode_prompt(tokenizer, prompt_a)
        ids_b = encode_prompt(tokenizer, prompt_b)

        if len(ids_a) != len(ids_b):
            raise RuntimeError("Token lengths differ")

        position = len(ids_a) - 1
        direction = 1 if writer == "Leo" else -1

        baseline = float(
            margin(model, tokenizer, prompt_a)
        )
        counterfactual = float(
            margin(model, tokenizer, prompt_b)
        )

        full_effect = direction * (
            counterfactual - baseline
        )

        print(
            f"\n=== L{layer_index} "
            f"{writer}/{carrier} {order} {probe} ==="
        )
        print(f"Full effect: {full_effect:+.4f}")

        row = {
            "writer": writer,
            "carrier": carrier,
            "order": order,
            "probe": probe,
            "layer": layer_index,
            "full_effect": full_effect,
        }

        for label, component in (
            ("attention", "self_attn"),
            ("mlp", "mlp"),
        ):
            donor = capture_component(
                model, tokenizer, prompt_b,
                layer_index, component, position
            )

            own = capture_component(
                model, tokenizer, prompt_a,
                layer_index, component, position
            )

            self_value = patch_component(
                model, tokenizer, prompt_a,
                layer_index, component, position, own
            )

            if abs(self_value - baseline) > 0.05:
                raise RuntimeError(
                    f"Self-patch parity failure: "
                    f"{label}, layer {layer_index}"
                )

            patched = patch_component(
                model, tokenizer, prompt_a,
                layer_index, component, position, donor
            )

            effect = direction * (patched - baseline)

            row[label] = {
                "effect": effect,
                "self_error": self_value - baseline,
                "donor_delta_norm": float(
                    mx.sqrt(mx.sum((donor - own) ** 2))
                ),
            }

            print(
                f"{label:10s} "
                f"effect={effect:+.4f}"
            )

        donor_block = capture(
            model, tokenizer, prompt_b,
            layer_index, position
        )

        whole = float(
            patched_margin(
                model, tokenizer, prompt_a,
                layer_index, position, donor_block
            )
        )

        whole_effect = direction * (whole - baseline)
        row["whole_block_effect"] = whole_effect

        print(
            f"{'whole block':10s} "
            f"effect={whole_effect:+.4f}"
        )

        results.append(row)

        output_path.write_text(
            json.dumps({
                "experiment": "ML-016",
                "source": str(source),
                "records": results,
            }, indent=2),
            encoding="utf-8",
        )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
