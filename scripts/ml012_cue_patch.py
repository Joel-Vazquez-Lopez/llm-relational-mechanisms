
import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from ml003_role_margins import MODELS, log_likelihood


NAMES = ("Leo", "Felix")


class CaptureLayer(nn.Module):
    def __init__(self, layer, position):
        super().__init__()
        self.layer = layer
        self.use_sliding = layer.use_sliding  
        self.position = position
        self.value = None 

    def __call__(self, *args, **kwargs):
        output = self.layer(*args, **kwargs)
        self.value = output[0, self.position, :]
        return output


class PatchLayer(nn.Module):
    def __init__(self, layer, position, replacement):
        super().__init__()
        self.layer = layer
        self.use_sliding = layer.use_sliding  
        self.position = position
        self.replacement = replacement


    def __call__(self, *args, **kwargs):
        output = self.layer(*args, **kwargs)
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


def encode_prompt(tokenizer, prompt):
    bos = tokenizer.bos_token
    add_special = bos is None or not prompt.startswith(bos)

    return tokenizer.encode(
        prompt,
        add_special_tokens=add_special,
    )


def find_single_change(tokenizer, original, alternative):
    a = encode_prompt(tokenizer, original)
    b = encode_prompt(tokenizer, alternative)

    changes = [
        change for change in SequenceMatcher(
            None, a, b, autojunk=False
        ).get_opcodes()
        if change[0] != "equal"
    ]

    print("Token differences:", changes)

    for tag, i, j, k, l in changes:
        print(
            "Original:", tokenizer.decode(a[i:j]),
            "| Alternative:", tokenizer.decode(b[k:l]),
        )

    if (
        len(changes) != 1
        or changes[0][0] != "replace"
        or changes[0][2] - changes[0][1] != 1
        or changes[0][4] - changes[0][3] != 1
        or len(a) != len(b)
    ):
        raise ValueError(
            "The two prompts are not aligned by one "
            "replaced token. No intervention performed."
        )

    _, start, _, other_start, _ = changes[0]
    assert start == other_start

    original_word = tokenizer.decode([a[start]]).strip()
    alternative_word = tokenizer.decode([b[start]]).strip()

    if (original_word, alternative_word) != (
        "wrote", "carried"
    ):
        raise ValueError(
            "The changed token is not the intended "
            "dependency word."
        )

    return start


def capture(model, tokenizer, prompt, layer_index, position):
    layers = model.model.layers
    original = layers[layer_index]
    wrapper = CaptureLayer(original, position)

    layers[layer_index] = wrapper

    try:
        ids = encode_prompt(tokenizer, prompt)
        tokens = mx.array([ids], dtype=mx.int32)

        logits = model(tokens)
        mx.eval(logits)

        if wrapper.value is None:
            raise RuntimeError("Layer capture failed")

        mx.eval(wrapper.value)
        result = wrapper.value

    finally:
        layers[layer_index] = original

    return result


def margin(model, tokenizer, prompt):
    scores = {}

    for name in NAMES:
        lp, _ = log_likelihood(
            model, tokenizer, prompt, name
        )
        scores[name] = lp

    return scores["Leo"] - scores["Felix"]


def patched_margin(
    model, tokenizer, prompt, layer_index,
    position, replacement
):
    layers = model.model.layers
    original = layers[layer_index]

    layers[layer_index] = PatchLayer(
        original, position, replacement
    )

    try:
        result = margin(model, tokenizer, prompt)
    finally:
        layers[layer_index] = original

    return result


def find_record(records, link, probe):
    matches = [
        r for r in records
        if r["writer"] == "Leo"
        and r["carrier"] == "Felix"
        and r["order"] == "bridge_first"
        and r["link"] == link
        and r["probe"] == probe
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one {link}/{probe} record"
        )

    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument(
    "--site",
    choices=("cue", "readout"),
    default="cue",
    )
    args = parser.parse_args()

    paths = sorted(
        Path("results").glob("ml011_llama3b_*.json")
    )
    if not paths:
        raise FileNotFoundError("ML-011 results missing")

    source = paths[-1]
    data = json.loads(source.read_text())

    print("Source:", source)
    print("Loading Llama 3B")

    model, tokenizer = load(
        str(MODELS / "llama3b")
    )

    n_layers = len(model.model.layers)

    if not 0 <= args.layer < n_layers:
        raise ValueError("Invalid layer index")

    print("Transformer layers:", n_layers)
    print("Selected layer:", args.layer)

    for probe in ("event", "writer"):
        original = find_record(
            data["records"], "wrote", probe
        )
        alternative = find_record(
            data["records"], "carried", probe
        )

        prompt_a = original["prompt"]
        prompt_b = alternative["prompt"]

        cue_pos = find_single_change(
            tokenizer, prompt_a, prompt_b
        )

        if args.site == "cue":
            pos = cue_pos
        else:
            ids_a = encode_prompt(tokenizer, prompt_a)
            ids_b = encode_prompt(tokenizer, prompt_b)

            assert len(ids_a) == len(ids_b)
            assert ids_a[-1] == ids_b[-1]

            # Final prompt token, immediately before the answer.
            pos = len(ids_a) - 1

        print(f"\n=== {probe.upper()} QUESTION ===")
        print("Patch site:", args.site)
        print("Patch position:", pos)
        
        baseline = margin(
                    model, tokenizer, prompt_a
        )
        counterfactual = margin(
            model, tokenizer, prompt_b
        )

        source_state = capture(
            model, tokenizer, prompt_b,
            args.layer, pos
        )

        own_state = capture(
            model, tokenizer, prompt_a,
            args.layer, pos
        )

        self_patch = patched_margin(
            model, tokenizer, prompt_a,
            args.layer, pos, own_state
        )

        intervention = patched_margin(
            model, tokenizer, prompt_a,
            args.layer, pos, source_state
        )

        print(f"Original margin:       {baseline:+.4f}")
        print(f"Counterfactual margin: {counterfactual:+.4f}")
        print(f"Self-patch margin:     {self_patch:+.4f}")
        print(f"Intervention margin:   {intervention:+.4f}")

        print(
            "Self-patch error:",
            f"{self_patch - baseline:+.4f}"
        )
        print(
            "Causal patch effect:",
            f"{intervention - baseline:+.4f}"
        )

        if abs(self_patch - baseline) > 0.05:
            raise RuntimeError(
                "Self-patch parity failed; do not interpret"
            )

        print(
            "Counterfactual behavioral effect:",
            f"{counterfactual - baseline:+.4f}"
        )


if __name__ == "__main__":
    main()
