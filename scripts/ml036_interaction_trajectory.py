
"""ML-036: Qwen voice-by-question interaction trajectory, layers 0..20.

Observe the same factorial interaction vector at every transformer-block output,
without rescaling or activation patching. We reuse the original 16 recipients.
This identifies candidate depth transitions for later causal tests; it does NOT
identify particular components or establish when an interaction was computed.
"""

import argparse
import hashlib
import json
from datetime import datetime
from itertools import product
from pathlib import Path
from statistics import mean

from ml028_event_interchange import VOICES, QUESTIONS
from ml033_causal_compatibility import recipient_cases, validate_audit
from ml035_trace_interaction import HASH as ML035_HASH

LAYERS = tuple(range(21))
RESULTS = Path("results")

SPEC = {
    "experiment": "ML-036",
    "model": "qwen7b",
    "site": "last prompt token (readout)",
    "layers": LAYERS,
    "cases": "16 fixed ML-033/035 recipients",
    "delta": "h11 - h10 - h01 + h00, same event and verb",
    "measures": [
        "norm(delta_l) / norm(recipient_l)",
        "cos(delta_l, delta_20)",
        "dot(delta_l, delta_20) / dot(delta_20, delta_20)",
        "change in projection from previous block output",
    ],
    "analysis": (
        "descriptive formation trajectory; select candidate blocks 14..20 "
        "by positive increments in projection onto delta_20"
    ),
    "source_hash": ML035_HASH,
    "no_behavior_gates_or_patches": True,
    "limitations": (
        "same previously explored stimuli; final-direction selection; "
        "projection is descriptive and not evidence of causal routing"
    ),
}

HASH = hashlib.sha256(
    json.dumps(SPEC, sort_keys=True).encode()
).hexdigest()


def source_check():
    matches = []

    for path in RESULTS.glob("ml035_qwen7b_trace_*.json"):
        data = json.loads(path.read_text())

        if (
            data.get("hash") == ML035_HASH
            and data.get("complete")
            and len(data.get("records", [])) == 384
        ):
            matches.append((path, data))

    if not matches:
        raise RuntimeError(
            "No complete ML-035 result with expected design"
        )

    return sorted(
        matches, key=lambda x: x[0].name
    )[-1]


def capture_many(
    model, tokenizer, prompt, position, encode_prompt
):
    """One forward pass, capture multiple block outputs."""

    import mlx.core as mx
    import mlx.nn as nn

    class Tap(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer
            self.use_sliding = getattr(
                layer, "use_sliding", False
            )
            self.value = None

        def __call__(self, *args, **kwargs):
            output = self.layer(*args, **kwargs)
            self.value = output[0, position, :]
            return output

    layers = model.model.layers

    if max(LAYERS) >= len(layers):
        raise RuntimeError(
            "Layer index exceeds model depth"
        )

    old = {
        idx: layers[idx]
        for idx in LAYERS
    }

    taps = {
        idx: Tap(old[idx])
        for idx in LAYERS
    }

    try:
        for idx in LAYERS:
            layers[idx] = taps[idx]

        ids = encode_prompt(tokenizer, prompt)
        tokens = mx.array(
            [ids], dtype=mx.int32
        )

        logits = model(tokens)
        mx.eval(logits)

        if any(
            taps[idx].value is None
            for idx in LAYERS
        ):
            raise RuntimeError(
                "One or more layer captures failed"
            )

        values = {
            idx: taps[idx].value
            for idx in LAYERS
        }

        mx.eval(*values.values())

    finally:
        for idx in LAYERS:
            layers[idx] = old[idx]

    return values


def norm(x):
    import mlx.core as mx

    return float(
        mx.sqrt(mx.sum(x * x))
    )


def dot(a, b):
    import mlx.core as mx

    return float(
        mx.sum(a * b)
    )


def analyze(data):
    if (
        data.get("hash") != HASH
        or not data.get("complete")
    ):
        raise RuntimeError(
            "Wrong design or incomplete ML-036 data"
        )

    rows = data.get("records", [])

    expected_count = (
        len(recipient_cases()) * len(LAYERS)
    )

    if len(rows) != expected_count:
        raise RuntimeError(
            "Incorrect number of layer/case observations"
        )

    index = {}

    for r in rows:
        key = (
            tuple(r["case"]),
            int(r["layer"]),
        )

        if key in index:
            raise RuntimeError(
                f"Duplicate observation: {key}"
            )

        index[key] = r

    expected = {
        (case, layer)
        for case in recipient_cases()
        for layer in LAYERS
    }

    if set(index) != expected:
        raise RuntimeError(
            "Missing or unexpected layer/case observations"
        )

    print(
        "\n=== ML-036 FACTORIAL INTERACTION TRAJECTORY ==="
    )

    print(
        "Descriptive: 16 nonindependent recipients, "
        "same exploratory prompts"
    )

    print(
        "layer  delta/h    cosine(L20)  "
        "projection(L20)  block increment"
    )

    for layer in LAYERS:
        rs = [
            index[case, layer]
            for case in recipient_cases()
        ]

        print(
            f"L{layer:02d}    "
            f"{mean(x['relative_delta'] for x in rs):8.4f}    "
            f"{mean(x['cos_final'] for x in rs):+9.4f}    "
            f"{mean(x['fraction_final'] for x in rs):+11.4f}    "
            f"{mean(x['increment_final'] for x in rs):+11.4f}"
        )

    candidates = sorted(
        (
            (
                layer,
                mean(
                    index[case, layer]["increment_final"]
                    for case in recipient_cases()
                ),
            )
            for layer in range(14, 21)
        ),
        key=lambda x: x[1],
        reverse=True,
    )

    print(
        "\nCandidate block outputs (14..20), "
        "ranked by projection increment:"
    )

    for layer, value in candidates:
        cases_pos = sum(
            index[case, layer]["increment_final"] > 0
            for case in recipient_cases()
        )

        print(
            f"  L{layer:02d}: "
            f"mean increment={value:+.4f}, "
            f"positive={cases_pos}/16"
        )

    print(
        "\nThis is NOT causal localization. "
        "It nominates blocks for ML-037."
    )


def collect():
    from mlx_lm import load
    from ml003_role_margins import MODELS
    from ml012_cue_patch import encode_prompt

    source_path, old = source_check()

    print(
        "Historical source:",
        source_path
    )

    model, tokenizer = load(
        str(MODELS / "qwen7b")
    )

    prepared = validate_audit(
        tokenizer, encode_prompt
    )

    print(
        "Audited prompts:",
        len(prepared)
    )

    print(
        "Available blocks:",
        len(model.model.layers)
    )

    if max(LAYERS) >= len(model.model.layers):
        raise RuntimeError(
            "Model has too few layers"
        )

    # Capture each distinct prompt once, at all 21 layers.
    state = {}

    for key, (prompt, position) in prepared.items():
        state[key] = capture_many(
            model,
            tokenizer,
            prompt,
            position,
            encode_prompt,
        )

        print(
            "CAPTURED",
            key,
            f"{len(state)}/32"
        )

    historical = {}

    for r in old["records"]:
        if (
            r["kind"] == "true"
            and int(r["sign"]) == 1
        ):
            case = tuple(r["case"])

            historical[
                case, int(r["layer"])
            ] = float(r["delta_over_h"])

    RESULTS.mkdir(exist_ok=True)

    path = RESULTS / (
        "ml036_qwen7b_trajectory_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    data = {
        "hash": HASH,
        "complete": False,
        "model": "qwen7b",
        "source": str(source_path),
        "cases": recipient_cases(),
        "records": [],
    }

    for case in recipient_cases():
        pid, verb, direction, voice, role = case

        flip_voice = next(
            x for x in VOICES
            if x != voice
        )

        flip_role = next(
            x for x in QUESTIONS
            if x != role
        )

        ds = {}
        norms = {}

        for layer in LAYERS:
            h00 = state[
                pid, verb, direction, voice, role
            ][layer]

            h10 = state[
                pid, verb, direction, flip_voice, role
            ][layer]

            h01 = state[
                pid, verb, direction, voice, flip_role
            ][layer]

            h11 = state[
                pid, verb, direction, flip_voice, flip_role
            ][layer]

            ds[layer] = (
                h11 - h10 - h01 + h00
            )

            norms[layer] = norm(h00)

        final = ds[20]
        final_sq = dot(final, final)

        if final_sq <= 1e-12:
            raise RuntimeError(
                f"Vanishing L20 interaction: {case}"
            )

        previous_fraction = 0.0

        for layer in LAYERS:
            dnorm = norm(ds[layer])

            if norms[layer] <= 1e-12:
                raise RuntimeError(
                    f"Vanishing residual norm: {case} L{layer}"
                )

            relative = (
                dnorm / norms[layer]
            )

            fraction = (
                dot(ds[layer], final)
                / final_sq
            )

            cosine = (
                dot(ds[layer], final)
                / max(
                    dnorm * (final_sq ** 0.5),
                    1e-12,
                )
            )

            increment = (
                fraction - previous_fraction
            )

            previous_fraction = fraction

            prior = historical.get(
                (case, layer)
            )

            if (
                prior is not None
                and abs(relative - prior) > 0.015
            ):
                raise RuntimeError(
                    f"ML-035 norm mismatch at {case} "
                    f"L{layer}: "
                    f"{relative:.5f} versus {prior:.5f}"
                )

            data["records"].append({
                "case": list(case),
                "layer": layer,
                "delta_norm": dnorm,
                "recipient_norm": norms[layer],
                "relative_delta": relative,
                "cos_final": cosine,
                "fraction_final": fraction,
                "increment_final": increment,
            })

        print(
            "ANALYZED",
            case,
            f"{len(data['records'])}/336"
        )

        path.write_text(
            json.dumps(data, indent=2)
        )

    data["complete"] = True

    path.write_text(
        json.dumps(data, indent=2)
    )

    print(
        "Saved:",
        path
    )

    analyze(data)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=(
            "design",
            "audit",
            "collect",
            "summarize",
        ),
        default="design",
    )

    args = parser.parse_args()

    print(
        "ML-036",
        args.phase,
        "qwen7b",
        HASH,
    )

    if args.phase == "design":
        print(
            json.dumps(SPEC, indent=2)
        )
        return

    source_path, _ = source_check()

    print(
        "Historical ML-035:",
        source_path
    )

    if args.phase == "summarize":
        paths = sorted(
            RESULTS.glob(
                "ml036_qwen7b_trajectory_*.json"
            ),
            reverse=True,
        )

        for path in paths:
            data = json.loads(
                path.read_text()
            )

            if (
                data.get("hash") == HASH
                and data.get("complete")
            ):
                print(
                    "Reading:",
                    path
                )

                analyze(data)
                return

        raise RuntimeError(
            "No complete matching ML-036 results"
        )

    if args.phase == "audit":
        from mlx_lm import load
        from ml003_role_margins import MODELS
        from ml012_cue_patch import encode_prompt

        model, tokenizer = load(
            str(MODELS / "qwen7b")
        )

        prompts = validate_audit(
            tokenizer,
            encode_prompt,
        )

        print(
            "Audited prompts:",
            len(prompts)
        )

        print(
            "Available blocks:",
            len(model.model.layers)
        )

        if max(LAYERS) >= len(model.model.layers):
            raise RuntimeError(
                "Selected layer does not exist"
            )

        return

    collect()


if __name__ == "__main__":
    main()
