
import json
from datetime import datetime
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from ml003_role_margins import MODELS
from ml012_cue_patch import (
    capture,
    find_single_change,
    margin,
    patched_margin,
)
from ml016_sublayer_test import (
    capture_component,
    patch_component,
)
from ml017_relevance import make_prompt


DONOR = ("Felix", "Leo", "bridge_first", "sulmed")

RECIPIENTS = [
    ("same_context", "Leo", "Felix",
     "bridge_first", "sulmed"),

    ("new_surface", "Leo", "Felix",
     "facts_first", "velked"),

    ("reverse_names", "Felix", "Leo",
     "bridge_first", "sulmed"),

    ("reverse_names_surface", "Felix", "Leo",
     "facts_first", "velked"),
]

SITES = [
    ("attention_L13", 13, "attention"),
    ("block_L16", 16, "block"),
]

DONOR_CONDITIONS = [
    "event_relevant",
    "event_irrelevant",
]


def prompts(tokenizer, config, condition):
    writer, carrier, order, verb = config

    a = make_prompt(
        tokenizer, writer, carrier, order,
        verb, condition, "wrote"
    )
    b = make_prompt(
        tokenizer, writer, carrier, order,
        verb, condition, "carried"
    )

    find_single_change(tokenizer, a, b)
    return a, b


def state_at(model, tokenizer, prompt, layer, kind):
    from ml012_cue_patch import encode_prompt

    position = len(encode_prompt(tokenizer, prompt)) - 1

    if kind == "attention":
        return capture_component(
            model, tokenizer, prompt,
            layer, "self_attn", position
        )

    return capture(
        model, tokenizer, prompt, layer, position
    )


def apply_state(
    model, tokenizer, prompt, layer, kind, state
):
    from ml012_cue_patch import encode_prompt

    position = len(encode_prompt(tokenizer, prompt)) - 1

    if kind == "attention":
        return float(patch_component(
            model, tokenizer, prompt,
            layer, "self_attn", position, state
        ))

    return float(patched_margin(
        model, tokenizer, prompt,
        layer, position, state
    ))


def length(vector):
    return float(
        mx.sqrt(mx.sum(vector * vector))
    )


def main():
    print("Loading Llama 3B")

    model, tokenizer = load(
        str(MODELS / "llama3b")
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = output_dir / (
        "ml018_cross_context_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    rows = []

    for site_label, layer, kind in SITES:
        print(f"\n=== {site_label} ===")

        donor_deltas = {}

        for condition in DONOR_CONDITIONS:
            a, b = prompts(
                tokenizer, DONOR, condition
            )

            state_a = state_at(
                model, tokenizer, a, layer, kind
            )
            state_b = state_at(
                model, tokenizer, b, layer, kind
            )

            delta = state_b - state_a
            mx.eval(delta)

            donor_deltas[condition] = delta

            print(
                f"Donor {condition:18s} "
                f"delta_norm={length(delta):.3f}"
            )

        for (
            label, writer, carrier, order, verb
        ) in RECIPIENTS:

            config = (writer, carrier, order, verb)

            a, b = prompts(
                tokenizer, config, "event_relevant"
            )

            sign = 1 if writer == "Leo" else -1

            baseline_raw = float(
                margin(model, tokenizer, a)
            )
            counterfactual_raw = float(
                margin(model, tokenizer, b)
            )

            base = sign * baseline_raw
            full_effect = sign * (
                counterfactual_raw - baseline_raw
            )

            own = state_at(
                model, tokenizer, a, layer, kind
            )
            own_b = state_at(
                model, tokenizer, b, layer, kind
            )

            # Technical self-patch validation.
            self_raw = apply_state(
                model, tokenizer, a,
                layer, kind, own
            )
            self_error = self_raw - baseline_raw

            if abs(self_error) > 0.05:
                raise RuntimeError(
                    f"Self-patch failure: "
                    f"{site_label} / {label}"
                )

            # Within-context positive control.
            local_delta = own_b - own
            mx.eval(local_delta)

            local_raw = apply_state(
                model, tokenizer, a,
                layer, kind, own + local_delta
            )

            local_effect = sign * (
                local_raw - baseline_raw
            )

            print(
                f"\n{label} | writer={writer} "
                f"carrier={carrier}"
            )
            print(
                f"Full input effect: {full_effect:+.4f}"
            )
            print(
                f"Local patch:       {local_effect:+.4f}"
            )

            row = {
                "site": site_label,
                "writer": writer,
                "carrier": carrier,
                "order": order,
                "verb": verb,
                "recipient": label,
                "baseline_margin": base,
                "full_effect": full_effect,
                "local_patch_effect": local_effect,
                "self_error": self_error,
                "transfer_effects": {},
            }

            for condition, delta in donor_deltas.items():

                patched = own + delta
                mx.eval(patched)

                raw = apply_state(
                    model, tokenizer, a,
                    layer, kind, patched
                )

                effect = sign * (
                    raw - baseline_raw
                )

                row["transfer_effects"][condition] = effect

                print(
                    f"{condition:18s} "
                    f"transfer={effect:+.4f}"
                )

            # Directional control: invert the
            # relation-relevant donor difference.
            reverse_delta = (
                -donor_deltas["event_relevant"]
            )

            reverse_raw = apply_state(
                model, tokenizer, a,
                layer, kind,
                own + reverse_delta
            )

            reverse_effect = sign * (
                reverse_raw - baseline_raw
            )

            row["reverse_delta_effect"] = reverse_effect

            print(
                "reverse_delta       "
                f"transfer={reverse_effect:+.4f}"
            )

            rows.append(row)

            output.write_text(
                json.dumps({
                    "experiment": "ML-018",
                    "model": "llama3b",
                    "donor": DONOR,
                    "records": rows,
                }, indent=2),
                encoding="utf-8",
            )

    print("\n=== SUMMARY ===")

    for site_label, _, _ in SITES:
        group = [
            r for r in rows
            if r["site"] == site_label
        ]

        print(f"\n{site_label}")

        for r in group:
            effect = r["transfer_effects"]["event_relevant"]
            control = r["transfer_effects"]["event_irrelevant"]

            print(
                f"{r['recipient']:22s} "
                f"local={r['local_patch_effect']:+.3f} "
                f"relevant={effect:+.3f} "
                f"irrelevant={control:+.3f}"
            )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
