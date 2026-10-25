
import argparse
import json
import statistics
from datetime import datetime
from itertools import product
from pathlib import Path

from mlx_lm import load

from ml003_role_margins import MODELS
from ml012_cue_patch import (
    encode_prompt,
    find_single_change,
    margin,
    capture,
    patched_margin,
)
from ml016_sublayer_test import (
    capture_component,
    patch_component,
)


CONDITIONS = (
    "event_relevant",
    "watcher_relevant",
    "event_irrelevant",
)

SITES = (
    ("attention_L13", 13, "attention"),
    ("block_L16", 16, "block"),
)


def make_prompt(
    tokenizer, writer, carrier, order,
    verb, condition, link
):
    facts = (
        f"{writer} wrote the letter. "
        f"{carrier} carried the letter. "
        "Nora prepared a note. Clara waited."
    )

    if condition == "event_relevant":
        event = (
            f"At dawn, the person who {link} "
            f"the letter {verb} Mira."
        )
        question = f"Who {verb} Mira?"

    else:
        event = (
            f"At dawn, {writer} {verb} Mira. "
            f"The person who {link} the letter "
            "watched the event."
        )

        if condition == "watcher_relevant":
            question = "Who watched the event?"
        else:
            question = f"Who {verb} Mira?"

    if order == "bridge_first":
        passage = f"{event} {facts}"
    else:
        passage = f"{facts} {event}"

    content = (
        "Answer with exactly one person's name.\n\n"
        f"Passage: {passage}\n"
        f"Question: {question}\n"
        "Answer:"
    )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def get_state(model, tokenizer, prompt, layer, kind, position):
    if kind == "attention":
        return capture_component(
            model, tokenizer, prompt,
            layer, "self_attn", position
        )

    return capture(
        model, tokenizer, prompt, layer, position
    )


def apply_patch(
    model, tokenizer, prompt,
    layer, kind, position, state
):
    if kind == "attention":
        return float(patch_component(
            model, tokenizer, prompt,
            layer, "self_attn", position, state
        ))

    return float(patched_margin(
        model, tokenizer, prompt,
        layer, position, state
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    assignments = (
        [("Leo", "Felix"), ("Felix", "Leo")]
        if args.full else [("Leo", "Felix")]
    )

    orders = (
        ["bridge_first", "facts_first"]
        if args.full else ["bridge_first"]
    )

    verbs = (
        ["sulmed", "velked"]
        if args.full else ["sulmed"]
    )

    print("Loading Llama 3B")
    model, tokenizer = load(
        str(MODELS / "llama3b")
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = output_dir / (
        "ml017_relevance_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    rows = []

    for (writer, carrier), order, verb, condition in product(
        assignments, orders, verbs, CONDITIONS
    ):
        a = make_prompt(
            tokenizer, writer, carrier,
            order, verb, condition, "wrote"
        )
        b = make_prompt(
            tokenizer, writer, carrier,
            order, verb, condition, "carried"
        )

        find_single_change(tokenizer, a, b)

        ids_a = encode_prompt(tokenizer, a)
        ids_b = encode_prompt(tokenizer, b)

        if len(ids_a) != len(ids_b):
            raise RuntimeError(
                "Counterfactual prompts are misaligned"
            )

        position = len(ids_a) - 1

        # Express every margin as writer minus carrier.
        direction = 1 if writer == "Leo" else -1

        base_raw = float(margin(model, tokenizer, a))
        donor_raw = float(margin(model, tokenizer, b))

        base = direction * base_raw
        donor_margin = direction * donor_raw
        full_effect = donor_margin - base

        row = {
            "writer": writer,
            "carrier": carrier,
            "order": order,
            "verb": verb,
            "condition": condition,
            "base_margin": base,
            "donor_margin": donor_margin,
            "full_effect": full_effect,
            "patches": {},
        }

        print(
            f"\n=== {condition} | {writer}/{carrier} "
            f"| {order} | {verb} ==="
        )
        print("Full effect:", f"{full_effect:+.4f}")

        for label, layer, kind in SITES:
            donor_state = get_state(
                model, tokenizer, b,
                layer, kind, position
            )
            own_state = get_state(
                model, tokenizer, a,
                layer, kind, position
            )

            self_raw = apply_patch(
                model, tokenizer, a,
                layer, kind, position, own_state
            )
            patched_raw = apply_patch(
                model, tokenizer, a,
                layer, kind, position, donor_state
            )

            self_error = self_raw - base_raw

            if abs(self_error) > 0.05:
                raise RuntimeError(
                    f"Self-patch parity failed: {label}"
                )

            effect = direction * (
                patched_raw - base_raw
            )

            row["patches"][label] = {
                "effect": effect,
                "self_error": self_error,
            }

            print(
                f"{label:15s} effect={effect:+.4f}"
            )

        rows.append(row)

        output.write_text(
            json.dumps({
                "experiment": "ML-017",
                "full": args.full,
                "records": rows,
            }, indent=2),
            encoding="utf-8",
        )

    print("\n=== SUMMARY ===")

    for condition in CONDITIONS:
        subset = [
            r for r in rows
            if r["condition"] == condition
        ]

        print(f"\n{condition}")

        for field in ("full", "attention_L13", "block_L16"):
            if field == "full":
                effects = [
                    r["full_effect"] for r in subset
                ]
            else:
                effects = [
                    r["patches"][field]["effect"]
                    for r in subset
                ]

            print(
                f"{field:15s} "
                f"mean={statistics.mean(effects):+.4f} "
                f"toward_carrier="
                f"{sum(x < 0 for x in effects)}/{len(effects)}"
            )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
