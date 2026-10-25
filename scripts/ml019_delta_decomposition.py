
import json
from datetime import datetime
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from ml003_role_margins import MODELS
from ml012_cue_patch import margin
from ml018_cross_context import (
    prompts,
    state_at,
    apply_state,
)


LAYER = 16
KIND = "block"

DONORS = {
    "felix_carrier": (
        "Leo", "Felix", "bridge_first", "sulmed"
    ),
    "leo_carrier": (
        "Felix", "Leo", "bridge_first", "sulmed"
    ),
}

# Fresh predicate, not used to construct either donor.
RECIPIENTS = [
    ("Leo", "Felix", "bridge_first", "vemped"),
    ("Felix", "Leo", "bridge_first", "vemped"),
    ("Leo", "Felix", "facts_first", "vemped"),
    ("Felix", "Leo", "facts_first", "vemped"),
]


def norm(x):
    x = x.astype(mx.float32)
    return float(mx.sqrt(mx.sum(x * x)))


def cosine(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    return float(
        mx.sum(a * b) / (norm(a) * norm(b))
    )


def donor_delta(model, tokenizer, config):
    a, b = prompts(
        tokenizer, config, "event_relevant"
    )

    s_a = state_at(
        model, tokenizer, a, LAYER, KIND
    )
    s_b = state_at(
        model, tokenizer, b, LAYER, KIND
    )

    delta = (
        s_b.astype(mx.float32)
        - s_a.astype(mx.float32)
    )
    mx.eval(delta)
    return delta


def main():
    model, tokenizer = load(
        str(MODELS / "llama3b")
    )

    delta_f = donor_delta(
        model, tokenizer, DONORS["felix_carrier"]
    )
    delta_l = donor_delta(
        model, tokenizer, DONORS["leo_carrier"]
    )

    shared = (delta_f + delta_l) / 2
    reversing = (delta_f - delta_l) / 2

    reference_norm = (
        norm(delta_f) + norm(delta_l)
    ) / 2

    def matched(x):
        if norm(x) < 1e-8:
            return x
        return x * (reference_norm / norm(x))

    vectors = {
        "donor_felix": delta_f,
        "donor_leo": delta_l,
        "shared": shared,
        "reversing": reversing,
        "negative_reversing": -reversing,
        "shared_norm_matched": matched(shared),
        "reversing_norm_matched": matched(reversing),
    }

    print("\n=== DONOR GEOMETRY ===")
    print("Felix donor norm:", norm(delta_f))
    print("Leo donor norm:", norm(delta_l))
    print("Donor cosine:", cosine(delta_f, delta_l))
    print("Shared norm:", norm(shared))
    print("Reversing norm:", norm(reversing))

    rows = []

    output = Path("results") / (
        "ml019_delta_decomposition_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    for writer, carrier, order, verb in RECIPIENTS:
        config = (writer, carrier, order, verb)
        a, b = prompts(
            tokenizer, config, "event_relevant"
        )

        sign = 1 if writer == "Leo" else -1

        base_raw = float(margin(
            model, tokenizer, a
        ))
        full_raw = float(margin(
            model, tokenizer, b
        ))

        base_state = state_at(
            model, tokenizer, a, LAYER, KIND
        ).astype(mx.float32)

        counter_state = state_at(
            model, tokenizer, b, LAYER, KIND
        ).astype(mx.float32)

        self_raw = apply_state(
            model, tokenizer, a,
            LAYER, KIND, base_state
        )

        if abs(self_raw - base_raw) > 0.05:
            raise RuntimeError("Self-patch parity failed")

        local_raw = apply_state(
            model, tokenizer, a,
            LAYER, KIND, counter_state
        )

        row = {
            "writer": writer,
            "carrier": carrier,
            "order": order,
            "verb": verb,
            "full_effect": sign * (full_raw - base_raw),
            "local_effect": sign * (local_raw - base_raw),
            "effects": {},
        }

        print(
            f"\n=== writer={writer}, carrier={carrier}, "
            f"order={order} ==="
        )
        print("Full effect:", f"{row['full_effect']:+.4f}")
        print("Local patch:", f"{row['local_effect']:+.4f}")

        for name, delta in vectors.items():
            modified = base_state + delta
            mx.eval(modified)

            patched_raw = apply_state(
                model, tokenizer, a,
                LAYER, KIND, modified
            )

            effect = sign * (patched_raw - base_raw)
            row["effects"][name] = effect

            print(
                f"{name:23s} effect={effect:+.4f}"
            )

        rows.append(row)

        output.write_text(
            json.dumps({
                "experiment": "ML-019",
                "layer": LAYER,
                "site": KIND,
                "donor_cosine": cosine(delta_f, delta_l),
                "donor_norms": {
                    "felix": norm(delta_f),
                    "leo": norm(delta_l),
                    "shared": norm(shared),
                    "reversing": norm(reversing),
                },
                "records": rows,
            }, indent=2),
            encoding="utf-8",
        )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
