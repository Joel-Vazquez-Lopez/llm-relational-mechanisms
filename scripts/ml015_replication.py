
import json
import statistics
from datetime import datetime
from itertools import product
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from ml003_role_margins import MODELS
from ml012_cue_patch import (
    capture,
    encode_prompt,
    find_single_change,
    margin,
    patched_margin,
)
from ml013_position_probe import locate_name


CHECKS = [
    (0, "cue"),
    (14, "cue"),
    (12, "readout"),
    (13, "readout"),
    (16, "readout"),
    (19, "readout"),
    (19, "writer_name"),
    (19, "carrier_name"),
]

ASSIGNMENTS = [
    ("Leo", "Felix"),
    ("Felix", "Leo"),
]

ORDERS = ["bridge_first", "facts_first"]
PROBES = ["event", "writer"]


def get_record(records, writer, carrier, order, link, probe):
    matches = [
        record for record in records
        if record["writer"] == writer
        and record["carrier"] == carrier
        and record["order"] == order
        and record["link"] == link
        and record["probe"] == probe
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one record: "
            f"{writer}, {carrier}, {order}, {link}, {probe}"
        )

    return matches[0]


def aligned_margin(model, tokenizer, prompt, writer):
    raw = float(margin(model, tokenizer, prompt))
    direction = 1 if writer == "Leo" else -1
    return direction * raw


def main():
    sources = sorted(
        Path("results").glob("ml011_llama3b_*.json")
    )

    if not sources:
        raise FileNotFoundError("ML-011 results missing")

    source = sources[-1]
    records = json.loads(source.read_text())["records"]

    print("Source:", source)
    print("Loading Llama 3B")

    model, tokenizer = load(str(MODELS / "llama3b"))
    results = []

    output = Path("results") / (
        "ml015_replication_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    for (writer, carrier), order, probe in product(
        ASSIGNMENTS, ORDERS, PROBES
    ):
        a = get_record(
            records, writer, carrier, order, "wrote", probe
        )
        b = get_record(
            records, writer, carrier, order, "carried", probe
        )

        prompt_a = a["prompt"]
        prompt_b = b["prompt"]

        ids_a = encode_prompt(tokenizer, prompt_a)
        ids_b = encode_prompt(tokenizer, prompt_b)

        cue = find_single_change(
            tokenizer, prompt_a, prompt_b
        )

        if len(ids_a) != len(ids_b):
            raise RuntimeError("Prompt lengths differ")

        writer_pos = locate_name(
            tokenizer, ids_a, writer
        )
        carrier_pos = locate_name(
            tokenizer, ids_a, carrier
        )

        if writer_pos is None or carrier_pos is None:
            raise RuntimeError(
                "A participant name is not one token"
            )

        positions = {
            "cue": cue,
            "readout": len(ids_a) - 1,
            "writer_name": writer_pos,
            "carrier_name": carrier_pos,
        }

        baseline = aligned_margin(
            model, tokenizer, prompt_a, writer
        )
        counterfactual = aligned_margin(
            model, tokenizer, prompt_b, writer
        )
        full_effect = counterfactual - baseline

        print(
            f"\n=== {writer} writes, {carrier} carries | "
            f"{order} | {probe} ==="
        )
        print(
            "Full change (writer minus carrier):",
            f"{full_effect:+.4f}"
        )

        for layer, site in CHECKS:
            pos = positions[site]

            donor = capture(
                model, tokenizer, prompt_b, layer, pos
            )

            patched_raw = float(
                patched_margin(
                    model, tokenizer, prompt_a,
                    layer, pos, donor
                )
            )

            direction = 1 if writer == "Leo" else -1
            patched = direction * patched_raw
            effect = patched - baseline

            prefix_delta = None

            # States preceding the edited word must be
            # unaffected by its later occurrence.
            if pos < cue:
                own = capture(
                    model, tokenizer, prompt_a, layer, pos
                )
                prefix_delta = float(
                    mx.max(mx.abs(donor - own))
                )

                if prefix_delta > 1e-4:
                    raise RuntimeError(
                        f"Causal-prefix parity failed: "
                        f"layer={layer}, site={site}, "
                        f"delta={prefix_delta}"
                    )

            self_error = None

            if (layer, site) in [
                (13, "readout"),
                (16, "readout"),
                (19, "carrier_name"),
            ]:
                own = capture(
                    model, tokenizer, prompt_a, layer, pos
                )

                self_raw = float(
                    patched_margin(
                        model, tokenizer, prompt_a,
                        layer, pos, own
                    )
                )
                self_error = direction * self_raw - baseline

                if abs(self_error) > 0.05:
                    raise RuntimeError(
                        f"Self-patch failed: "
                        f"layer={layer}, site={site}"
                    )

            fraction = (
                effect / full_effect
                if abs(full_effect) >= 0.25
                else None
            )

            result = {
                "writer": writer,
                "carrier": carrier,
                "order": order,
                "probe": probe,
                "layer": layer,
                "site": site,
                "position": pos,
                "baseline": baseline,
                "counterfactual": counterfactual,
                "full_effect": full_effect,
                "effect": effect,
                "fraction": fraction,
                "prefix_delta": prefix_delta,
                "self_error": self_error,
            }
            results.append(result)

            print(
                f"L{layer:02d} {site:13s} "
                f"effect={effect:+.4f} "
                f"fraction={fraction if fraction is not None else 'NA'}"
            )

        # Preserve results after each condition.
        output.write_text(
            json.dumps({
                "experiment": "ML-015",
                "source": str(source),
                "records": results,
            }, indent=2),
            encoding="utf-8",
        )

    print("\n=== AGGREGATED EFFECTS ===")

    for probe in PROBES:
        print(f"\n{probe.upper()}")

        for layer, site in CHECKS:
            group = [
                r for r in results
                if r["probe"] == probe
                and r["layer"] == layer
                and r["site"] == site
            ]

            effects = [r["effect"] for r in group]
            expected = sum(x < 0 for x in effects)

            print(
                f"L{layer:02d} {site:13s} "
                f"mean={statistics.mean(effects):+.4f} "
                f"toward_carrier={expected}/{len(group)}"
            )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
