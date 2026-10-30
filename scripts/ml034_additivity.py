
"""ML-034: Representational vs downstream interaction in Qwen L20 readout.

Reuse ML-033 recipient cases and measured donor effects. Only new model
intervention: synthetic additive state h10 + h01 - h00 at two strengths.
"""
import argparse
import hashlib
import json
from datetime import datetime
from itertools import product
from pathlib import Path
from statistics import mean

from ml028_event_interchange import (
    PAIRS, VOICES, QUESTIONS, gold, capture, patched_scores,
    answer_scores, gold_margin,
)
from ml033_causal_compatibility import (
    HASH as ML033_HASH, LAYER, recipient_cases, validate_audit,
)

RESULTS = Path("results")
ALPHAS = (0.5, 1.0)

SPEC = {
    "experiment": "ML-034",
    "status": "prospective decomposition motivated by ML-033 (exploratory)",
    "model": "qwen7b",
    "source_hash": ML033_HASH,
    "layer": LAYER,
    "site": "readout",
    "cases": "same frozen 16 ML-033 recipients; event reversal=0",
    "alpha": ALPHAS,
    "synthetic": "h_add(a)=h00+a*(h10+h01-2*h00)",
    "decomposition": (
        "I_total=(y11-y10-y01+y00)="
        "(y11-yadd)+(yadd-y10-y01+y00)"
    ),
    "primary_prediction": (
        "at alpha=.5, mean(|downstream|) <= .25*mean(|total|), "
        "and representational term has total's sign >=12/16, "
        "and mean total >=+5"
    ),
    "note": (
        "Not a location of computation; synthetic additive states "
        "may be off manifold."
    ),
}

HASH = hashlib.sha256(
    json.dumps(SPEC, sort_keys=True).encode()
).hexdigest()


def read_ml033():
    paths = sorted(
        RESULTS.glob("ml033_qwen7b_compatibility_*.json")
    )

    matches = []

    for path in paths:
        obj = json.loads(path.read_text())

        if (
            obj.get("hash") == ML033_HASH
            and obj.get("complete")
        ):
            matches.append((path, obj))

    if not matches:
        raise RuntimeError(
            "No complete, matching ML-033 Qwen result"
        )

    path, obj = matches[-1]

    if len(obj["records"]) != 640:
        raise RuntimeError(
            "Expected 640 ML-033 records"
        )

    if [
        tuple(x) for x in obj["recipient_cases"]
    ] != recipient_cases():
        raise RuntimeError(
            "ML-033 recipient-case mismatch"
        )

    return path, obj


def previous_cells(data):
    cells = {}
    baselines = {}

    for row in data["records"]:
        case = tuple(
            row[k]
            for k in (
                "pid", "verb", "direction", "voice", "role"
            )
        )

        if case not in recipient_cases():
            raise RuntimeError(
                "Unexpected recipient"
            )

        if (
            row["kind"] != "donor"
            or int(row["event_rev"]) != 0
        ):
            continue

        alpha = float(row["alpha"])

        if alpha not in ALPHAS:
            continue

        key = (
            case,
            alpha,
            int(row["voice_flip"]),
            int(row["question_rev"]),
        )

        if key in cells:
            raise RuntimeError(
                f"Duplicate donor effect {key}"
            )

        cells[key] = float(row["effect"])
        b = float(row["baseline_margin"])

        if (
            case in baselines
            and abs(b - baselines[case]) > 1e-4
        ):
            raise RuntimeError(
                f"Inconsistent ML-033 baseline {case}"
            )

        baselines[case] = b

    expected = len(recipient_cases()) * len(ALPHAS) * 4

    if len(cells) != expected:
        raise RuntimeError(
            f"Expected {expected} source cells, got {len(cells)}"
        )

    return cells, baselines


def analyze(rows, source_cells):
    lookup = {}

    for row in rows:
        case = tuple(row["case"])
        key = (case, float(row["alpha"]))

        if key in lookup:
            raise RuntimeError(
                f"Duplicate ML-034 cell {key}"
            )

        lookup[key] = row

    expected = len(recipient_cases()) * len(ALPHAS)

    if len(lookup) != expected:
        raise RuntimeError(
            f"Expected {expected} additive cells, got {len(lookup)}"
        )

    print(
        "\n=== ML-034 REPRESENTATION / "
        "DOWNSTREAM DECOMPOSITION ==="
    )

    geometry = [
        r for r in rows
        if float(r["alpha"]) == 1.0
    ]

    if len(geometry) != 16:
        raise RuntimeError(
            "Missing structural geometry measurements"
        )

    ratios = [
        r["interaction_norm"]
        / max(r["full_donor_delta_norm"], 1e-12)
        for r in geometry
    ]

    print(
        "State interaction norm / donor shift norm: "
        f"mean={mean(ratios):.4f}"
    )

    all_parts = {}

    for alpha in ALPHAS:
        parts = []

        for case in recipient_cases():
            effects = {
                (vf, qr): source_cells[
                    case, alpha, vf, qr
                ]
                for vf, qr in product((0, 1), repeat=2)
            }

            y00, y10, y01, y11 = (
                effects[0, 0],
                effects[1, 0],
                effects[0, 1],
                effects[1, 1],
            )

            yadd = float(
                lookup[case, alpha]["additive_effect"]
            )

            total = y11 - y10 - y01 + y00
            rep = y11 - yadd
            downstream = yadd - y10 - y01 + y00

            if abs(total - (rep + downstream)) > 1e-5:
                raise RuntimeError(
                    "Decomposition identity failed"
                )

            parts.append(
                (total, rep, downstream)
            )

        all_parts[alpha] = parts

        totals, reps, downs = zip(*parts)

        abs_total = mean(
            abs(x) for x in totals
        )

        abs_down = mean(
            abs(x) for x in downs
        )

        congruent = sum(
            t * r > 0
            for t, r in zip(totals, reps)
        )

        print(
            f"\nalpha={alpha:.2f} n={len(parts)}"
        )

        print(
            f"  total: mean={mean(totals):+.4f} "
            f"positive={sum(x > 0 for x in totals)}/16"
        )

        print(
            f"  representational: mean={mean(reps):+.4f} "
            f"sign agrees={congruent}/16"
        )

        print(
            f"  downstream: mean={mean(downs):+.4f} "
            f"mean_abs={abs_down:.4f}"
        )

        print(
            "  downstream/total mean_abs="
            f"{abs_down / max(abs_total, 1e-12):.4f}"
        )

    totals, reps, downs = zip(
        *all_parts[0.5]
    )

    passed = (
        mean(totals) >= 5
        and mean(
            abs(x) for x in downs
        ) <= 0.25 * mean(
            abs(x) for x in totals
        )
        and sum(
            t * r > 0
            for t, r in zip(totals, reps)
        ) >= 12
    )

    print(
        "\nFROZEN PRIMARY PREDICTION:",
        "PASS" if passed else "FAIL"
    )

    print(
        "Interpretation is conditional on "
        "synthetic-state compatibility."
    )

    return passed


def collect():
    import mlx.core as mx

    from mlx_lm import load

    from ml003_role_margins import (
        MODELS, log_likelihood
    )

    from ml012_cue_patch import encode_prompt

    source_path, previous = read_ml033()

    cells, baselines = previous_cells(
        previous
    )

    model, tokenizer = load(
        str(MODELS / "qwen7b")
    )

    prompts = validate_audit(
        tokenizer, encode_prompt
    )

    print(
        "Audited prompts:", len(prompts)
    )

    print("Source:", source_path)

    RESULTS.mkdir(exist_ok=True)

    output = RESULTS / (
        "ml034_qwen7b_additivity_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    data = {
        "hash": HASH,
        "source": str(source_path),
        "complete": False,
        "records": [],
    }

    output.write_text(
        json.dumps(data, indent=2)
    )

    cached_group = None
    states = {}

    for case in recipient_cases():
        pid, verb, direction, voice, role = case
        group = (pid, verb)

        if group != cached_group:
            states = {}

            for d, v, q in product(
                (0, 1), VOICES, QUESTIONS
            ):
                prompt, pos = prompts[
                    (pid, verb, d, v, q)
                ]

                states[d, v, q] = capture(
                    model,
                    tokenizer,
                    prompt,
                    LAYER,
                    pos,
                    encode_prompt,
                )

            cached_group = group

        other_voice = next(
            v for v in VOICES
            if v != voice
        )

        other_role = next(
            q for q in QUESTIONS
            if q != role
        )

        h00 = states[
            direction, voice, role
        ]

        h10 = states[
            direction, other_voice, role
        ]

        h01 = states[
            direction, voice, other_role
        ]

        h11 = states[
            direction, other_voice, other_role
        ]

        h_interaction = (
            h11 - h10 - h01 + h00
        )

        norm_interaction = float(
            mx.sqrt(
                mx.sum(
                    h_interaction * h_interaction
                )
            )
        )

        difference = h11 - h00

        norm_h11_delta = float(
            mx.sqrt(
                mx.sum(
                    difference * difference
                )
            )
        )

        prompt, pos = prompts[case]
        pair = PAIRS[pid]

        expected = gold(
            pair, direction, role
        )

        baseline = gold_margin(
            answer_scores(
                model,
                tokenizer,
                prompt,
                log_likelihood,
                pair,
            ),
            expected,
            pair,
        )

        if abs(
            baseline - baselines[case]
        ) > 0.10:
            raise RuntimeError(
                f"Baseline drift at {case}: "
                f"{baseline} vs {baselines[case]}"
            )

        for alpha in ALPHAS:
            replacement = (
                h00
                + alpha * (
                    h10 + h01 - 2 * h00
                )
            )

            mx.eval(replacement)

            score = patched_scores(
                model,
                tokenizer,
                prompt,
                pair,
                LAYER,
                pos,
                replacement,
                log_likelihood,
            )

            effect = gold_margin(
                score,
                expected,
                pair,
            ) - baseline

            data["records"].append({
                "case": list(case),
                "alpha": alpha,
                "additive_effect": effect,
                "interaction_norm": norm_interaction,
                "full_donor_delta_norm": norm_h11_delta,
            })

        output.write_text(
            json.dumps(data, indent=2)
        )

        print(
            f"DONE "
            f"{len(data['records']) // len(ALPHAS)}"
            f"/{len(recipient_cases())}: {case}"
        )

    data["complete"] = True

    output.write_text(
        json.dumps(data, indent=2)
    )

    print("Saved:", output)

    analyze(
        data["records"], cells
    )


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
        "ML-034",
        args.phase,
        "qwen7b",
        HASH,
    )

    if args.phase == "design":
        print(
            json.dumps(SPEC, indent=2)
        )
        return

    source_path, source_data = read_ml033()
    cells, _ = previous_cells(source_data)

    print(
        "Source:",
        source_path,
        "source cells:",
        len(cells),
    )

    if args.phase == "summarize":
        paths = sorted(
            RESULTS.glob(
                "ml034_qwen7b_additivity_*.json"
            )
        )

        for path in reversed(paths):
            data = json.loads(
                path.read_text()
            )

            if (
                data.get("hash") == HASH
                and data.get("complete")
                and data.get("source")
                == str(source_path)
            ):
                print("Reading:", path)

                analyze(
                    data["records"], cells
                )
                return

        raise RuntimeError(
            "No complete ML-034 result "
            "matching this ML-033 source"
        )

    if args.phase == "audit":
        from mlx_lm import load

        from ml003_role_margins import MODELS

        from ml012_cue_patch import encode_prompt

        _, tokenizer = load(
            str(MODELS / "qwen7b")
        )

        prepared = validate_audit(
            tokenizer, encode_prompt
        )

        print(
            "Audited prompts:",
            len(prepared)
        )
        return

    collect()


if __name__ == "__main__":
    main()
