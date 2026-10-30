
"""ML-033: Causal state compatibility at Qwen L20 readout.

Frozen post-ML-031 follow-up: interpolate donor-recipient residual
differences.

Control: independently sign-flip coordinates of each donor difference,
preserving coordinate magnitudes and Euclidean norm while disrupting
the donor direction.

Exploratory mechanistic test, not identification of a grammar circuit.
"""

import argparse
import hashlib
import json

from datetime import datetime
from itertools import product
from pathlib import Path
from statistics import mean

import numpy as np

from ml028_event_interchange import (
    PAIRS, VERBS, VOICES, QUESTIONS,
    gold, make_prompt, positions,
    answer_scores, gold_margin,
    capture, patched_scores,
)

from ml029_answer_event_disentangle import (
    DESIGN_HASH as ML029_HASH,
)

from ml031_voice_factorial import (
    HASH as ML031_HASH,
    get_file,
    load_records,
)


LAYER = 20

DOSES = (0.25, 0.5, 1.0)
CONTROL_DOSES = (0.25, 0.5)

RESULTS = Path("results")


DESIGN = {
    "experiment": "ML-033",
    "status": "exploratory after ML-031; fixed before new inference",
    "model": "qwen7b",
    "layer": LAYER,
    "site": "readout (last prompt token)",
    "stimuli": "original ML-029/031 prompts",
    "recipient_subset": (
        "16 of 32; "
        "parity(pair_id,verb_idx,direction,voice_idx,role_idx) = 0"
    ),
    "donors": (
        "voice mismatch x event reversal x question reversal"
    ),
    "dose": DOSES,
    "control_dose": CONTROL_DOSES,
    "control": (
        "Deterministic per-donor coordinate sign flips "
        "of donor-minus-recipient residual difference"
    ),
    "primary": (
        "At alpha=0.5, true donor voice-by-question "
        "interaction >+5 and positive >=12/16; "
        "paired excess over signflip >+5 and "
        "positive >=12/16"
    ),
    "replication": (
        "At alpha=1.0, all effects must match "
        "ML-029/031 within 0.10 margin units"
    ),
    "limitations": (
        "Not an on-manifold control, not independent "
        "heldout data, not a circuit-specific intervention"
    ),
    "ml029_hash": ML029_HASH,
    "ml031_hash": ML031_HASH,
}

HASH = hashlib.sha256(
    json.dumps(DESIGN, sort_keys=True).encode()
).hexdigest()


def recipient_cases():
    """Deterministic balanced half-fraction of original cases."""

    cases = []

    for pid, vi, direction, voice_idx, role_idx in product(
        range(len(PAIRS)),
        range(len(VERBS)),
        (0, 1),
        (0, 1),
        (0, 1),
    ):
        parity = (
            pid ^ vi ^ direction ^ voice_idx ^ role_idx
        )

        if parity == 0:
            cases.append((
                pid,
                VERBS[vi],
                direction,
                VOICES[voice_idx],
                QUESTIONS[role_idx],
            ))

    assert len(cases) == 16

    return cases


def old_signature(case, event_rev, question_rev):
    return (
        *case,
        int(event_rev),
        int(question_rev),
    )


def sources():
    """Validate existing ML-029 and ML-031 measurements."""

    opposite_path = get_file(
        "ml029_qwen7b_*.json"
    )

    same_path = get_file(
        "ml031_qwen7b_samevoice_*.json"
    )

    opposite = load_records(
        opposite_path, ML029_HASH, 128
    )

    same = load_records(
        same_path, ML031_HASH, 128
    )

    if set(opposite) != set(same):
        raise RuntimeError(
            "ML-029 and ML-031 condition keys disagree"
        )

    for case in recipient_cases():
        for er, qr in product((0, 1), repeat=2):
            key = old_signature(case, er, qr)

            if key not in same:
                raise RuntimeError(
                    f"Missing historical case: {key}"
                )

    return (
        (same, opposite),
        (str(same_path), str(opposite_path)),
    )


def seed_for(case, voice_flip, event_rev, question_rev):
    """Reproducible independent control for each donor."""

    payload = json.dumps(
        [
            HASH,
            list(case),
            voice_flip,
            event_rev,
            question_rev,
        ],
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        payload.encode()
    ).digest()

    return int.from_bytes(
        digest[:8], "big"
    )


def validate_audit(tokenizer, encode_prompt):
    """Check prompt alignment and candidate tokenization."""

    prepared = {}

    for pid, verb, direction, voice, role in product(
        range(len(PAIRS)),
        VERBS,
        (0, 1),
        VOICES,
        QUESTIONS,
    ):
        key = (
            pid, verb, direction, voice, role
        )

        pair = PAIRS[pid]

        prompt = make_prompt(
            tokenizer,
            pair,
            verb,
            direction,
            voice,
            role,
        )

        position = positions(
            tokenizer,
            prompt,
            encode_prompt,
        )["readout"]

        ids = encode_prompt(
            tokenizer, prompt
        )

        lengths = []

        for name in pair:
            extended = encode_prompt(
                tokenizer,
                prompt + name,
            )

            if extended[:len(ids)] != ids:
                raise RuntimeError(
                    f"Candidate prefix changed: {key} {name}"
                )

            lengths.append(
                len(extended) - len(ids)
            )

        if (
            lengths[0] != lengths[1]
            or lengths[0] <= 0
        ):
            raise RuntimeError(
                f"Unequal candidate lengths: {key} {lengths}"
            )

        prepared[key] = (
            prompt, position
        )

    assert len(prepared) == 32

    return prepared


def intervention_key(row):
    return (
        row["pid"],
        row["verb"],
        row["direction"],
        row["voice"],
        row["role"],
        int(row["voice_flip"]),
        int(row["event_rev"]),
        int(row["question_rev"]),
        row["kind"],
        float(row["alpha"]),
    )


def contrast(cells):
    """Voice x question interaction, averaged over event reversal."""

    def question_effect(voice_flip):
        return mean(
            cells[voice_flip, er, 1]
            - cells[voice_flip, er, 0]
            for er in (0, 1)
        )

    return (
        question_effect(1)
        - question_effect(0)
    )


def summarize(data):
    """Evaluate frozen contrasts, preserving individual cases."""

    if (
        data.get("hash") != HASH
        or not data.get("complete")
        or len(data.get("records", [])) != 640
    ):
        raise RuntimeError(
            "Incomplete or wrong-design ML-033 result"
        )

    by_case = {}
    seen = set()

    for row in data["records"]:
        unique = intervention_key(row)

        if unique in seen:
            raise RuntimeError(
                f"Duplicate measurement: {unique}"
            )

        seen.add(unique)

        case = unique[:5]

        cell = (
            row["kind"],
            float(row["alpha"]),
            int(row["voice_flip"]),
            int(row["event_rev"]),
            int(row["question_rev"]),
        )

        by_case.setdefault(case, {})[cell] = float(
            row["effect"]
        )

    if set(by_case) != set(recipient_cases()):
        raise RuntimeError(
            "Recipient subset mismatch"
        )

    output = {}

    print(
        "\n=== ML-033 CAUSAL COMPATIBILITY ==="
    )

    for kind, alphas in (
        ("donor", DOSES),
        ("signflip", CONTROL_DOSES),
    ):
        for alpha in alphas:
            values = []

            for case in recipient_cases():
                cells = {}

                for vf, er, qr in product(
                    (0, 1), repeat=3
                ):
                    key = (
                        kind, alpha,
                        vf, er, qr,
                    )

                    if key not in by_case[case]:
                        raise RuntimeError(
                            f"Missing cell: {case} {key}"
                        )

                    cells[vf, er, qr] = (
                        by_case[case][key]
                    )

                values.append(
                    contrast(cells)
                )

            output[kind, alpha] = values

            print(
                f"{kind:8s} "
                f"alpha={alpha:.2f} "
                f"voice_x_question={mean(values):+.3f} "
                f"positive={sum(x > 0 for x in values)}/16"
            )

    print(
        "\n=== PAIRED DONOR MINUS CONTROL ==="
    )

    for alpha in CONTROL_DOSES:
        differences = [
            actual - control
            for actual, control in zip(
                output["donor", alpha],
                output["signflip", alpha],
            )
        ]

        print(
            f"alpha={alpha:.2f} "
            f"excess={mean(differences):+.3f} "
            f"positive={sum(x > 0 for x in differences)}/16"
        )

    actual = output["donor", 0.5]
    control = output["signflip", 0.5]

    excess = [
        d - c
        for d, c in zip(actual, control)
    ]

    passed = (
        mean(actual) > 5
        and sum(x > 0 for x in actual) >= 12
        and mean(excess) > 5
        and sum(x > 0 for x in excess) >= 12
    )

    print(
        "\nFROZEN PRIMARY CRITERION:",
        "PASS" if passed else "FAIL",
    )

    print(
        "Passing indicates directional causal sensitivity, "
        "not an identified grammar circuit."
    )

    return passed


def load_existing():
    matching = []

    for path in RESULTS.glob(
        "ml033_qwen7b_compatibility_*.json"
    ):
        data = json.loads(
            path.read_text()
        )

        if data.get("hash") == HASH:
            matching.append(
                (path, data)
            )

    return sorted(
        matching,
        key=lambda x: x[0].name,
    )


def collect():
    import mlx.core as mx

    from mlx_lm import load

    from ml003_role_margins import (
        MODELS,
        log_likelihood,
    )

    from ml012_cue_patch import (
        encode_prompt,
    )

    old, paths = sources()

    print(
        "Validated historical sources:",
        *paths,
        sep="\n  ",
    )

    model, tokenizer = load(
        str(MODELS / "qwen7b")
    )

    prompts = validate_audit(
        tokenizer,
        encode_prompt,
    )

    print(
        "Audited prompts:",
        len(prompts),
    )

    existing = load_existing()

    if any(
        data.get("complete")
        for _, data in existing
    ):
        raise RuntimeError(
            "Complete matching ML-033 result already exists"
        )

    if existing:
        path, data = existing[-1]

        if data.get("source_files") != list(paths):
            raise RuntimeError(
                "Historical source files changed"
            )

        done = len(
            data["records"]
        ) // 40

        if len(data["records"]) != done * 40:
            raise RuntimeError(
                "Checkpoint has incomplete recipient records"
            )

        for i in range(done):
            case = recipient_cases()[i]

            for row in data["records"][i * 40:(i + 1) * 40]:
                recorded = tuple(
                    row[k]
                    for k in (
                        "pid",
                        "verb",
                        "direction",
                        "voice",
                        "role",
                    )
                )

                if recorded != case:
                    raise RuntimeError(
                        "Checkpoint case ordering differs"
                    )

        print(
            f"Resuming after {done}/16 recipients"
        )

    else:
        done = 0

        RESULTS.mkdir(
            exist_ok=True
        )

        path = RESULTS / (
            "ml033_qwen7b_compatibility_"
            f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
        )

        data = {
            "experiment": "ML-033",
            "hash": HASH,
            "complete": False,
            "model": "qwen7b",
            "source_files": list(paths),
            "recipient_cases": recipient_cases(),
            "records": [],
        }

        path.write_text(
            json.dumps(data, indent=2)
        )

    cached_group = None
    states = {}

    for index, case in enumerate(
        recipient_cases()[done:],
        start=done,
    ):
        pid, verb, direction, voice, role = case

        group = (
            pid, verb
        )

        if group != cached_group:
            states = {}

            for d, v, q in product(
                (0, 1),
                VOICES,
                QUESTIONS,
            ):
                prompt, position = prompts[
                    pid, verb, d, v, q
                ]

                states[d, v, q] = capture(
                    model,
                    tokenizer,
                    prompt,
                    LAYER,
                    position,
                    encode_prompt,
                )

            cached_group = group

        prompt, position = prompts[case]

        pair = PAIRS[pid]

        expected = gold(
            pair,
            direction,
            role,
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

        recipient = states[
            direction, voice, role
        ]

        recip_norm = mx.sqrt(
            mx.sum(recipient * recipient)
        )

        mx.eval(recip_norm)

        recip_norm_float = float(
            recip_norm
        )

        recipient_rows = []

        for vf, er, qr in product(
            (0, 1),
            repeat=3,
        ):
            donor_voice = (
                VOICES[1 - VOICES.index(voice)]
                if vf
                else voice
            )

            donor_direction = (
                1 - direction
                if er
                else direction
            )

            donor_role = (
                QUESTIONS[1 - QUESTIONS.index(role)]
                if qr
                else role
            )

            donor = states[
                donor_direction,
                donor_voice,
                donor_role,
            ]

            difference = (
                donor - recipient
            )

            norm = mx.sqrt(
                mx.sum(difference * difference)
            )

            mx.eval(norm)

            donor_norm = float(
                norm
            )

            rng = np.random.default_rng(
                seed_for(case, vf, er, qr)
            )

            sign_values = rng.choice(
                (-1.0, 1.0),
                size=(difference.shape[-1],),
            )

            signs = mx.array(
                sign_values.astype(np.float32)
            ).astype(difference.dtype)

            signflipped = difference * signs

            mx.eval(signflipped)

            for kind, alphas in (
                ("donor", DOSES),
                ("signflip", CONTROL_DOSES),
            ):
                for alpha in alphas:
                    if (
                        kind == "donor"
                        and alpha == 1.0
                    ):
                        # Exact state replacement.
                        replacement = donor

                    else:
                        delta = (
                            difference
                            if kind == "donor"
                            else signflipped
                        )

                        replacement = (
                            recipient + alpha * delta
                        )

                    mx.eval(replacement)

                    scores = patched_scores(
                        model,
                        tokenizer,
                        prompt,
                        pair,
                        LAYER,
                        position,
                        replacement,
                        log_likelihood,
                    )

                    patched_margin = gold_margin(
                        scores,
                        expected,
                        pair,
                    )

                    effect = (
                        patched_margin - baseline
                    )

                    # Ensure exact-state patching is consistent.
                    if (
                        vf == er == qr == 0
                        and abs(effect) > 0.05
                    ):
                        raise RuntimeError(
                            "Self-patch failed: "
                            f"{case} {kind} {alpha}: "
                            f"{effect}"
                        )

                    # Replicate previously measured full patches.
                    if (
                        kind == "donor"
                        and alpha == 1.0
                    ):
                        reference = old[vf][
                            old_signature(case, er, qr)
                        ]

                        if abs(
                            effect - reference
                        ) > 0.10:
                            raise RuntimeError(
                                "Historical replication mismatch: "
                                f"case={case} "
                                f"donor={(vf, er, qr)} "
                                f"new={effect:+.4f} "
                                f"old={reference:+.4f}"
                            )

                    recipient_rows.append({
                        "pid": pid,
                        "verb": verb,
                        "direction": direction,
                        "voice": voice,
                        "role": role,
                        "voice_flip": vf,
                        "event_rev": er,
                        "question_rev": qr,
                        "kind": kind,
                        "alpha": alpha,
                        "baseline_margin": baseline,
                        "patched_margin": patched_margin,
                        "effect": effect,
                        "donor_distance": donor_norm,
                        "relative_distance": (
                            donor_norm
                            / max(recip_norm_float, 1e-12)
                        ),
                    })

        assert len(recipient_rows) == 40

        data["records"].extend(
            recipient_rows
        )

        # Checkpoint every complete recipient.
        path.write_text(
            json.dumps(data, indent=2)
        )

        print(
            f"DONE {index+1}/16 "
            f"{case} "
            f"records={len(data['records'])}"
        )

    data["complete"] = True

    path.write_text(
        json.dumps(data, indent=2)
    )

    print(
        "Saved:",
        path,
    )

    summarize(data)


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
        "ML-033 |",
        args.phase,
        "| qwen7b | hash:",
        HASH,
    )

    if args.phase == "design":
        print(
            json.dumps(DESIGN, indent=2)
        )

        print(
            "Selected recipient cases:",
            *recipient_cases(),
            sep="\n  ",
        )

        return

    if args.phase == "summarize":
        matching = load_existing()

        complete = [
            (path, data)
            for path, data in matching
            if data.get("complete")
        ]

        if not complete:
            raise RuntimeError(
                "No complete matching ML-033 result"
            )

        path, data = complete[-1]

        print(
            "Reading:",
            path,
        )

        summarize(data)
        return

    if args.phase == "collect":
        collect()
        return

    # Audit phase
    from mlx_lm import load

    from ml003_role_margins import MODELS

    from ml012_cue_patch import encode_prompt

    _, tokenizer = load(
        str(MODELS / "qwen7b")
    )

    prepared = validate_audit(
        tokenizer,
        encode_prompt,
    )

    old, paths = sources()

    print(
        "Audited prompts:",
        len(prepared),
    )

    print(
        "Matching historical conditions:",
        len(old[0]),
        len(old[1]),
    )

    print(
        "Data sources:",
        *paths,
        sep="\n  ",
    )


if __name__ == "__main__":
    main()
