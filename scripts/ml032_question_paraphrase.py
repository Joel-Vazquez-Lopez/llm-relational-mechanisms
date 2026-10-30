
"""ML-032: Prospective lexical transfer of Qwen's voice x question effect.

Same-event donor-state readout interchange at layer 20, factors:
  donor voice mismatch x donor question-role reversal x question wording.

New verbs and first tokenization-valid pairs from a frozen candidate list.
No event reversal; answer mismatch consequently equals question-role reversal.

Scientific status: exploratory lexical transfer, not evidence of abstract grammar.
"""
import argparse
import hashlib
import json
from datetime import datetime
from itertools import product
from pathlib import Path
from statistics import mean

from ml028_event_interchange import (
    VOICES, QUESTIONS, event, gold, positions, answer_scores, gold_margin,
    capture, patched_scores,
)

LAYER = 20
VERBS = ("thanked", "followed")

CANDIDATE_PAIRS = (
    ("Maya", "Lena"),
    ("Omar", "Rosa"),
    ("Julia", "Sofia"),
    ("Luca", "Theo"),
    ("Iris", "Mila"),
    ("Evan", "Noah"),
    ("Emma", "Alice"),
    ("Ava", "Zoe"),
)

PHRASES = {
    "A": {
        "agent": "Who performed the action?",
        "patient": "Who received the action?",
    },
    "B": {
        "agent": "Who carried out the action?",
        "patient": "Who was acted upon?",
    },
}

SPEC = {
    "id": "ML-032",
    "model_primary": "qwen7b",
    "layer": LAYER,
    "position": "readout",
    "verbs": VERBS,
    "candidate_pairs": CANDIDATE_PAIRS,
    "selection": (
        "first two pairs satisfying token-only audits; "
        "no likelihood filtering"
    ),
    "questions": PHRASES,
    "donor_event": "same actor/patient and verb",
    "recipient_question_style": "A",
    "donor_factors": [
        "voice_mismatch",
        "role_reversed",
        "wording_B",
    ],
    "competence": (
        "all 64 prepared prompts: >=56/64 and "
        ">=6/8 per voice x role x style"
    ),
    "primary": (
        "On new pairs and verbs, voice-by-role contrast D_A and D_B "
        "each mean >+5 and >=24/32 positive. D_B must be positive "
        "in >=12/16 for each name pair."
    ),
    "lexical_specificity": (
        "abs(mean voice-by-wording W) <= "
        "0.5*abs(mean D_B)"
    ),
    "note": (
        "Exploratory transfer after prior ML-031; "
        "no abstract-mechanism claim."
    ),
}

HASH = hashlib.sha256(
    json.dumps(SPEC, sort_keys=True).encode()
).hexdigest()

ROOT = Path("results")


def flip(value, values):
    assert value in values and len(values) == 2
    return values[1] if value == values[0] else values[0]


def make_prompt(
    tokenizer,
    pair,
    verb,
    direction,
    voice,
    role,
    style,
):
    content = (
        "Answer with exactly one person's name.\n"
        f"Event: {event(pair, verb, direction, voice)}\n"
        f"Question: {PHRASES[style][role]}\n"
        "Answer:"
    )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def audit_pair(tokenizer, pair, encode_prompt):
    lengths = [
        len(
            tokenizer.encode(
                n,
                add_special_tokens=False,
            )
        )
        for n in pair
    ]

    if lengths[0] != lengths[1] or pair[0] == pair[1]:
        return None

    result = {}

    for verb, direction, voice, role, style in product(
        VERBS,
        (0, 1),
        VOICES,
        QUESTIONS,
        ("A", "B"),
    ):
        prompt = make_prompt(
            tokenizer,
            pair,
            verb,
            direction,
            voice,
            role,
            style,
        )

        try:
            position = positions(
                tokenizer,
                prompt,
                encode_prompt,
            )["readout"]

            ids = encode_prompt(tokenizer, prompt)

            contextual = []

            for name in pair:
                extended = encode_prompt(
                    tokenizer,
                    prompt + name,
                )

                if extended[:len(ids)] != ids:
                    return None

                contextual.append(
                    len(extended) - len(ids)
                )

            if (
                contextual[0] != contextual[1]
                or contextual[0] <= 0
            ):
                return None

        except (ValueError, IndexError):
            return None

        result[
            verb, direction, voice, role, style
        ] = (prompt, position)

    return result


def prepare(tokenizer, encode_prompt):
    selected = []
    prepared = {}

    for pair in CANDIDATE_PAIRS:
        audited = audit_pair(
            tokenizer,
            pair,
            encode_prompt,
        )

        if audited is not None:
            selected.append(pair)
            prepared[len(selected) - 1] = audited

        if len(selected) == 2:
            break

    if len(selected) != 2:
        raise RuntimeError(
            "Fewer than two tokenization-valid name pairs. "
            "STOP. Do not tune on model behavior."
        )

    assert sum(
        len(x) for x in prepared.values()
    ) == 64

    return selected, prepared


def cases():
    return list(
        product(
            range(2),
            VERBS,
            (0, 1),
            VOICES,
            QUESTIONS,
        )
    )


def get_complete(model_name, phase, pairs):
    paths = sorted(
        ROOT.glob(
            f"ml032_{model_name}_{phase}_*.json"
        )
    )

    for path in reversed(paths):
        data = json.loads(path.read_text())

        if (
            data.get("complete")
            and data.get("hash") == HASH
            and data.get("pairs")
            == [list(p) for p in pairs]
        ):
            return data, path

    raise FileNotFoundError(
        f"No complete matching {phase} result. "
        "Do not skip earlier gates."
    )


def save(data, path):
    path.write_text(
        json.dumps(data, indent=2)
    )


def summarize(rows):
    by_case = {}

    for r in rows:
        key = (
            r["pair_id"],
            r["verb"],
            r["direction"],
            r["voice"],
            r["role"],
        )

        donor = (
            int(r["voice_mismatch"]),
            int(r["role_reversed"]),
            r["style"],
        )

        cell = by_case.setdefault(key, {})

        if donor in cell:
            raise RuntimeError(
                f"Repeated donor condition: {key} {donor}"
            )

        cell[donor] = r["effect"]

    if (
        len(by_case) != 32
        or any(
            len(e) != 8
            for e in by_case.values()
        )
    ):
        raise RuntimeError(
            "Missing or extra factorial cells"
        )

    records = []

    for key, e in sorted(by_case.items()):

        def question_effect(
            voice_mismatch,
            style,
        ):
            return (
                e[voice_mismatch, 1, style]
                - e[voice_mismatch, 0, style]
            )

        da = (
            question_effect(1, "A")
            - question_effect(0, "A")
        )

        db = (
            question_effect(1, "B")
            - question_effect(0, "B")
        )

        wording = (
            (
                e[1, 0, "B"]
                - e[1, 0, "A"]
            )
            - (
                e[0, 0, "B"]
                - e[0, 0, "A"]
            )
        )

        records.append(
            (key, da, db, wording)
        )

    print(
        "\n=== ML-032 FROZEN CONTRASTS ==="
    )

    print(
        "D_style = "
        "(opposite voice: reversed role - same role) "
        "- (same voice: reversed role - same role)"
    )

    labels = (
        (1, "D_A original-question wording"),
        (2, "D_B paraphrased-question wording"),
        (3, "W voice-by-wording control, role fixed"),
    )

    for i, label in labels:
        values = [r[i] for r in records]

        print(
            f"{label}: "
            f"mean={mean(values):+.4f} "
            f"positive="
            f"{sum(x > 0 for x in values)}/32"
        )

        for pid in (0, 1):
            subset = [
                r[i]
                for r in records
                if r[0][0] == pid
            ]

            print(
                f"  pair{pid}: "
                f"mean={mean(subset):+.4f} "
                f"positive="
                f"{sum(x > 0 for x in subset)}/16"
            )

    ma = mean(r[1] for r in records)
    mb = mean(r[2] for r in records)
    mw = mean(r[3] for r in records)

    sign_a = sum(
        r[1] > 0 for r in records
    )

    sign_b = sum(
        r[2] > 0 for r in records
    )

    bypair_ok = all(
        sum(
            r[2] > 0
            for r in records
            if r[0][0] == pid
        ) >= 12
        for pid in (0, 1)
    )

    transfer = (
        ma > 5
        and mb > 5
        and sign_a >= 24
        and sign_b >= 24
        and bypair_ok
    )

    lexical = (
        abs(mw) <= 0.5 * abs(mb)
    )

    print(
        "\nPredeclared lexical-transfer threshold:",
        "PASS" if transfer else "FAIL",
    )

    print(
        "Predeclared wording-specificity threshold:",
        "PASS" if lexical else "FAIL",
    )

    print(
        "NOTE: Patches may be off-manifold, "
        "and voice remains confounded "
        "with surface word order."
    )

    return transfer, lexical


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=(
            "design",
            "audit",
            "behavior",
            "patch",
            "summarize",
        ),
        default="design",
    )

    parser.add_argument(
        "--model",
        choices=("qwen7b", "llama3b"),
        default="qwen7b",
    )

    args = parser.parse_args()

    print(
        f"ML-032 | {args.phase} | {args.model} "
        f"| hash={HASH}"
    )

    if args.phase == "design":
        print(
            json.dumps(SPEC, indent=2)
        )
        return

    from mlx_lm import load

    from ml003_role_margins import (
        MODELS,
        log_likelihood,
    )

    from ml012_cue_patch import (
        encode_prompt,
    )

    model, tokenizer = load(
        str(MODELS / args.model)
    )

    pairs, prepared = prepare(
        tokenizer,
        encode_prompt,
    )

    print(
        "Selected pairs by tokenization ONLY:",
        pairs,
    )
    print("Audited prompts: 64/64")

    if args.phase == "audit":
        return

    ROOT.mkdir(exist_ok=True)

    if args.phase == "summarize":
        patch, path = get_complete(
            args.model,
            "patch",
            pairs,
        )

        print("Reading:", path)
        summarize(patch["records"])
        return

    if args.phase == "behavior":
        out = ROOT / (
            f"ml032_{args.model}_behavior_"
            f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
        )

        data = {
            "hash": HASH,
            "complete": False,
            "pairs": pairs,
            "records": [],
        }

        for pid, (
            verb,
            direction,
            voice,
            role,
            style,
        ) in product(
            range(2),
            product(
                VERBS,
                (0, 1),
                VOICES,
                QUESTIONS,
                ("A", "B"),
            ),
        ):
            prompt, _ = prepared[pid][
                verb,
                direction,
                voice,
                role,
                style,
            ]

            margin = gold_margin(
                answer_scores(
                    model,
                    tokenizer,
                    prompt,
                    log_likelihood,
                    pairs[pid],
                ),
                gold(
                    pairs[pid],
                    direction,
                    role,
                ),
                pairs[pid],
            )

            data["records"].append({
                "pid": pid,
                "verb": verb,
                "direction": direction,
                "voice": voice,
                "role": role,
                "style": style,
                "margin": margin,
                "correct": margin > 0,
            })

            save(data, out)

        data["complete"] = True
        save(data, out)

        n = sum(
            x["correct"]
            for x in data["records"]
        )

        print(
            f"Behavior: {n}/64; saved: {out}"
        )

        for voice, role, style in product(
            VOICES,
            QUESTIONS,
            ("A", "B"),
        ):
            subset = [
                x
                for x in data["records"]
                if (
                    x["voice"],
                    x["role"],
                    x["style"],
                ) == (voice, role, style)
            ]

            print(
                f"{voice} {role} {style}: "
                f"{sum(x['correct'] for x in subset)}/8"
            )

        return

    behavior, _ = get_complete(
        args.model,
        "behavior",
        pairs,
    )

    rows = behavior["records"]

    stratified = [
        sum(
            x["correct"]
            for x in rows
            if (
                x["voice"],
                x["role"],
                x["style"],
            ) == (v, q, s)
        )
        for v, q, s in product(
            VOICES,
            QUESTIONS,
            ("A", "B"),
        )
    ]

    if (
        len(rows) != 64
        or sum(
            x["correct"]
            for x in rows
        ) < 56
        or min(stratified) < 6
    ):
        raise RuntimeError(
            "Behavior gate failed. "
            "No patches. "
            "Do not alter thresholds after looking."
        )

    out = ROOT / (
        f"ml032_{args.model}_patch_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    data = {
        "hash": HASH,
        "complete": False,
        "pairs": pairs,
        "model": args.model,
        "layer": LAYER,
        "site": "readout",
        "records": [],
    }

    for pid, verb in product(
        range(2),
        VERBS,
    ):
        states = {}

        for direction, voice, role, style in product(
            (0, 1),
            VOICES,
            QUESTIONS,
            ("A", "B"),
        ):
            prompt, position = prepared[pid][
                verb,
                direction,
                voice,
                role,
                style,
            ]

            states[
                direction,
                voice,
                role,
                style,
            ] = capture(
                model,
                tokenizer,
                prompt,
                LAYER,
                position,
                encode_prompt,
            )

        for direction, voice, role in product(
            (0, 1),
            VOICES,
            QUESTIONS,
        ):
            prompt, position = prepared[pid][
                verb,
                direction,
                voice,
                role,
                "A",
            ]

            pair = pairs[pid]

            baseline = gold_margin(
                answer_scores(
                    model,
                    tokenizer,
                    prompt,
                    log_likelihood,
                    pair,
                ),
                gold(
                    pair,
                    direction,
                    role,
                ),
                pair,
            )

            for voice_flip, role_flip, style in product(
                (0, 1),
                (0, 1),
                ("A", "B"),
            ):
                donor_voice = (
                    flip(voice, VOICES)
                    if voice_flip
                    else voice
                )

                donor_role = (
                    flip(role, QUESTIONS)
                    if role_flip
                    else role
                )

                donor_state = states[
                    direction,
                    donor_voice,
                    donor_role,
                    style,
                ]

                patched = gold_margin(
                    patched_scores(
                        model,
                        tokenizer,
                        prompt,
                        pair,
                        LAYER,
                        position,
                        donor_state,
                        log_likelihood,
                    ),
                    gold(
                        pair,
                        direction,
                        role,
                    ),
                    pair,
                )

                effect = patched - baseline

                if (
                    not voice_flip
                    and not role_flip
                    and style == "A"
                    and abs(effect) > 0.05
                ):
                    raise RuntimeError(
                        "Self-control failed. "
                        "No complete result will be written."
                    )

                data["records"].append({
                    "pair_id": pid,
                    "verb": verb,
                    "direction": direction,
                    "voice": voice,
                    "role": role,
                    "voice_mismatch": bool(voice_flip),
                    "role_reversed": bool(role_flip),
                    "style": style,
                    "baseline": baseline,
                    "patched": patched,
                    "effect": effect,
                })

            save(data, out)

            print(
                f"DONE pair={pid} verb={verb} "
                f"direction={direction} "
                f"voice={voice} role={role} "
                f"records={len(data['records'])}"
            )

    data["complete"] = True
    save(data, out)

    print("Saved:", out)

    summarize(data["records"])


if __name__ == "__main__":
    main()
