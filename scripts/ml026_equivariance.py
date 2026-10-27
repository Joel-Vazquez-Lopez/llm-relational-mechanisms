
"""ML-026: Coordinated Relational Transformation.

Test functional equivariance under:
1. Semantic-equivalent active/passive paraphrases.
2. Semantic role reversal.

No neural mechanism is assumed.
"""

import argparse
import hashlib
import json

from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from itertools import product
from pathlib import Path


# Frozen before running ML-026.
PACKS = {
    "calibration": {
        "a_noun": "letter",
        "a_base": "wrote",
        "a_alt": "carried",
        "b_noun": "map",
        "b_base": "folded",
        "b_alt": "painted",
        "verbs": ("helped", "praised"),
    },
    "holdout": {
        "a_noun": "parcel",
        "a_base": "sent",
        "a_alt": "kept",
        "b_noun": "cup",
        "b_base": "washed",
        "b_alt": "dried",
        "verbs": ("thanked", "greeted"),
    },
}

ASSIGNMENTS = (
    {
        "a0": "Leo",
        "a1": "Felix",
        "b0": "Nora",
        "b1": "Clara",
    },
    {
        "a0": "Felix",
        "a1": "Leo",
        "b0": "Clara",
        "b1": "Nora",
    },
)

ORDERS = ("a_first", "b_first")
VOICES = ("active", "passive")
QUESTIONS = ("agent", "patient")
EDITS = ("base", "a", "b")

FROZEN_HYPOTHESES = {
    "semantic_equivalence": "G(target) = G(source)",
    "role_reversal": "G(target) = -G(source)",
    "surface_competitor": "Opposite transformation",
    "zero_competitor": "G(target) = 0",
    "training_mean_competitor": (
        "Calibration mean for target order and voice"
    ),
}


def family_a_is_agent(order, voice):
    return (
        (order == "a_first")
        == (voice == "active")
    )


def make_passage(
    pack, assignment, order, voice, edit, verb
):
    a_word = (
        pack["a_alt"]
        if edit == "a"
        else pack["a_base"]
    )

    b_word = (
        pack["b_alt"]
        if edit == "b"
        else pack["b_base"]
    )

    a_desc = (
        f"the person who {a_word} "
        f"the {pack['a_noun']}"
    )

    b_desc = (
        f"the person who {b_word} "
        f"the {pack['b_noun']}"
    )

    if order == "a_first":
        first, second = a_desc, b_desc
    else:
        first, second = b_desc, a_desc

    if voice == "active":
        event = (
            f"At dusk, {first} {verb} {second}."
        )
    else:
        event = (
            f"At dusk, {first} was {verb} "
            f"by {second}."
        )

    facts = " ".join((
        f"{assignment['a0']} "
        f"{pack['a_base']} the {pack['a_noun']}.",

        f"{assignment['a1']} "
        f"{pack['a_alt']} the {pack['a_noun']}.",

        f"{assignment['b0']} "
        f"{pack['b_base']} the {pack['b_noun']}.",

        f"{assignment['b1']} "
        f"{pack['b_alt']} the {pack['b_noun']}.",
    ))

    return f"{event} {facts}"


def prompt_for(
    tokenizer, passage, verb, question
):
    if question == "agent":
        q = f"Who {verb} the other person?"
    else:
        q = f"Who was {verb}?"

    content = (
        "Answer with exactly one person's name.\n\n"
        f"Passage: {passage}\n"
        f"Question: {q}\n"
        "Answer:"
    )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def gold(
    assignment, order, voice, edit, question
):
    a = assignment[
        "a1" if edit == "a" else "a0"
    ]

    b = assignment[
        "b1" if edit == "b" else "b0"
    ]

    choose_a = (
        family_a_is_agent(order, voice)
        == (question == "agent")
    )

    return a if choose_a else b


def design_spec(split):
    return {
        "experiment": "ML-026",
        "split": split,
        "pack": PACKS[split],
        "assignments": ASSIGNMENTS,
        "orders": ORDERS,
        "voices": VOICES,
        "questions": QUESTIONS,
        "edits": EDITS,
        "hypotheses": FROZEN_HYPOTHESES,
        "score": "Whole-answer name log likelihood",
        "profile": (
            "S[q,c] = margin_base[q,c] "
            "- margin_edited[q,c]"
        ),
        "gap": (
            "G[c] = S[agent,c] - S[patient,c]"
        ),
    }


def design_hash(split):
    raw = json.dumps(
        design_spec(split),
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def cases_for(split):
    pack = PACKS[split]

    for aid, verb, order, voice in product(
        range(2),
        pack["verbs"],
        ORDERS,
        VOICES,
    ):
        yield aid, verb, order, voice


def prompts_for_case(
    tokenizer, split, case
):
    aid, verb, order, voice = case

    pack = PACKS[split]
    assignment = ASSIGNMENTS[aid]

    return {
        (q, edit): prompt_for(
            tokenizer,
            make_passage(
                pack,
                assignment,
                order,
                voice,
                edit,
                verb,
            ),
            verb,
            q,
        )
        for q, edit in product(
            QUESTIONS, EDITS
        )
    }


def single_substitution(
    base, counterfactual, label
):
    edits = [
        e for e in SequenceMatcher(
            None,
            base,
            counterfactual,
            autojunk=False,
        ).get_opcodes()
        if e[0] != "equal"
    ]

    good = (
        len(base) == len(counterfactual)
        and len(edits) == 1
        and edits[0][0] == "replace"
        and edits[0][2] - edits[0][1] == 1
        and edits[0][4] - edits[0][3] == 1
        and edits[0][1] == edits[0][3]
    )

    if not good:
        raise ValueError(
            f"{label}: invalid cue alignment: "
            f"{edits}"
        )

    return edits[0][1]


def audit_case(
    tokenizer, prompts, encode_prompt
):
    locations = {}

    for q in QUESTIONS:
        ids = {
            edit: encode_prompt(
                tokenizer, prompts[(q, edit)]
            )
            for edit in EDITS
        }

        pa = single_substitution(
            ids["base"],
            ids["a"],
            f"{q}/a",
        )

        pb = single_substitution(
            ids["base"],
            ids["b"],
            f"{q}/b",
        )

        if pa == pb:
            raise ValueError(
                "Cue positions collide"
            )

        locations[q] = (pa, pb)

    return locations


def score_names(
    model, tokenizer, prompt, log_likelihood
):
    names = (
        "Leo", "Felix", "Nora", "Clara"
    )

    return {
        name: float(
            log_likelihood(
                model,
                tokenizer,
                prompt,
                name,
            )[0]
        )
        for name in names
    }


def margin(scores, assignment, cue):
    return (
        scores[assignment[f"{cue}0"]]
        - scores[assignment[f"{cue}1"]]
    )


def profile(records):
    groups = defaultdict(dict)

    for r in records:
        key = (
            r["assignment"],
            r["verb"],
            r["order"],
            r["voice"],
        )

        groups[key][
            (r["question"], r["edit"])
        ] = r

    output = {}

    for key, rows in groups.items():
        expected = set(
            product(QUESTIONS, EDITS)
        )

        if set(rows) != expected:
            raise ValueError(
                f"Incomplete group: {key}"
            )

        assignment = ASSIGNMENTS[key[0]]
        S = {}

        for q, cue in product(
            QUESTIONS, ("a", "b")
        ):
            base = margin(
                rows[(q, "base")]["scores"],
                assignment,
                cue,
            )

            edited = margin(
                rows[(q, cue)]["scores"],
                assignment,
                cue,
            )

            S[q, cue] = base - edited

        G = {
            cue: (
                S["agent", cue]
                - S["patient", cue]
            )
            for cue in ("a", "b")
        }

        output[key] = {
            "S": S,
            "G": G,
        }

    return output


def mse(pairs):
    return sum(
        (actual - prediction) ** 2
        for actual, prediction in pairs
    ) / len(pairs)


def mae(pairs):
    return sum(
        abs(actual - prediction)
        for actual, prediction in pairs
    ) / len(pairs)


def evaluate(calibration, holdout):
    train = profile(
        calibration["records"]
    )

    test = profile(
        holdout["records"]
    )

    calibration_means = {}

    for order, voice, cue in product(
        ORDERS, VOICES, ("a", "b")
    ):
        values = [
            result["G"][cue]
            for (aid, verb, o, v), result
            in train.items()
            if o == order and v == voice
        ]

        calibration_means[
            order, voice, cue
        ] = sum(values) / len(values)

    results = {
        kind: defaultdict(list)
        for kind in (
            "equivalent", "reversed"
        )
    }

    def add_pair(
        kind,
        source,
        target,
        target_order,
        target_voice,
    ):
        for cue in ("a", "b"):
            s = source["G"][cue]
            t = target["G"][cue]

            if kind == "equivalent":
                relational = s
                surface = -s
            else:
                relational = -s
                surface = s

            collection = results[kind]

            collection["relational"].append(
                (t, relational)
            )

            collection["surface"].append(
                (t, surface)
            )

            collection["zero"].append(
                (t, 0.0)
            )

            collection[
                "calibration_mean"
            ].append((
                t,
                calibration_means[
                    target_order,
                    target_voice,
                    cue,
                ],
            ))

    for aid, verb in product(
        range(2),
        PACKS["holdout"]["verbs"],
    ):
        def get(order, voice):
            return test[
                (aid, verb, order, voice)
            ]

        # Same event, different constructions.
        # Predict unchanged G.
        add_pair(
            "equivalent",
            get("a_first", "active"),
            get("b_first", "passive"),
            "b_first",
            "passive",
        )

        add_pair(
            "equivalent",
            get("b_first", "active"),
            get("a_first", "passive"),
            "a_first",
            "passive",
        )

        # Same noun-phrase order, reversed roles.
        # Predict sign-flipped G.
        for order in ORDERS:
            add_pair(
                "reversed",
                get(order, "active"),
                get(order, "passive"),
                order,
                "passive",
            )

    print(
        "\n=== FROZEN HELD-OUT "
        "TRANSFORMATION TEST ==="
    )

    print(
        "G[c] = S[agent,c] - S[patient,c]"
    )

    print(
        "Lower MAE and MSE are better."
    )

    for kind, models in results.items():
        print(
            f"\n{kind.upper()} "
            f"({len(models['relational'])} "
            "cue-level comparisons)"
        )

        for name in (
            "relational",
            "surface",
            "zero",
            "calibration_mean",
        ):
            values = models[name]

            print(
                f"{name:18s} "
                f"MAE={mae(values):.4f} "
                f"MSE={mse(values):.4f}"
            )

    aligned = []

    for (
        aid, verb, order, voice
    ), entry in test.items():

        sign = (
            1
            if family_a_is_agent(
                order, voice
            )
            else -1
        )

        aligned.extend((
            sign * entry["G"]["a"],
            -sign * entry["G"]["b"],
        ))

    print(
        "\n=== ROLE-ALIGNED GAP ==="
    )

    print(
        "Mean:",
        round(
            sum(aligned) / len(aligned), 4
        ),
    )

    print(
        "Positive:",
        sum(x > 0 for x in aligned),
        "/",
        len(aligned),
    )

    print(
        "Mean absolute magnitude:",
        round(
            sum(abs(x) for x in aligned)
            / len(aligned),
            4,
        ),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=(
            "design",
            "audit",
            "collect",
            "evaluate",
        ),
        default="design",
    )

    parser.add_argument(
        "--split",
        choices=tuple(PACKS),
        default="calibration",
    )

    parser.add_argument(
        "--model",
        choices=("llama3b", "qwen7b"),
        default="llama3b",
    )

    args = parser.parse_args()

    print(
        "ML-026 |",
        args.phase,
        "|",
        args.split,
        "|",
        args.model,
    )

    for split in PACKS:
        print(
            f"{split} SHA-256:",
            design_hash(split),
        )

    print(
        "Frozen predictions:",
        FROZEN_HYPOTHESES,
    )

    if args.phase == "design":
        for split in PACKS:
            for order, voice in product(
                ORDERS, VOICES
            ):
                print(
                    f"\n{split}: {order}, "
                    f"{voice}"
                )

                print(
                    make_passage(
                        PACKS[split],
                        ASSIGNMENTS[0],
                        order,
                        voice,
                        "base",
                        PACKS[split][
                            "verbs"
                        ][0],
                    )
                )

        print(
            "\n96 prompts per split."
        )

        print(
            "No inference performed."
        )

        return

    if args.phase == "evaluate":
        files = {}

        for split in PACKS:
            paths = sorted(
                Path("results").glob(
                    f"ml026_{args.model}_"
                    f"{split}_*.json"
                )
            )

            if not paths:
                raise FileNotFoundError(
                    f"Missing {split} results"
                )

            path = paths[-1]

            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                not data.get("complete")
                or data.get("design_hash")
                != design_hash(split)
            ):
                raise ValueError(
                    f"Invalid results: {path}"
                )

            files[split] = data

            print(
                f"Using {split}: {path}"
            )

        evaluate(
            files["calibration"],
            files["holdout"],
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

    prompts_by_case = {}

    for case in cases_for(args.split):
        prompts = prompts_for_case(
            tokenizer,
            args.split,
            case,
        )

        sites = audit_case(
            tokenizer,
            prompts,
            encode_prompt,
        )

        prompts_by_case[case] = prompts

        print(
            "AUDIT",
            case,
            sites,
        )

    print(
        "All alignment audits passed."
    )

    if args.phase == "audit":
        print(
            "No likelihood scoring performed."
        )
        return

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = output_dir / (
        f"ml026_{args.model}_"
        f"{args.split}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}"
        ".json"
    )

    data = {
        "experiment": "ML-026",
        "model": args.model,
        "split": args.split,
        "design_hash": design_hash(
            args.split
        ),
        "complete": False,
        "records": [],
    }

    correct = 0

    for case, prompts in prompts_by_case.items():
        aid, verb, order, voice = case

        for question, edit in product(
            QUESTIONS, EDITS
        ):
            scores = score_names(
                model,
                tokenizer,
                prompts[
                    (question, edit)
                ],
                log_likelihood,
            )

            predicted = max(
                scores,
                key=scores.get,
            )

            expected = gold(
                ASSIGNMENTS[aid],
                order,
                voice,
                edit,
                question,
            )

            correct += (
                predicted == expected
            )

            data["records"].append({
                "assignment": aid,
                "verb": verb,
                "order": order,
                "voice": voice,
                "question": question,
                "edit": edit,
                "scores": scores,
                "gold": expected,
                "forced": predicted,
            })

        output.write_text(
            json.dumps(
                data, indent=2
            ),
            encoding="utf-8",
        )

        print(
            f"DONE {case} | "
            f"records={len(data['records'])}"
        )

    data["complete"] = True

    output.write_text(
        json.dumps(
            data, indent=2
        ),
        encoding="utf-8",
    )

    print(
        f"Accuracy: {correct}/"
        f"{len(data['records'])}"
    )

    print("Saved:", output)


if __name__ == "__main__":
    main()
