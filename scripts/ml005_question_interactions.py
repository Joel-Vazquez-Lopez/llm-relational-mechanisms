
import argparse
import json
from collections import defaultdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from mlx_lm import load
from ml003_role_margins import MODELS, log_likelihood


ASSIGNMENTS = [
    ("Anika", "Boris"),
    ("Boris", "Anika"),
]

VERBS = {
    "praised": "praising",
    "daxed": "daxing",
    "wugged": "wugging",
}


def passage(agent, patient, voice, verb):
    if voice == "active":
        return f"At dawn, {agent} {verb} {patient}."
    return f"At dawn, {patient} was {verb} by {agent}."


def question(verb, gerund, role, style):
    if style == "generic":
        return (
            "At dawn, who performed the action?"
            if role == "agent"
            else "At dawn, who was acted on?"
        )

    return (
        f"At dawn, who did the {gerund}?"
        if role == "agent"
        else f"At dawn, who was {verb}?"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("llama3b", "qwen7b"),
        required=True,
    )
    parser.add_argument(
        "--pairs", type=int, choices=(1, 2), default=1
    )
    args = parser.parse_args()

    print("Loading:", args.model)
    model, tokenizer = load(str(MODELS / args.model))

    records = []

    for pair_id, (agent, patient) in enumerate(
        ASSIGNMENTS[:args.pairs]
    ):
        for verb, gerund in VERBS.items():
            for voice in ("active", "passive"):
                text = passage(agent, patient, voice, verb)

                for style in ("generic", "verb_specific"):
                    for role in ("agent", "patient"):
                        q = question(verb, gerund, role, style)

                        content = (
                            "Answer with exactly one person's name.\n\n"
                            f"Passage: {text}\n"
                            f"Question: {q}\n"
                            "Answer:"
                        )

                        formatted = tokenizer.apply_chat_template(
                            [{"role": "user", "content": content}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )

                        agent_lp, agent_tokens = log_likelihood(
                            model, tokenizer, formatted, agent
                        )
                        patient_lp, patient_tokens = log_likelihood(
                            model, tokenizer, formatted, patient
                        )

                        record = {
                            "pair_id": pair_id,
                            "agent": agent,
                            "patient": patient,
                            "verb": verb,
                            "voice": voice,
                            "style": style,
                            "role": role,
                            "passage": text,
                            "question": q,
                            "formatted_prompt": formatted,
                            "margin": agent_lp - patient_lp,
                            "agent_lp": agent_lp,
                            "patient_lp": patient_lp,
                            "agent_tokens": agent_tokens,
                            "patient_tokens": patient_tokens,
                        }
                        records.append(record)

    paired = defaultdict(dict)

    for record in records:
        key = (
            record["pair_id"],
            record["verb"],
            record["voice"],
            record["style"],
        )
        paired[key][record["role"]] = record["margin"]

    shifts = {}

    print("\n=== QUESTION-CONDITIONED ROLE SHIFTS ===")

    for key, margins in paired.items():
        assert set(margins) == {"agent", "patient"}

        shift = margins["agent"] - margins["patient"]
        shifts[key] = shift

        pair_id, verb, voice, style = key
        print(
            f"pair={pair_id} {verb:8s} "
            f"{voice:7s} {style:13s} "
            f"R={shift:+.3f}"
        )

    print("\n=== EFFECT OF VERB-SPECIFIC QUESTIONS ===")

    comparisons = []

    for pair_id in range(args.pairs):
        for verb in VERBS:
            for voice in ("active", "passive"):
                generic = shifts[
                    (pair_id, verb, voice, "generic")
                ]
                specific = shifts[
                    (pair_id, verb, voice, "verb_specific")
                ]

                effect = specific - generic

                comparisons.append({
                    "pair_id": pair_id,
                    "verb": verb,
                    "voice": voice,
                    "generic_shift": generic,
                    "specific_shift": specific,
                    "wording_effect": effect,
                })

                print(
                    f"pair={pair_id} {verb:8s} "
                    f"{voice:7s} "
                    f"specific-minus-generic={effect:+.3f}"
                )

    checkpoint = next(
        r for r in json.loads(
            (MODELS / "RESTORATION.json").read_text()
        )
        if r["model"] == args.model
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = output_dir / (
        f"ml005_{args.model}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    output.write_text(
        json.dumps({
            "experiment": "ML-005",
            "model": args.model,
            "revision": checkpoint["revision"],
            "mlx_lm_version": version("mlx-lm"),
            "records": records,
            "comparisons": comparisons,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
