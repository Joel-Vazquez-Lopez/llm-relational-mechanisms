
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

CONDITIONS = (
    "praised",
    "daxed",
    "wugged",
    "no_action",
)

VOICES = ("active", "passive")

QUESTIONS = {
    "agent": "At dawn, who performed the action?",
    "patient": "At dawn, who was acted on?",
}


def make_passage(agent, patient, voice, condition):
    if voice == "active":
        first, second = agent, patient
    else:
        first, second = patient, agent

    if condition == "no_action":
        return (
            f"At dawn, {first} and {second} "
            "were present."
        )

    if voice == "active":
        return (
            f"At dawn, {agent} "
            f"{condition} {patient}."
        )

    return (
        f"At dawn, {patient} "
        f"was {condition} by {agent}."
    )


def generate_trials(n_assignments):
    for pair_id, (agent, patient) in enumerate(
        ASSIGNMENTS[:n_assignments]
    ):
        for voice in VOICES:
            for condition in CONDITIONS:
                passage = make_passage(
                    agent, patient, voice, condition
                )

                for role, question in QUESTIONS.items():
                    yield {
                        "pair_id": pair_id,
                        "agent": agent,
                        "patient": patient,
                        "voice": voice,
                        "condition": condition,
                        "role": role,
                        "passage": passage,
                        "question": question,
                    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("llama3b", "qwen7b"),
        required=True,
    )
    parser.add_argument(
        "--pairs",
        type=int,
        choices=(1, 2),
        default=1,
    )
    args = parser.parse_args()

    print("Loading:", args.model)
    model, tokenizer = load(str(MODELS / args.model))

    records = []

    for i, trial in enumerate(
        generate_trials(args.pairs), start=1
    ):
        prompt = (
            "Answer with exactly one person's name.\n\n"
            f"Passage: {trial['passage']}\n"
            f"Question: {trial['question']}\n"
            "Answer:"
        )

        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        agent_lp, agent_tokens = log_likelihood(
            model, tokenizer, formatted, trial["agent"]
        )

        patient_lp, patient_tokens = log_likelihood(
            model, tokenizer, formatted, trial["patient"]
        )

        margin = agent_lp - patient_lp

        records.append({
            **trial,
            "formatted_prompt": formatted,
            "agent_logprob": agent_lp,
            "patient_logprob": patient_lp,
            "agent_tokens": agent_tokens,
            "patient_tokens": patient_tokens,
            "agent_margin": margin,
        })

        print(
            f"{i:02d} "
            f"{trial['condition']:9s} "
            f"{trial['voice']:7s} "
            f"{trial['role']:7s} "
            f"margin={margin:+.3f}"
        )

    paired = defaultdict(dict)

    for record in records:
        key = (
            record["pair_id"],
            record["voice"],
            record["condition"],
        )
        paired[key][record["role"]] = record

    contrasts = []

    print("\n=== ROLE-ACCESS SHIFTS ===")

    for key, group in paired.items():
        assert set(group) == {"agent", "patient"}

        shift = (
            group["agent"]["agent_margin"]
            - group["patient"]["agent_margin"]
        )

        contrasts.append({
            "pair_id": key[0],
            "voice": key[1],
            "condition": key[2],
            "role_shift": shift,
        })

        print(
            f"pair={key[0]} "
            f"{key[1]:7s} "
            f"{key[2]:9s} "
            f"R={shift:+.3f}"
        )

    lookup = {
        (c["pair_id"], c["voice"], c["condition"]):
            c["role_shift"]
        for c in contrasts
    }

    print("\n=== INCREMENT OVER NO-ACTION CONTROL ===")

    for c in contrasts:
        if c["condition"] == "no_action":
            continue

        neutral = lookup[
            (c["pair_id"], c["voice"], "no_action")
        ]

        delta = c["role_shift"] - neutral
        c["over_neutral"] = delta

        print(
            f"pair={c['pair_id']} "
            f"{c['voice']:7s} "
            f"{c['condition']:9s} "
            f"delta_R={delta:+.3f}"
        )

    print("\n=== MEAN ROLE SHIFTS ===")

    for condition in CONDITIONS:
        values = [
            c["role_shift"]
            for c in contrasts
            if c["condition"] == condition
        ]
        mean = sum(values) / len(values)
        print(f"{condition:9s}: {mean:+.3f}")

    metadata = json.loads(
        (MODELS / "RESTORATION.json").read_text()
    )
    checkpoint = next(
        x for x in metadata
        if x["model"] == args.model
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    stamp = datetime.now().strftime(
        "%Y%m%dT%H%M%S%f"
    )

    output = output_dir / (
        f"ml004_{args.model}_{stamp}.json"
    )

    output.write_text(
        json.dumps({
            "experiment": "ML-004",
            "model": args.model,
            "revision": checkpoint["revision"],
            "mlx_lm_version": version("mlx-lm"),
            "assignments": args.pairs,
            "records": records,
            "contrasts": contrasts,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
