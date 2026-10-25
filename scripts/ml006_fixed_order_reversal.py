
import argparse
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from mlx_lm import load
from ml003_role_margins import MODELS, log_likelihood


PAIRS = [
    ("Mira", "Leo"),
    ("Nora", "Felix"),
]

VERBS = ("praised", "tavved")

QUESTIONS = {
    "agent": "At dawn, who performed the action?",
    "patient": "At dawn, who was acted on?",
    "first": "At dawn, which person was mentioned first?",
    "second": "At dawn, which person was mentioned second?",
}


def make_passage(first, second, verb, voice):
    if voice == "active":
        return f"At dawn, {first} {verb} {second}."
    return f"At dawn, {first} was {verb} by {second}."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=("llama3b", "qwen7b"),
    )
    parser.add_argument(
        "--pairs", type=int, choices=(1, 2), default=1
    )
    args = parser.parse_args()

    model_dir = MODELS / args.model
    print("Loading:", model_dir)

    model, tokenizer = load(str(model_dir))
    records = []
    summaries = []

    for pair_id, (first, second) in enumerate(
        PAIRS[:args.pairs]
    ):
        for verb in VERBS:
            margins = {}

            for voice in ("active", "passive"):
                passage = make_passage(
                    first, second, verb, voice
                )

                assert passage.index(first) < passage.index(second)

                for role, question in QUESTIONS.items():
                    content = (
                        "Answer with exactly one person's name.\n\n"
                        f"Passage: {passage}\n"
                        f"Question: {question}\n"
                        "Answer:"
                    )

                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": content}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                    first_lp, first_tokens = log_likelihood(
                        model, tokenizer, prompt, first
                    )
                    second_lp, second_tokens = log_likelihood(
                        model, tokenizer, prompt, second
                    )

                    margin = first_lp - second_lp
                    margins[(voice, role)] = margin

                    if role == "agent":
                        expected = (
                            first if voice == "active"
                            else second
                        )
                    elif role == "patient":
                        expected = (
                            second if voice == "active"
                            else first
                        )
                    elif role == "first":
                        expected = first
                    else:
                        expected = second

                    records.append({
                        "pair_id": pair_id,
                        "first": first,
                        "second": second,
                        "verb": verb,
                        "voice": voice,
                        "role": role,
                        "passage": passage,
                        "question": question,
                        "prompt": prompt,
                        "expected": expected,
                        "first_logprob": first_lp,
                        "second_logprob": second_lp,
                        "first_tokens": first_tokens,
                        "second_tokens": second_tokens,
                        "first_minus_second": margin,
                    })

            def voice_effect(role):
                return (
                    margins[("active", role)]
                    - margins[("passive", role)]
                )

            agent_effect = voice_effect("agent")
            patient_effect = voice_effect("patient")
            first_effect = voice_effect("first")
            second_effect = voice_effect("second")

            interaction = agent_effect - patient_effect

            summary = {
                "pair_id": pair_id,
                "verb": verb,
                "first": first,
                "second": second,
                "agent_voice_effect": agent_effect,
                "patient_voice_effect": patient_effect,
                "first_voice_effect": first_effect,
                "second_voice_effect": second_effect,
                "role_voice_interaction": interaction,
            }
            summaries.append(summary)

            print(f"\n=== {first}, {second}: {verb} ===")

            for voice in ("active", "passive"):
                for role in QUESTIONS:
                    print(
                        f"{voice:7s} {role:7s} "
                        f"first-minus-second="
                        f"{margins[(voice, role)]:+.3f}"
                    )

            print(
                "Agent voice effect:",
                f"{agent_effect:+.3f}"
            )
            print(
                "Patient voice effect:",
                f"{patient_effect:+.3f}"
            )
            print(
                "First-position voice effect:",
                f"{first_effect:+.3f}"
            )
            print(
                "Second-position voice effect:",
                f"{second_effect:+.3f}"
            )
            print(
                "Role-by-voice interaction:",
                f"{interaction:+.3f}"
            )

    metadata = json.loads(
        (MODELS / "RESTORATION.json").read_text()
    )
    checkpoint = next(
        item for item in metadata
        if item["model"] == args.model
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = output_dir / (
        f"ml006_{args.model}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    output.write_text(
        json.dumps({
            "experiment": "ML-006",
            "model": args.model,
            "revision": checkpoint["revision"],
            "mlx_lm_version": version("mlx-lm"),
            "records": records,
            "summaries": summaries,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
