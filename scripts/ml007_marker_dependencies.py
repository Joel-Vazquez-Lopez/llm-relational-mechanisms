
import argparse
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from mlx_lm import load

from ml003_role_margins import MODELS, log_likelihood


PAIRS = (
    ("Mira", "Leo"),
    ("Nora", "Felix"),
)

VERBS = ("praised", "tavved")
MARKERS = ("by", "near")

PROBES = (
    "second_actor",
    "first_actor",
    "second_mentioned",
)


def question_for(probe, first, second):
    if probe == "second_actor":
        return (
            f"Does the passage explicitly identify {second} "
            "as the person who performed the action?"
        )

    if probe == "first_actor":
        return (
            f"Does the passage explicitly identify {first} "
            "as the person who performed the action?"
        )

    if probe == "second_mentioned":
        return f"Does the passage mention {second}?"

    raise ValueError(probe)


def expected_answer(probe, marker):
    if probe == "second_actor":
        return "Yes" if marker == "by" else "No"

    if probe == "first_actor":
        return "No"

    if probe == "second_mentioned":
        return "Yes"

    raise ValueError(probe)


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

    model, tokenizer = load(
        str(MODELS / args.model)
    )

    records = []
    summaries = []

    for pair_id, (first, second) in enumerate(
        PAIRS[:args.pairs]
    ):
        for verb in VERBS:
            scores = {}

            for marker in MARKERS:
                passage = (
                    f"At dawn, {first} was "
                    f"{verb} {marker} {second}."
                )

                for probe in PROBES:
                    question = question_for(
                        probe, first, second
                    )

                    content = (
                        "Answer only Yes or No.\n\n"
                        f"Passage: {passage}\n"
                        f"Question: {question}\n"
                        "Answer:"
                    )

                    prompt = tokenizer.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": content,
                            }
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                    yes_lp, yes_tokens = log_likelihood(
                        model, tokenizer, prompt, "Yes"
                    )

                    no_lp, no_tokens = log_likelihood(
                        model, tokenizer, prompt, "No"
                    )

                    score = yes_lp - no_lp
                    scores[(marker, probe)] = score

                    expected = expected_answer(
                        probe, marker
                    )

                    predicted = (
                        "Yes" if score > 0 else "No"
                    )

                    records.append({
                        "pair_id": pair_id,
                        "first": first,
                        "second": second,
                        "verb": verb,
                        "marker": marker,
                        "probe": probe,
                        "passage": passage,
                        "question": question,
                        "prompt": prompt,
                        "yes_logprob": yes_lp,
                        "no_logprob": no_lp,
                        "yes_tokens": yes_tokens,
                        "no_tokens": no_tokens,
                        "yes_minus_no": score,
                        "expected": expected,
                        "preferred_candidate": predicted,
                        "candidate_correct": (
                            predicted == expected
                        ),
                    })

                    print(
                        f"{verb:8s} "
                        f"{marker:4s} "
                        f"{probe:17s} "
                        f"Yes-No={score:+.3f} "
                        f"expected={expected}"
                    )

            def marker_effect(probe):
                return (
                    scores[("by", probe)]
                    - scores[("near", probe)]
                )

            actor_effect = marker_effect(
                "second_actor"
            )

            first_control = marker_effect(
                "first_actor"
            )

            mention_control = marker_effect(
                "second_mentioned"
            )

            interaction = (
                actor_effect - mention_control
            )

            summary = {
                "pair_id": pair_id,
                "verb": verb,
                "actor_marker_effect": actor_effect,
                "first_actor_control_effect": (
                    first_control
                ),
                "mention_control_effect": (
                    mention_control
                ),
                "actor_specific_interaction": (
                    interaction
                ),
            }

            summaries.append(summary)

            print(
                f"\nPAIR {pair_id} / {verb}"
                f"\nActor marker effect: {actor_effect:+.3f}"
                f"\nFirst-actor control: {first_control:+.3f}"
                f"\nMention control: {mention_control:+.3f}"
                f"\nActor-specific interaction: {interaction:+.3f}\n"
            )

    checkpoint = next(
        item
        for item in json.loads(
            (MODELS / "RESTORATION.json").read_text()
        )
        if item["model"] == args.model
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = output_dir / (
        f"ml007_{args.model}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    output.write_text(
        json.dumps(
            {
                "experiment": "ML-007",
                "model": args.model,
                "revision": checkpoint["revision"],
                "mlx_lm_version": version("mlx-lm"),
                "records": records,
                "summaries": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
