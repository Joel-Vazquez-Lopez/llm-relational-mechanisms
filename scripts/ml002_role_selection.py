
import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from importlib.metadata import version
from itertools import product
from pathlib import Path

from mlx_lm import load, generate


ROOT = (
    Path.home()
    / "Documents/ChatGPT/project null"
    / "llm-relational-mechanisms/.local_models"
)

NAME_PAIRS = [
    ("Anika", "Boris"),
    ("Clara", "David"),
]

VOICES = ("active", "passive")
ROLES = ("agent", "patient")


def make_trials():
    for pair, swap, voice, role in product(
        NAME_PAIRS, (False, True), VOICES, ROLES
    ):
        agent, patient = pair[::-1] if swap else pair

        if voice == "active":
            sentence = f"{agent} praised {patient}"
            first_mentioned = agent
        else:
            sentence = f"{patient} was praised by {agent}"
            first_mentioned = patient

        question = (
            "At dawn, who did the praising?"
            if role == "agent"
            else "At dawn, who was praised?"
        )

        yield {
            "voice": voice,
            "role": role,
            "agent": agent,
            "patient": patient,
            "first_mentioned": first_mentioned,
            "gold": agent if role == "agent" else patient,
            "passage": f"At dawn, {sentence}.",
            "question": question,
        }


def parse_answer(response, candidates):
    cleaned = response.strip().strip("\"'")
    cleaned = re.sub(r"[.!?]+\s*$", "", cleaned).strip()

    for name in candidates:
        if cleaned.casefold() == name.casefold():
            return name

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("llama3b", "qwen7b"),
        required=True,
    )
    args = parser.parse_args()

    model_dir = ROOT / args.model
    records = []

    print("Loading:", args.model)
    model, tokenizer = load(str(model_dir))

    for index, trial in enumerate(make_trials(), start=1):
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

        response = generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=16,
            verbose=False,
        )

        answer = parse_answer(
            response,
            (trial["agent"], trial["patient"]),
        )

        record = {
            "case_id": index,
            "model": args.model,
            **trial,
            "prompt": formatted,
            "raw_response": response,
            "parsed_answer": answer,
            "correct": answer == trial["gold"],
            "first_mentioned_match": (
                answer == trial["first_mentioned"]
            ),
        }

        records.append(record)

        print(
            f"{index:02d} | "
            f"{trial['voice']:7s} | "
            f"{trial['role']:7s} | "
            f"gold={trial['gold']:6s} | "
            f"answer={str(answer):6s} | "
            f"correct={record['correct']}"
        )

    print("\n=== RESULTS ===")

    valid = sum(r["parsed_answer"] is not None for r in records)
    correct = sum(r["correct"] for r in records)
    first = sum(r["first_mentioned_match"] for r in records)

    print(f"Valid answers: {valid}/{len(records)}")
    print(f"Relational accuracy: {correct}/{len(records)}")
    print(f"First-mentioned agreement: {first}/{len(records)}")

    by_condition = defaultdict(list)

    for record in records:
        key = (record["voice"], record["role"])
        by_condition[key].append(record)

    for key, group in sorted(by_condition.items()):
        n_correct = sum(r["correct"] for r in group)
        n_first = sum(r["first_mentioned_match"] for r in group)

        print(
            f"{key}: correct={n_correct}/{len(group)}, "
            f"first-mentioned={n_first}/{len(group)}"
        )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    output = output_dir / f"ml002_{args.model}_{stamp}.json"

    metadata = json.loads(
        (ROOT / "RESTORATION.json").read_text()
    )
    restoration = next(
        r for r in metadata if r["model"] == args.model
    )

    output.write_text(
        json.dumps(
            {
                "experiment": "ML-002",
                "model": args.model,
                "revision": restoration["revision"],
                "mlx_lm_version": version("mlx-lm"),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
