
import argparse
import json
import re
from datetime import datetime
from importlib.metadata import version
from itertools import product
from pathlib import Path

from mlx_lm import load, generate

from ml003_role_margins import MODELS, log_likelihood


NAMES = ("Leo", "Felix", "Nora", "Clara")
ACTORS = ("Leo", "Felix")
WRITERS = ("Nora", "Clara")
VERB = "mepked"

QUESTIONS = {
    "actor": f"At dawn, who {VERB} Mira?",
    "note": "Who wrote the note?",
}


def make_passage(actor, writer, construction):
    other_actor = next(n for n in ACTORS if n != actor)
    other_writer = next(n for n in WRITERS if n != writer)

    if construction == "passive_bridge":
        passage = (
            f"At dawn, Mira was {VERB} by the person "
            "who wrote the letter. "
            f"The letter was written by {actor}. "
            f"{other_actor} watched. "
            f"According to a note by {writer}, "
            f"{other_writer} was nearby."
        )

    elif construction == "active_bridge":
        passage = (
            f"According to a note by {writer}, "
            f"{other_writer} was nearby. "
            f"The person who wrote the letter {VERB} "
            "Mira at dawn. "
            f"{other_actor} watched. "
            f"{actor} wrote the letter."
        )

    else:
        raise ValueError(construction)

    assert all(passage.count(name) == 1 for name in NAMES)
    assert passage.count("Mira") == 1

    return passage


def parse_native(response):
    cleaned = response.strip().strip('"\'')
    cleaned = re.sub(r"[.!?,;:\s]+$", "", cleaned).strip()
    return next(
        (n for n in NAMES if n.casefold() == cleaned.casefold()),
        None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("llama3b", "qwen7b"),
        required=True,
    )
    parser.add_argument(
        "--construction",
        choices=("passive_bridge", "active_bridge"),
        default="passive_bridge",
    )
    args = parser.parse_args()

    model_dir = MODELS / args.model
    print("Loading:", args.model)
    model, tokenizer = load(str(model_dir))

    records = []
    lookup = {}

    for actor, writer, role in product(
        ACTORS, WRITERS, QUESTIONS
    ):
        passage = make_passage(
            actor, writer, args.construction
        )
        gold = actor if role == "actor" else writer

        content = (
            "Answer with exactly one person's name.\n\n"
            f"Passage: {passage}\n"
            f"Question: {QUESTIONS[role]}\n"
            "Answer:"
        )

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

        scores = {}
        token_counts = {}

        for name in NAMES:
            score, count = log_likelihood(
                model, tokenizer, prompt, name
            )
            scores[name] = score
            token_counts[name] = count

        preferred = max(NAMES, key=lambda n: scores[n])

        native = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=12,
            verbose=False,
        )
        native_answer = parse_native(native)

        record = {
            "actor": actor,
            "writer": writer,
            "role": role,
            "construction": args.construction,
            "passage": passage,
            "question": QUESTIONS[role],
            "prompt": prompt,
            "gold": gold,
            "logprobs": scores,
            "token_counts": token_counts,
            "preferred": preferred,
            "native_response": native,
            "native_answer": native_answer,
            "actor_margin": scores["Leo"] - scores["Felix"],
            "writer_margin": scores["Nora"] - scores["Clara"],
        }

        records.append(record)
        lookup[(actor, writer, role)] = record

        print(
            f"{role:5s} actor={actor:5s} "
            f"writer={writer:5s} "
            f"gold={gold:5s} "
            f"forced={preferred:5s} "
            f"native={str(native_answer):5s} "
            f"A={record['actor_margin']:+.3f} "
            f"W={record['writer_margin']:+.3f}"
        )

    print("\n=== COUNTERFACTUAL ACTOR SWITCHES ===")

    for writer in WRITERS:
        target = (
            lookup[("Leo", writer, "actor")]["actor_margin"]
            - lookup[("Felix", writer, "actor")]["actor_margin"]
        )

        control = (
            lookup[("Leo", writer, "note")]["actor_margin"]
            - lookup[("Felix", writer, "note")]["actor_margin"]
        )

        print(
            f"writer={writer:5s} "
            f"actor_question_change={target:+.3f} "
            f"note_question_control={control:+.3f} "
            f"selectivity={target - control:+.3f}"
        )

    print("\n=== COUNTERFACTUAL WRITER SWITCHES ===")

    for actor in ACTORS:
        target = (
            lookup[(actor, "Nora", "note")]["writer_margin"]
            - lookup[(actor, "Clara", "note")]["writer_margin"]
        )

        control = (
            lookup[(actor, "Nora", "actor")]["writer_margin"]
            - lookup[(actor, "Clara", "actor")]["writer_margin"]
        )

        print(
            f"actor={actor:5s} "
            f"note_question_change={target:+.3f} "
            f"actor_question_control={control:+.3f} "
            f"selectivity={target - control:+.3f}"
        )

    forced_correct = sum(
        r["preferred"] == r["gold"] for r in records
    )
    native_correct = sum(
        r["native_answer"] == r["gold"] for r in records
    )

    print("\n=== SUMMARY ===")
    print(f"Forced candidate selection: {forced_correct}/8")
    print(f"Native generation: {native_correct}/8")
    print("Candidate token counts:", records[0]["token_counts"])

    checkpoint = next(
        x for x in json.loads(
            (MODELS / "RESTORATION.json").read_text()
        )
        if x["model"] == args.model
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    path = output_dir / (
        f"ml010_{args.model}_{args.construction}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    path.write_text(
        json.dumps(
            {
                "experiment": "ML-010",
                "model": args.model,
                "revision": checkpoint["revision"],
                "mlx_lm_version": version("mlx-lm"),
                "construction": args.construction,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Saved:", path)


if __name__ == "__main__":
    main()
