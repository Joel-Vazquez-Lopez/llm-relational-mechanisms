
import argparse
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from mlx_lm import load
from ml003_role_margins import MODELS, log_likelihood


NAMES = ("Leo", "Nora")
VERBS = ("praised", "tavved")
ORDERS = ("event_first", "note_first")


def make_passage(actor, writer, verb, order):
    event = f"Mira was {verb} by {actor}"
    note = f"according to a note by {writer}"

    if order == "event_first":
        return f"At dawn, {event}, {note}."

    return (
        f"According to a note by {writer}, "
        f"at dawn {event}."
    )


def make_trials():
    for verb in VERBS:
        for actor in NAMES:
            writer = next(n for n in NAMES if n != actor)

            for order in ORDERS:
                passage = make_passage(
                    actor, writer, verb, order
                )

                assert passage.count(" by ") == 2
                assert passage.count(actor) == 1
                assert passage.count(writer) == 1

                for relation, question, gold in (
                    ("event", f"Who {verb} Mira?", actor),
                    ("note", "Who wrote the note?", writer),
                ):
                    yield {
                        "verb": verb,
                        "actor": actor,
                        "writer": writer,
                        "order": order,
                        "relation": relation,
                        "passage": passage,
                        "question": question,
                        "gold": gold,
                    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=("llama3b", "qwen7b"),
    )
    args = parser.parse_args()

    print("Loading:", args.model)
    model, tokenizer = load(str(MODELS / args.model))

    records = []

    for index, trial in enumerate(make_trials(), 1):
        content = (
            "Answer with exactly one person's name.\n\n"
            f"Passage: {trial['passage']}\n"
            f"Question: {trial['question']}\n"
            "Answer:"
        )

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

        leo_lp, leo_tokens = log_likelihood(
            model, tokenizer, prompt, "Leo"
        )
        nora_lp, nora_tokens = log_likelihood(
            model, tokenizer, prompt, "Nora"
        )

        margin = leo_lp - nora_lp

        if margin > 0:
            predicted = "Leo"
        elif margin < 0:
            predicted = "Nora"
        else:
            predicted = "Tie"

        first_by = (
            trial["actor"]
            if trial["order"] == "event_first"
            else trial["writer"]
        )

        last_by = (
            trial["writer"]
            if trial["order"] == "event_first"
            else trial["actor"]
        )

        record = {
            **trial,
            "prompt": prompt,
            "leo_logprob": leo_lp,
            "nora_logprob": nora_lp,
            "leo_tokens": leo_tokens,
            "nora_tokens": nora_tokens,
            "leo_minus_nora": margin,
            "predicted": predicted,
            "correct": predicted == trial["gold"],
            "first_by_prediction": first_by,
            "last_by_prediction": last_by,
        }

        records.append(record)

        print(
            f"{index:02d} "
            f"{trial['verb']:7s} "
            f"{trial['order']:11s} "
            f"{trial['relation']:5s} "
            f"gold={trial['gold']:4s} "
            f"pred={predicted:4s} "
            f"margin={margin:+.3f} "
            f"tokens=({leo_tokens},{nora_tokens})"
        )

    print("\n=== COMPARISON WITH BASELINES ===")

    def accuracy(key):
        return sum(
            r[key] == r["gold"]
            for r in records
        )

    print("Relational candidate selection:",
          f"{accuracy('predicted')}/{len(records)}")

    print("First-by heuristic:",
          f"{accuracy('first_by_prediction')}/{len(records)}")

    print("Last-by heuristic:",
          f"{accuracy('last_by_prediction')}/{len(records)}")

    print("\n=== PAIRED QUESTION EFFECTS ===")

    for verb in VERBS:
        for actor in NAMES:
            for order in ORDERS:
                pair = [
                    r for r in records
                    if r["verb"] == verb
                    and r["actor"] == actor
                    and r["order"] == order
                ]

                assert len(pair) == 2
                indexed = {
                    r["relation"]: r
                    for r in pair
                }

                event_margin = indexed["event"]["leo_minus_nora"]
                note_margin = indexed["note"]["leo_minus_nora"]

                shift = event_margin - note_margin

                expected_direction = (
                    1 if actor == "Leo" else -1
                )

                print(
                    f"{verb:7s} actor={actor:4s} "
                    f"{order:11s} "
                    f"event={event_margin:+.3f} "
                    f"note={note_margin:+.3f} "
                    f"shift={shift:+.3f} "
                    f"expected_direction="
                    f"{'positive' if expected_direction > 0 else 'negative'}"
                )

    checkpoint = next(
        item for item in json.loads(
            (MODELS / "RESTORATION.json").read_text()
        )
        if item["model"] == args.model
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = output_dir / (
        f"ml009_{args.model}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    output.write_text(
        json.dumps({
            "experiment": "ML-009",
            "model": args.model,
            "revision": checkpoint["revision"],
            "mlx_lm_version": version("mlx-lm"),
            "records": records,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
