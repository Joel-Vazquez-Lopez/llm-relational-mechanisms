
import argparse
import json
from datetime import datetime
from pathlib import Path

from mlx_lm import load, generate


ARCHIVE = (
    Path.home()
    / "Documents/ChatGPT/project null"
    / "llm-relational-mechanisms/.local_models"
)

CASES = [
    ("active", "A_agent",
     "At dawn, Anika praised Boris.", "Anika"),

    ("passive", "A_agent",
     "At dawn, Boris was praised by Anika.", "Anika"),

    ("active", "B_agent",
     "At dawn, Boris praised Anika.", "Boris"),

    ("passive", "B_agent",
     "At dawn, Anika was praised by Boris.", "Boris"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["llama3b", "qwen7b"],
        default="llama3b",
    )
    args = parser.parse_args()

    model_dir = ARCHIVE / args.model

    print(f"Loading {args.model} from {model_dir}")
    model, tokenizer = load(str(model_dir))

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    output = Path("results") / f"ml001_{args.model}_{run_id}.jsonl"
    output.parent.mkdir(exist_ok=True)

    correct = 0

    with output.open("x", encoding="utf-8") as handle:
        for index, (voice, agent_condition, passage, gold) in enumerate(CASES):
            question = "Who did the praising at dawn?"

            prompt = (
                "Answer with exactly one person's name.\n\n"
                f"Passage: {passage}\n"
                f"Question: {question}\n"
                "Answer:"
            )

            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

            answer = generate(
                model,
                tokenizer,
                prompt=formatted,
                max_tokens=12,
                verbose=False,
            )

            valid = answer.strip() in {"Anika", "Boris"}
            is_correct = valid and answer.strip() == gold
            correct += int(is_correct)

            record = {
                "case_id": index,
                "model": args.model,
                "voice": voice,
                "agent_condition": agent_condition,
                "passage": passage,
                "question": question,
                "prompt": formatted,
                "gold": gold,
                "response": answer,
                "valid": valid,
                "correct": is_correct,
            }

            handle.write(json.dumps(record) + "\n")

            print(f"\nCase {index + 1}")
            print("Passage:", passage)
            print("Gold:", gold)
            print("Model:", repr(answer))
            print("Correct:", is_correct)

    print(f"\nCorrect: {correct}/{len(CASES)}")
    print("Results saved:", output)


if __name__ == "__main__":
    main()
