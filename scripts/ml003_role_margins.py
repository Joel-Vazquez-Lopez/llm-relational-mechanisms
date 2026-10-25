
import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import mlx.core as mx
from mlx_lm import load


MODELS = (
    Path.home()
    / "Documents/ChatGPT/project null"
    / "llm-relational-mechanisms/.local_models"
)


def log_likelihood(model, tokenizer, prompt, answer):
    # Match MLX-LM's handling of already-formatted chat prompts.
    bos = tokenizer.bos_token
    add_special = bos is None or not prompt.startswith(bos)

    prefix = tokenizer.encode(
        prompt,
        add_special_tokens=add_special,
    )

    full = tokenizer.encode(
        prompt + answer,
        add_special_tokens=add_special,
    )

    # A necessary guard against tokenization changing
    # across the prompt/answer boundary.
    if full[:len(prefix)] != prefix:
        raise ValueError(
            "Tokenizer boundary changed for answer "
            f"{answer!r}; inspect the prompt before scoring."
        )

    suffix = full[len(prefix):]

    if not suffix:
        raise ValueError("Empty answer continuation")

    tokens = mx.array([full[:-1]], dtype=mx.int32)
    logits = model(tokens)

    start = len(prefix) - 1
    selected_logits = logits[
        0,
        start:start + len(suffix),
        :
    ].astype(mx.float32)

    log_probs = (
        selected_logits
        - mx.logsumexp(
            selected_logits,
            axis=-1,
            keepdims=True,
        )
    )

    targets = mx.array(
        suffix,
        dtype=mx.int32,
    )[:, None]

    selected = mx.take_along_axis(
        log_probs,
        targets,
        axis=-1,
    )

    score = float(mx.sum(selected).item())
    mx.clear_cache()

    return score, len(suffix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("llama3b", "qwen7b"),
        required=True,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    inputs = sorted(
        Path("results").glob(
            f"ml002_{args.model}_*.json"
        )
    )

    if not inputs:
        raise FileNotFoundError(
            "No ML-002 results found"
        )

    source = inputs[-1]
    data = json.loads(source.read_text())

    if data["model"] != args.model:
        raise ValueError("Model metadata mismatch")

    print("Source:", source)
    print("Loading:", args.model)

    model, tokenizer = load(
        str(MODELS / args.model)
    )

    scored = []

    for record in data["records"][:args.limit]:
        prompt = record["prompt"]

        agent_lp, agent_tokens = log_likelihood(
            model,
            tokenizer,
            prompt,
            record["agent"],
        )

        patient_lp, patient_tokens = log_likelihood(
            model,
            tokenizer,
            prompt,
            record["patient"],
        )

        # Positive means the model prefers
        # the agent over the patient.
        margin = agent_lp - patient_lp

        result = {
            **record,
            "agent_logprob": agent_lp,
            "patient_logprob": patient_lp,
            "agent_tokens": agent_tokens,
            "patient_tokens": patient_tokens,
            "agent_margin": margin,
        }
        scored.append(result)

        print(
            f"{record['case_id']:02d} "
            f"{record['voice']:7s} "
            f"{record['role']:7s} "
            f"margin={margin:+.3f} "
            f"tokens=({agent_tokens},{patient_tokens})"
        )

    paired = defaultdict(dict)

    for record in scored:
        key = (
            record["passage"],
            record["voice"],
            record["agent"],
            record["patient"],
        )
        paired[key][record["role"]] = record

    contrasts = []

    print("\n=== ROLE-CONDITIONED CONTRASTS ===")

    for key, group in paired.items():
        if set(group) != {"agent", "patient"}:
            continue

        shift = (
            group["agent"]["agent_margin"]
            - group["patient"]["agent_margin"]
        )

        result = {
            "passage": key[0],
            "voice": key[1],
            "agent": key[2],
            "patient": key[3],
            "role_access_shift": shift,
        }

        contrasts.append(result)

        print(
            f"{key[1]:7s} "
            f"{key[2]:6s}->{key[3]:6s} "
            f"role_shift={shift:+.3f}"
        )

    print("\n=== BY VOICE ===")

    for voice in ("active", "passive"):
        values = [
            c["role_access_shift"]
            for c in contrasts
            if c["voice"] == voice
        ]

        if values:
            mean = sum(values) / len(values)
            positive = sum(v > 0 for v in values)
            print(
                f"{voice}: mean={mean:+.3f}, "
                f"positive={positive}/{len(values)}"
            )

    output = (
        Path("results")
        / (
            f"ml003_{args.model}_"
            f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
        )
    )

    output.write_text(
        json.dumps(
            {
                "experiment": "ML-003",
                "source": str(source),
                "model": args.model,
                "revision": data["revision"],
                "scored": scored,
                "contrasts": contrasts,
            },
            indent=2,
        )
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
