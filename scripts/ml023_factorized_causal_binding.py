
"""ML-023: Factorized Causal Binding."""

import argparse
import hashlib
import json

from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path


WRITER, CARRIER = "Leo", "Felix"
FOLDER, PAINTER = "Nora", "Clara"

NAMES = (WRITER, CARRIER, FOLDER, PAINTER)

# Prespecified from previous research, not selected
# using the results of this experiment.
LAYERS = (0, 7, 13)

QUESTIONS = {
    "agent": "Who helped the other person?",
    "patient": "Who was helped?",
}

SWAPS = {
    "base": ("wrote", "folded"),
    "agent": ("carried", "folded"),
    "patient": ("wrote", "painted"),
    "both": ("carried", "painted"),
}


def passage(agent_selector, patient_selector):
    return (
        f"At dusk, the person who {agent_selector} "
        f"the letter helped the person who "
        f"{patient_selector} the map. "
        "Leo wrote the letter. "
        "Felix carried the letter. "
        "Nora folded the map. "
        "Clara painted the map."
    )


def prompt_for(tokenizer, question, agent_sel, patient_sel):
    content = (
        "Answer using exactly one person's name.\n\n"
        f"Passage: {passage(agent_sel, patient_sel)}\n"
        f"Question: {QUESTIONS[question]}\n"
        "Answer:"
    )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def replacement_position(reference, changed, label):
    edits = [
        edit
        for edit in SequenceMatcher(
            None, reference, changed, autojunk=False
        ).get_opcodes()
        if edit[0] != "equal"
    ]

    valid = (
        len(edits) == 1
        and edits[0][0] == "replace"
        and edits[0][2] - edits[0][1] == 1
        and edits[0][4] - edits[0][3] == 1
        and edits[0][1] == edits[0][3]
        and len(reference) == len(changed)
    )

    if not valid:
        raise ValueError(
            f"{label}: expected a single-token, "
            f"position-aligned substitution. "
            f"Actual edits: {edits}. "
            "Do not perform invalid activation patches."
        )

    return edits[0][1]


def audit_token_alignment(
    tokenizer, prompts, encode_prompt
):
    ids = {
        name: encode_prompt(tokenizer, prompt)
        for name, prompt in prompts.items()
    }

    base = ids["base"]

    agent_pos = replacement_position(
        base, ids["agent"], "agent"
    )

    patient_pos = replacement_position(
        base, ids["patient"], "patient"
    )

    if not agent_pos < patient_pos:
        raise ValueError("Unexpected dependency positions")

    both_differences = [
        i
        for i, (original, changed) in enumerate(
            zip(base, ids["both"])
        )
        if original != changed
    ]

    if (
        len(ids["both"]) != len(base)
        or both_differences != [agent_pos, patient_pos]
    ):
        raise ValueError(
            "Combined edit has invalid token alignment: "
            f"{both_differences}"
        )

    if (
        ids["both"][agent_pos] != ids["agent"][agent_pos]
        or ids["both"][patient_pos]
        != ids["patient"][patient_pos]
    ):
        raise ValueError("Unexpected combined edit tokens")

    if any(len(tokens) != len(base) for tokens in ids.values()):
        raise ValueError("Prompts have different token lengths")

    # A location preceding both edits.
    # In a causal transformer, it cannot attend forward.
    prefix_pos = max(0, agent_pos - 2)

    print("Token count:", len(base))
    print("Agent cue:", agent_pos)
    print("Patient cue:", patient_pos)
    print("Prefix control:", prefix_pos)

    print(
        "Agent tokens:",
        repr(tokenizer.decode([base[agent_pos]])),
        "->",
        repr(tokenizer.decode([ids["agent"][agent_pos]])),
    )

    print(
        "Patient tokens:",
        repr(tokenizer.decode([base[patient_pos]])),
        "->",
        repr(tokenizer.decode([ids["patient"][patient_pos]])),
    )

    return {
        "agent": agent_pos,
        "patient": patient_pos,
        "prefix": prefix_pos,
    }


def margins(model, tokenizer, prompt, log_likelihood):
    scores = {}

    for name in NAMES:
        lp, _ = log_likelihood(
            model, tokenizer, prompt, name
        )
        scores[name] = float(lp)

    return {
        "agent": scores[WRITER] - scores[CARRIER],
        "patient": scores[FOLDER] - scores[PAINTER],
    }


def difference(after, before):
    return {
        role: after[role] - before[role]
        for role in ("agent", "patient")
    }



def capture(
    model, tokenizer, prompt,
    layer_index, positions, encode_prompt
):
    import mlx.core as mx
    import mlx.nn as nn

    class Capture(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.layer = original
            self.use_sliding = getattr(
                original, "use_sliding", False
            )
            self.value = None

        def __call__(self, *args, **kwargs):
            output = self.layer(*args, **kwargs)

            self.value = {
                name: output[0, position, :]
                for name, position in positions.items()
            }

            return output

    layers = model.model.layers
    original = layers[layer_index]
    wrapper = Capture(original)

    layers[layer_index] = wrapper

    try:
        tokens = mx.array(
            [encode_prompt(tokenizer, prompt)],
            dtype=mx.int32,
        )

        logits = model(tokens)
        mx.eval(logits)

        if wrapper.value is None:
            raise RuntimeError("Layer capture failed")

        mx.eval(*wrapper.value.values())

        result = dict(wrapper.value)

    finally:
        layers[layer_index] = original

    return result



def patch_margins(
    model, tokenizer, prompt,
    layer_index, patch_map, log_likelihood
):
    import mlx.core as mx
    import mlx.nn as nn

    class Patch(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.layer = original
            self.use_sliding = getattr(
                original, "use_sliding", False
            )

        def __call__(self, *args, **kwargs):
            output = self.layer(*args, **kwargs)

            chunks = []
            previous = 0

            for position, value in sorted(
                patch_map.items()
            ):
                chunks.append(
                    output[:, previous:position, :]
                )

                chunks.append(
                    value[None, None, :].astype(
                        output.dtype
                    )
                )

                previous = position + 1

            chunks.append(output[:, previous:, :])

            return mx.concatenate(chunks, axis=1)

    layers = model.model.layers
    original = layers[layer_index]

    layers[layer_index] = Patch(original)

    try:
        result = margins(
            model, tokenizer, prompt, log_likelihood
        )

    finally:
        layers[layer_index] = original

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=("llama3b", "qwen7b"),
        default="llama3b",
    )

    parser.add_argument(
        "--phase",
        choices=("audit", "behavior", "patch"),
        default="audit",
    )

    args = parser.parse_args()

    design = {
        "experiment": "ML-023",
        "names": NAMES,
        "stimuli": {
            question: {
                label: {
                    "passage": passage(*selectors),
                    "question": question_text,
                }
                for label, selectors in SWAPS.items()
            }
            for question, question_text in QUESTIONS.items()
        },
        "layers": LAYERS,
        "instruction": (
            "Answer using exactly one person's name."
        ),
    }

    digest = hashlib.sha256(
        json.dumps(
            design, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()

    print("Experiment: ML-023")
    print("Model:", args.model)
    print("Phase:", args.phase)
    print("Design SHA-256:", digest)

    print("\nPrespecified predictions:")
    print("1. Agent edit shifts agent margin toward carrier.")
    print("2. Patient edit shifts patient margin toward painter.")
    print("3. Single-cue patches preferentially affect their role.")
    print("4. Double patches allow a factorial interaction test.")
    print("5. No single-site patch proves a binding mechanism.")

    from mlx_lm import load
    from ml003_role_margins import (
        MODELS, log_likelihood
    )
    from ml012_cue_patch import encode_prompt

    model_path = MODELS / args.model

    print("\nLoading:", model_path)
    model, tokenizer = load(str(model_path))

    prompts = {}
    positions = {}

    for question in QUESTIONS:
        prompt_set = {
            name: prompt_for(
                tokenizer, question, *selectors
            )
            for name, selectors in SWAPS.items()
        }

        print(f"\nToken audit: {question}")

        positions[question] = audit_token_alignment(
            tokenizer, prompt_set, encode_prompt
        )

        prompts[question] = prompt_set

    if args.phase == "audit":
        print("\nAlignment passed for both questions.")
        print("No model inference performed.")
        return

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = output_dir / (
        f"ml023_{args.model}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    data = {
        "experiment": "ML-023",
        "model": args.model,
        "model_path": str(model_path),
        "phase": args.phase,
        "design_sha256": digest,
        "complete": False,
        "questions": {},
    }

    def save():
        output.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    for question in QUESTIONS:
        print(f"\n=== {question.upper()} QUESTION ===")

        prompt_set = prompts[question]

        full_margins = {
            name: margins(
                model, tokenizer, prompt, log_likelihood
            )
            for name, prompt in prompt_set.items()
        }

        baseline = full_margins["base"]

        full_effects = {
            name: difference(values, baseline)
            for name, values in full_margins.items()
            if name != "base"
        }

        full_interaction = {
            role: (
                full_margins["both"][role]
                - full_margins["agent"][role]
                - full_margins["patient"][role]
                + baseline[role]
            )
            for role in ("agent", "patient")
        }

        print("Baseline:", {
            k: round(v, 3)
            for k, v in baseline.items()
        })

        for name, effects in full_effects.items():
            print(
                f"FULL {name:8s}",
                {
                    k: round(v, 3)
                    for k, v in effects.items()
                },
            )

        print("Full-input interaction:", {
            k: round(v, 3)
            for k, v in full_interaction.items()
        })

        data["questions"][question] = {
            "baseline": baseline,
            "full": full_margins,
            "full_effects": full_effects,
            "full_interaction": full_interaction,
            "positions": positions[question],
            "layers": {},
        }

        save()

        if args.phase != "patch":
            continue

        pos = positions[question]

        for layer in LAYERS:
            if layer >= len(model.model.layers):
                raise ValueError(
                    f"Layer {layer} is unavailable"
                )

            captured = {
                name: capture(
                    model,
                    tokenizer,
                    prompt_set[name],
                    layer,
                    pos,
                    encode_prompt,
                )
                for name in SWAPS
            }

            patch_configs = {
                "self": {
                    pos["agent"]:
                        captured["base"]["agent"],
                    pos["patient"]:
                        captured["base"]["patient"],
                },

                "prefix_control": {
                    pos["prefix"]:
                        captured["agent"]["prefix"],
                },

                "agent_only": {
                    pos["agent"]:
                        captured["agent"]["agent"],
                },

                "patient_only": {
                    pos["patient"]:
                        captured["patient"]["patient"],
                },

                "both_independent": {
                    pos["agent"]:
                        captured["agent"]["agent"],
                    pos["patient"]:
                        captured["patient"]["patient"],
                },

                "both_joint_donor": {
                    pos["agent"]:
                        captured["both"]["agent"],
                    pos["patient"]:
                        captured["both"]["patient"],
                },
            }

            entries = {}

            for label, patch_map in patch_configs.items():
                measured = patch_margins(
                    model,
                    tokenizer,
                    prompt_set["base"],
                    layer,
                    patch_map,
                    log_likelihood,
                )

                effect = difference(measured, baseline)

                entries[label] = {
                    "margins": measured,
                    "effects": effect,
                }

                print(
                    f"L{layer:02d} {label:17s} "
                    f"dAgent={effect['agent']:+.3f} "
                    f"dPatient={effect['patient']:+.3f}"
                )

                if label in ("self", "prefix_control"):
                    error = max(
                        abs(v) for v in effect.values()
                    )

                    if error > 0.05:
                        data["questions"][question][
                            "layers"
                        ][str(layer)] = entries

                        save()

                        raise RuntimeError(
                            f"{label} integrity check "
                            f"failed at layer {layer}"
                        )

            agent_effect = entries[
                "agent_only"
            ]["effects"]

            patient_effect = entries[
                "patient_only"
            ]["effects"]

            double_effect = entries[
                "both_independent"
            ]["effects"]

            patch_interaction = {
                role: (
                    double_effect[role]
                    - agent_effect[role]
                    - patient_effect[role]
                )
                for role in ("agent", "patient")
            }

            print(
                f"L{layer:02d} two-patch interaction:",
                {
                    k: round(v, 3)
                    for k, v in patch_interaction.items()
                },
            )

            data["questions"][question][
                "layers"
            ][str(layer)] = {
                "results": entries,
                "double_patch_interaction":
                    patch_interaction,
            }

            save()

    data["complete"] = True
    save()

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
