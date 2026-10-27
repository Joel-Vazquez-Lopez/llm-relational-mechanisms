
"""ML-028: Causal interchange of relational event meaning."""

import argparse
import hashlib
import json

from datetime import datetime
from itertools import product
from pathlib import Path
from statistics import mean


PAIRS = (
    ("Nora", "Clara"),
    ("Felix", "Nora"),
)

VERBS = ("helped", "praised")
VOICES = ("active", "passive")
QUESTIONS = ("agent", "patient")

# Frozen intervention locations and layers.
LAYERS = (7, 13, 20)
SITES = ("event_end", "readout")

DONORS = (
    "self",
    "pre_event",
    "equivalent",
    "reverse_same_voice",
    "reverse_other_voice",
)


def event(pair, verb, direction, voice):
    actor, receiver = (
        pair if direction == 0 else pair[::-1]
    )

    if voice == "active":
        return (
            f"At dusk, {actor} {verb} {receiver}."
        )

    return (
        f"At dusk, {receiver} was {verb} by {actor}."
    )


def gold(pair, direction, question):
    actor, receiver = (
        pair if direction == 0 else pair[::-1]
    )

    return (
        actor if question == "agent"
        else receiver
    )


def make_prompt(
    tokenizer, pair, verb,
    direction, voice, question
):
    question_text = (
        "Who performed the action?"
        if question == "agent"
        else "Who received the action?"
    )

    content = (
        "Answer with exactly one person's name.\n"
        f"Event: {event(pair, verb, direction, voice)}\n"
        f"Question: {question_text}\n"
        "Answer:"
    )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def design():
    spec = {
        "experiment": "ML-028",
        "pairs": PAIRS,
        "verbs": VERBS,
        "voices": VOICES,
        "questions": QUESTIONS,
        "directions": (0, 1),
        "layers": LAYERS,
        "sites": SITES,
        "donors": DONORS,
        "competence_gate_overall": 0.875,
        "competence_gate_each_role_voice": 0.75,
        "primary_contrast": (
            "reverse_other_voice minus equivalent "
            "on recipient gold margin"
        ),
        "success_direction": "negative",
        "state_kind": (
            "Transformer block output at event-final "
            "or answer-readout token"
        ),
    }

    digest = hashlib.sha256(
        json.dumps(
            spec, sort_keys=True
        ).encode()
    ).hexdigest()

    return spec, digest


def cases():
    return list(product(
        range(len(PAIRS)),
        VERBS,
        (0, 1),
        VOICES,
        QUESTIONS,
    ))



def positions(tokenizer, prompt, encode_prompt):
    """Locate safe causal positions in the FULL tokenization.

    Avoid re-encoding truncated prompt strings, because
    their tokenization may differ at the truncation boundary.
    """

    event_marker = "\nEvent: "
    question_marker = "\nQuestion: "

    if (
        prompt.count(event_marker) != 1
        or prompt.count(question_marker) != 1
    ):
        raise ValueError(
            "Cannot uniquely identify event/question markers"
        )

    event_start = prompt.index(event_marker) + 1
    question_start = prompt.index(question_marker) + 1

    if not event_start < question_start:
        raise ValueError("Invalid event/question ordering")

    event_period = prompt.rfind(
        ".", event_start, question_start
    )

    if event_period < 0:
        raise ValueError("Missing event-final period")

    full_ids = encode_prompt(tokenizer, prompt)

    def decode(ids):
        try:
            return tokenizer.decode(
                ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return tokenizer.decode(
                ids,
                skip_special_tokens=False,
            )

    full_text = decode(full_ids)

    # Some encoders may prepend a special token.
    # Account for that explicitly rather than guessing.
    if full_text == prompt:
        offset = 0
    elif full_text.endswith(prompt):
        offset = len(full_text) - len(prompt)
    else:
        raise ValueError(
            "Full prompt does not round-trip through "
            "the tokenizer.\n"
            f"Original prefix: {prompt[:100]!r}\n"
            f"Decoded prefix: {full_text[:100]!r}"
        )

    boundaries = []

    for count in range(1, len(full_ids) + 1):
        prefix = decode(full_ids[:count])

        if not full_text.startswith(prefix):
            raise ValueError(
                "Token-prefix decoding is not stable "
                f"at token count {count}"
            )

        boundaries.append(
            (count - 1, len(prefix))
        )

    # The pre-event state must not see "Event:".
    pre_event_candidates = [
        token_index
        for token_index, char_end in boundaries
        if char_end <= offset + event_start
    ]

    # The event-end state must include the entire event
    # (including its final period) but must not include
    # any characters of the "Question:" label.
    event_candidates = [
        token_index
        for token_index, char_end in boundaries
        if (
            char_end >= offset + event_period + 1
            and char_end <= offset + question_start
        )
    ]

    if not pre_event_candidates:
        raise ValueError(
            "No safe pre-event token boundary"
        )

    if not event_candidates:
        raise ValueError(
            "No token boundary after the complete "
            "event and before the question label. "
            "Do not patch a partial event or "
            "question-contaminated state."
        )

    control_pos = max(pre_event_candidates)
    event_pos = max(event_candidates)
    readout_pos = len(full_ids) - 1

    if not (
        0 <= control_pos < event_pos < readout_pos
    ):
        raise ValueError(
            "Invalid causal ordering: "
            f"pre={control_pos}, "
            f"event={event_pos}, "
            f"readout={readout_pos}"
        )

    # Verify that the selected event state genuinely
    # contains the complete event.
    event_prefix = decode(full_ids[:event_pos + 1])

    if event_prefix[offset + event_period] != ".":
        raise ValueError(
            "Event punctuation was not preserved"
        )

    return {
        "event_end": event_pos,
        "readout": readout_pos,
        "pre_event": control_pos,
        "length": len(full_ids),
    }


def answer_scores(
    model, tokenizer, prompt,
    log_likelihood, pair
):
    return {
        name: float(
            log_likelihood(
                model, tokenizer, prompt, name
            )[0]
        )
        for name in pair
    }


def gold_margin(scores, correct, pair):
    other = (
        pair[0]
        if correct == pair[1]
        else pair[1]
    )

    return scores[correct] - scores[other]


def capture(
    model, tokenizer, prompt,
    layer_index, position, encode_prompt
):
    import mlx.core as mx
    import mlx.nn as nn

    class Capture(nn.Module):
        def __init__(self, layer):
            super().__init__()

            self.layer = layer
            self.use_sliding = getattr(
                layer, "use_sliding", False
            )
            self.value = None

        def __call__(self, *args, **kwargs):
            output = self.layer(*args, **kwargs)
            self.value = output[0, position, :]
            return output

    layers = model.model.layers
    original = layers[layer_index]
    wrapper = Capture(original)

    layers[layer_index] = wrapper

    try:
        ids = encode_prompt(tokenizer, prompt)

        tokens = mx.array(
            [ids], dtype=mx.int32
        )

        mx.eval(model(tokens))

        if wrapper.value is None:
            raise RuntimeError(
                "No activation captured"
            )

        mx.eval(wrapper.value)
        result = wrapper.value

    finally:
        layers[layer_index] = original

    return result


def patched_scores(
    model, tokenizer, prompt, pair,
    layer_index, position, replacement,
    log_likelihood
):
    import mlx.core as mx
    import mlx.nn as nn

    class Patch(nn.Module):
        def __init__(self, layer):
            super().__init__()

            self.layer = layer
            self.use_sliding = getattr(
                layer, "use_sliding", False
            )

        def __call__(self, *args, **kwargs):
            output = self.layer(*args, **kwargs)

            return mx.concatenate(
                (
                    output[:, :position, :],
                    replacement[
                        None, None, :
                    ].astype(output.dtype),
                    output[:, position + 1:, :],
                ),
                axis=1,
            )

    layers = model.model.layers
    original = layers[layer_index]

    layers[layer_index] = Patch(original)

    try:
        result = answer_scores(
            model,
            tokenizer,
            prompt,
            log_likelihood,
            pair,
        )
    finally:
        layers[layer_index] = original

    return result


def summarize(rows):
    print(
        "\n=== MATCHED-VOICE "
        "SEMANTIC CONTRAST ==="
    )

    print(
        "Contrast = reverse_other_voice "
        "- equivalent"
    )

    print(
        "Negative: intervention follows "
        "reversed donor semantics"
    )

    groups = {}

    for row in rows:
        key = (
            row["site"],
            row["layer"],
            row["pair_id"],
            row["verb"],
            row["direction"],
            row["voice"],
            row["question"],
        )

        groups.setdefault(key, {})[
            row["donor"]
        ] = row["effect"]

    for site, layer in product(SITES, LAYERS):
        differences = []

        for key, results in groups.items():
            if key[:2] != (site, layer):
                continue

            if (
                "equivalent" not in results
                or "reverse_other_voice" not in results
            ):
                continue

            difference = (
                results["reverse_other_voice"]
                - results["equivalent"]
            )

            differences.append((
                key[-1], difference
            ))

        if not differences:
            continue

        values = [
            value for _, value in differences
        ]

        print(
            f"\n{site} L{layer} "
            f"n={len(values)} "
            f"mean={mean(values):+.3f} "
            f"negative="
            f"{sum(x < 0 for x in values)}"
            f"/{len(values)}"
        )

        for question in QUESTIONS:
            subset = [
                value
                for role, value in differences
                if role == question
            ]

            print(
                f"  {question}: "
                f"mean={mean(subset):+.3f} "
                f"negative="
                f"{sum(x < 0 for x in subset)}"
                f"/{len(subset)}"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=(
            "design",
            "audit",
            "behavior",
            "patch",
            "summarize",
        ),
        default="design",
    )

    parser.add_argument(
        "--model",
        choices=("llama3b", "qwen7b"),
        default="llama3b",
    )

    parser.add_argument(
        "--layer",
        type=int,
        choices=LAYERS,
        default=13,
    )

    parser.add_argument(
        "--site",
        choices=SITES,
        default="event_end",
    )

    args = parser.parse_args()

    spec, digest = design()

    print(
        f"ML-028 | {args.phase} "
        f"| {args.model} "
        f"| hash={digest}"
    )

    if args.phase == "design":
        print(json.dumps(spec, indent=2))

        for pair in PAIRS:
            for voice, direction in product(
                VOICES, (0, 1)
            ):
                print(
                    event(
                        pair, "helped",
                        direction, voice
                    )
                )

        print(
            f"Cases: {len(cases())}"
        )
        print(
            "Patches per layer/site:",
            len(cases()) * len(DONORS),
        )
        return

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    if args.phase == "summarize":
        rows = []

        for file in sorted(
            results_dir.glob(
                f"ml028_{args.model}_patch_*.json"
            )
        ):
            data = json.loads(
                file.read_text()
            )

            if (
                data.get("design_hash") == digest
                and data.get("complete")
            ):
                rows.extend(
                    data["records"]
                )

                print("Reading:", file)

        if not rows:
            raise FileNotFoundError(
                "No complete patch results"
            )

        summarize(rows)
        return

    from mlx_lm import load

    from ml003_role_margins import (
        MODELS,
        log_likelihood,
    )

    from ml012_cue_patch import (
        encode_prompt,
    )

    model, tokenizer = load(
        str(MODELS / args.model)
    )

    for pair in PAIRS:
        lengths = [
            len(
                tokenizer.encode(
                    name,
                    add_special_tokens=False,
                )
            )
            for name in pair
        ]

        print(
            "Name pair:", pair,
            "token lengths:", lengths
        )

        if len(set(lengths)) != 1:
            raise ValueError(
                "Unequal candidate token lengths. "
                "Do not proceed."
            )

    prepared = {}

    for key in cases():
        pair_id, verb, direction, voice, question = key
        pair = PAIRS[pair_id]

        prompt = make_prompt(
            tokenizer,
            pair,
            verb,
            direction,
            voice,
            question,
        )

        site_positions = positions(
            tokenizer,
            prompt,
            encode_prompt,
        )

        prompt_ids = encode_prompt(
            tokenizer, prompt
        )

        contextual_lengths = []

        for name in pair:
            extended = encode_prompt(
                tokenizer, prompt + name
            )

            if (
                extended[:len(prompt_ids)]
                != prompt_ids
            ):
                raise ValueError(
                    "Answer-token boundary "
                    f"changes for {name!r}"
                )

            contextual_lengths.append(
                len(extended) - len(prompt_ids)
            )

        if len(set(contextual_lengths)) != 1:
            raise ValueError(
                "Unequal contextual answer lengths: "
                f"{contextual_lengths}"
            )

        prepared[key] = (
            prompt, site_positions
        )

    print(
        "Token audits passed:",
        len(prepared),
        "/",
        len(cases()),
    )

    if args.phase == "audit":
        print(
            "No likelihood scoring performed"
        )
        return

    def last_behavior():
        paths = sorted(
            results_dir.glob(
                f"ml028_{args.model}_behavior_*.json"
            )
        )

        for filename in reversed(paths):
            data = json.loads(
                filename.read_text()
            )

            if (
                data.get("design_hash") == digest
                and data.get("complete")
            ):
                return data

        raise FileNotFoundError(
            "Run behavior phase first"
        )

    if args.phase == "behavior":
        output = results_dir / (
            f"ml028_{args.model}_behavior_"
            f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
        )

        data = {
            "experiment": "ML-028",
            "design_hash": digest,
            "complete": False,
            "records": [],
        }

        for key, (prompt, pos) in prepared.items():
            pair_id, verb, direction, voice, question = key
            pair = PAIRS[pair_id]

            scores = answer_scores(
                model,
                tokenizer,
                prompt,
                log_likelihood,
                pair,
            )

            correct = gold(
                pair, direction, question
            )

            margin = gold_margin(
                scores, correct, pair
            )

            row = {
                "pair_id": pair_id,
                "verb": verb,
                "direction": direction,
                "voice": voice,
                "question": question,
                "gold": correct,
                "scores": scores,
                "margin": margin,
                "correct": margin > 0,
            }

            data["records"].append(row)

            print(
                f"{pair_id} {verb} {direction} "
                f"{voice} {question}: "
                f"margin={margin:+.3f} "
                f"{'OK' if margin > 0 else 'FAIL'}"
            )

            output.write_text(
                json.dumps(data, indent=2)
            )

        data["complete"] = True

        output.write_text(
            json.dumps(data, indent=2)
        )

        print("\nSaved:", output)

        print(
            "Overall:",
            sum(r["correct"] for r in data["records"]),
            "/",
            len(data["records"]),
        )

        for feature in ("voice", "question"):
            possible = (
                VOICES if feature == "voice"
                else QUESTIONS
            )

            for value in possible:
                subset = [
                    r for r in data["records"]
                    if r[feature] == value
                ]

                print(
                    feature,
                    value,
                    sum(r["correct"] for r in subset),
                    "/",
                    len(subset),
                )

        return

    behavior = last_behavior()
    records = behavior["records"]

    overall = (
        sum(r["correct"] for r in records)
        / len(records)
    )

    strata = []

    for voice, question in product(
        VOICES, QUESTIONS
    ):
        subset = [
            r for r in records
            if (
                r["voice"] == voice
                and r["question"] == question
            )
        ]

        strata.append(
            sum(r["correct"] for r in subset)
            / len(subset)
        )

    if overall < 0.875 or min(strata) < 0.75:
        raise RuntimeError(
            "Behavioral competence gate failed: "
            f"overall={overall:.3f}, "
            f"role/voice={strata}. "
            "No causal patches performed."
        )

    output = results_dir / (
        f"ml028_{args.model}_patch_"
        f"{args.site}_L{args.layer}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    data = {
        "experiment": "ML-028",
        "design_hash": digest,
        "site": args.site,
        "layer": args.layer,
        "complete": False,
        "records": [],
    }

    for pair_id, verb, question in product(
        range(len(PAIRS)),
        VERBS,
        QUESTIONS,
    ):
        pair = PAIRS[pair_id]

        variants = {}

        for direction, voice in product(
            (0, 1), VOICES
        ):
            key = (
                pair_id,
                verb,
                direction,
                voice,
                question,
            )

            prompt, pos = prepared[key]

            variant = {
                "prompt": prompt,
                "pos": pos,
            }

            for site in (
                args.site, "pre_event"
            ):
                variant[site] = capture(
                    model,
                    tokenizer,
                    prompt,
                    args.layer,
                    pos[site],
                    encode_prompt,
                )

            variants[
                direction, voice
            ] = variant

        for direction, voice in product(
            (0, 1), VOICES
        ):
            recipient = variants[
                direction, voice
            ]

            expected = gold(
                pair, direction, question
            )

            base_scores = answer_scores(
                model,
                tokenizer,
                recipient["prompt"],
                log_likelihood,
                pair,
            )

            baseline = gold_margin(
                base_scores,
                expected,
                pair,
            )

            other_voice = (
                "passive"
                if voice == "active"
                else "active"
            )

            origins = {
                "self": (
                    direction, voice
                ),
                "pre_event": (
                    1 - direction, voice
                ),
                "equivalent": (
                    direction, other_voice
                ),
                "reverse_same_voice": (
                    1 - direction, voice
                ),
                "reverse_other_voice": (
                    1 - direction, other_voice
                ),
            }

            for donor_type, donor_key in origins.items():
                donor = variants[donor_key]

                patch_site = (
                    "pre_event"
                    if donor_type == "pre_event"
                    else args.site
                )

                patched = patched_scores(
                    model,
                    tokenizer,
                    recipient["prompt"],
                    pair,
                    args.layer,
                    recipient["pos"][patch_site],
                    donor[patch_site],
                    log_likelihood,
                )

                effect = (
                    gold_margin(
                        patched,
                        expected,
                        pair,
                    )
                    - baseline
                )

                row = {
                    "pair_id": pair_id,
                    "verb": verb,
                    "direction": direction,
                    "voice": voice,
                    "question": question,
                    "donor": donor_type,
                    "donor_direction": donor_key[0],
                    "donor_voice": donor_key[1],
                    "site": args.site,
                    "layer": args.layer,
                    "baseline_margin": baseline,
                    "effect": effect,
                    "patched_margin": baseline + effect,
                }

                data["records"].append(row)

                print(
                    f"{pair_id} {verb} {direction} "
                    f"{voice} {question} "
                    f"{donor_type:19s} "
                    f"effect={effect:+.3f}"
                )

                output.write_text(
                    json.dumps(data, indent=2)
                )

                if (
                    donor_type in ("self", "pre_event")
                    and abs(effect) > 0.05
                ):
                    raise RuntimeError(
                        f"Control failure: {donor_type}, "
                        f"effect={effect:+.4f}. "
                        "Partial output saved."
                    )

    data["complete"] = True

    output.write_text(
        json.dumps(data, indent=2)
    )

    print("\nSaved:", output)
    summarize(data["records"])


if __name__ == "__main__":
    main()
