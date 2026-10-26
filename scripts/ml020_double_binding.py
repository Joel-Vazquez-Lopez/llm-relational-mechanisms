
import argparse
import hashlib
import json
import re
import statistics
from datetime import datetime
from itertools import product
from pathlib import Path


SELECTORS = list(product(
    ("wrote", "carried"),
    ("folded", "painted"),
))
PROBES = ("agent", "patient")


def configurations(mode):
    if mode == "dev":
        return [{
            "writer": "Leo",
            "carrier": "Felix",
            "folder": "Nora",
            "painter": "Clara",
            "verb": "norped",
            "voice": "active",
            "order": "bridge_first",
        }]

    configs = []
    for a, b, voice, order in product(
        (0, 1), (0, 1),
        ("active", "passive"),
        ("bridge_first", "facts_first"),
    ):
        letter = ("Iris", "Omar")
        map_people = ("Lena", "Hugo")

        if a:
            letter = letter[::-1]
        if b:
            map_people = map_people[::-1]

        configs.append({
            "writer": letter[0],
            "carrier": letter[1],
            "folder": map_people[0],
            "painter": map_people[1],
            "verb": "dovked",
            "voice": voice,
            "order": order,
        })

    return configs


def make_passage(c, agent_selector, patient_selector):
    agent = f"the person who {agent_selector} the letter"
    patient = f"the person who {patient_selector} the map"

    if c["voice"] == "active":
        event = f"At dusk, {agent} {c['verb']} {patient}."
    else:
        event = (
            f"At dusk, {patient} was {c['verb']} by {agent}."
        )

    facts = " ".join((
        f"{c['writer']} wrote the letter.",
        f"{c['carrier']} carried the letter.",
        f"{c['folder']} folded the map.",
        f"{c['painter']} painted the map.",
    ))

    passage = (
        f"{event} {facts}"
        if c["order"] == "bridge_first"
        else f"{facts} {event}"
    )

    names = (
        c["writer"], c["carrier"],
        c["folder"], c["painter"],
    )
    assert len(set(names)) == 4
    assert all(passage.count(n) == 1 for n in names)

    return passage


def question_for(c, probe):
    if probe == "agent":
        return f"Who {c['verb']} the other person?"
    return f"Who was {c['verb']}?"


def gold_answer(c, agent_selector, patient_selector, probe):
    if probe == "agent":
        return (
            c["writer"] if agent_selector == "wrote"
            else c["carrier"]
        )
    return (
        c["folder"] if patient_selector == "folded"
        else c["painter"]
    )


def lexical_syntax_baseline(passage, question):
    """
    Simple deterministic baseline using only the passage
    and question. Does not receive gold or configuration.
    """
    event_match = re.search(
        r"At dusk, ([^.]+)\.", passage
    )
    if event_match is None:
        raise ValueError("Cannot locate event sentence")

    event = event_match.group(1)

    active = re.fullmatch(
        r"the person who (wrote|carried) the letter "
        r"[A-Za-z]+ the person who (folded|painted) the map",
        event,
    )
    passive = re.fullmatch(
        r"the person who (folded|painted) the map "
        r"was [A-Za-z]+ by the person who "
        r"(wrote|carried) the letter",
        event,
    )

    if active:
        agent_selector, patient_selector = active.groups()
    elif passive:
        patient_selector, agent_selector = passive.groups()
    else:
        raise ValueError("Unrecognized event syntax")

    is_patient = question.startswith("Who was ")
    selector = (
        patient_selector if is_patient else agent_selector
    )
    object_name = "map" if is_patient else "letter"

    matches = re.findall(
        rf"\b([A-Z][a-z]+) {selector} "
        rf"the {object_name}\.",
        passage,
    )

    if len(matches) != 1:
        raise ValueError(
            f"Ambiguous factual match: {matches}"
        )

    return matches[0]


def make_trials(mode):
    trials = []

    for context_id, c in enumerate(configurations(mode)):
        for (agent_sel, patient_sel), probe in product(
            SELECTORS, PROBES
        ):
            passage = make_passage(c, agent_sel, patient_sel)
            question = question_for(c, probe)
            gold = gold_answer(
                c, agent_sel, patient_sel, probe
            )
            baseline = lexical_syntax_baseline(
                passage, question
            )

            assert baseline == gold, (
                "Symbolic baseline disagrees with gold"
            )

            trials.append({
                "context_id": context_id,
                "config": c,
                "agent_selector": agent_sel,
                "patient_selector": patient_sel,
                "probe": probe,
                "passage": passage,
                "question": question,
                "gold": gold,
                "symbolic_baseline": baseline,
            })

    expected = 8 if mode == "dev" else 128
    assert len(trials) == expected

    contexts = len(configurations(mode))
    for context_id in range(contexts):
        group = [
            r for r in trials
            if r["context_id"] == context_id
        ]
        assert len(group) == 8
        assert len(set(
            (
                r["agent_selector"],
                r["patient_selector"],
                r["probe"],
            )
            for r in group
        )) == 8

    return trials


def make_prompt(tokenizer, passage, question):
    content = (
        "Answer using exactly one person's name.\n\n"
        f"Passage: {passage}\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_native(raw, names):
    cleaned = raw.strip().strip(" \"'.,!?;:\n\t")
    return next(
        (
            name for name in names
            if cleaned.casefold() == name.casefold()
        ),
        None,
    )


def summarize(records):
    print("\n=== ACCURACY ===")

    for probe in PROBES:
        group = [r for r in records if r["probe"] == probe]
        print(
            f"{probe:8s} "
            f"forced={sum(r['forced'] == r['gold'] for r in group)}"
            f"/{len(group)} "
            f"native={sum(r['native'] == r['gold'] for r in group)}"
            f"/{len(group)} "
            f"symbolic={sum(r['symbolic_baseline'] == r['gold'] for r in group)}"
            f"/{len(group)}"
        )

    contrasts = []

    print("\n=== DOUBLE DISSOCIATION ===")

    for context_id in sorted(set(
        r["context_id"] for r in records
    )):
        group = {
            (
                r["agent_selector"],
                r["patient_selector"],
                r["probe"],
            ): r
            for r in records
            if r["context_id"] == context_id
        }

        def a_margin(a, p):
            return group[(a, p, "agent")]["agent_margin"]

        def p_margin(a, p):
            return group[(a, p, "patient")]["patient_margin"]

        agent_direct = [
            a_margin("wrote", p) - a_margin("carried", p)
            for p in ("folded", "painted")
        ]
        agent_cross = [
            a_margin(a, "folded") - a_margin(a, "painted")
            for a in ("wrote", "carried")
        ]
        patient_direct = [
            p_margin(a, "folded") - p_margin(a, "painted")
            for a in ("wrote", "carried")
        ]
        patient_cross = [
            p_margin("wrote", p) - p_margin("carried", p)
            for p in ("folded", "painted")
        ]

        row = {
            "context_id": context_id,
            "agent_direct": statistics.mean(agent_direct),
            "agent_cross_abs": statistics.mean(
                abs(x) for x in agent_cross
            ),
            "patient_direct": statistics.mean(patient_direct),
            "patient_cross_abs": statistics.mean(
                abs(x) for x in patient_cross
            ),
            "agent_direct_pairwise": agent_direct,
            "agent_cross_pairwise": agent_cross,
            "patient_direct_pairwise": patient_direct,
            "patient_cross_pairwise": patient_cross,
        }

        contrasts.append(row)

        print(
            f"context={context_id:02d} "
            f"agent_direct={row['agent_direct']:+.3f} "
            f"agent_cross_abs={row['agent_cross_abs']:.3f} "
            f"patient_direct={row['patient_direct']:+.3f} "
            f"patient_cross_abs={row['patient_cross_abs']:.3f}"
        )

    print("\n=== AGGREGATE ===")
    for key in (
        "agent_direct", "agent_cross_abs",
        "patient_direct", "patient_cross_abs"
    ):
        values = [row[key] for row in contrasts]
        print(
            f"{key:18s} "
            f"mean={statistics.mean(values):+.4f}"
        )

    return contrasts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("dev", "test"), default="dev"
    )
    parser.add_argument(
        "--audit-only", action="store_true"
    )
    parser.add_argument(
        "--model",
        choices=("llama3b", "qwen7b"),
        default="llama3b",
    )
    
    args = parser.parse_args()
    trials = make_trials(args.mode)

    design_json = json.dumps(
        trials, sort_keys=True, ensure_ascii=False
    )
    design_hash = hashlib.sha256(
        design_json.encode("utf-8")
    ).hexdigest()

    print("Mode:", args.mode)
    print("Contexts:", len(configurations(args.mode)))
    print("Trials:", len(trials))
    print("Symbolic baseline:", f"{len(trials)}/{len(trials)}")
    print("Design SHA-256:", design_hash)

    if args.audit_only:
        print("Design audit passed. No inference performed.")
        return

    print("\nFrozen expectations:")
    print("Agent direct effects should be positive.")
    print("Patient direct effects should be positive.")
    print("Cross-dependency absolute effects should be smaller.")
    print("These results alone cannot identify a neural mechanism.")

    from mlx_lm import load, generate
    from ml003_role_margins import MODELS, log_likelihood

    print("Loading:", args.model)
    model, tokenizer = load(str(MODELS / args.model))

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = output_dir / (
        f"ml020_{args.mode}_{args.model}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    records = []

    for trial in trials:
        c = trial["config"]
        names = (
            c["writer"], c["carrier"],
            c["folder"], c["painter"]
        )

        prompt = make_prompt(
            tokenizer, trial["passage"], trial["question"]
        )

        scores = {}
        token_counts = {}

        for name in names:
            lp, count = log_likelihood(
                model, tokenizer, prompt, name
            )
            scores[name] = float(lp)
            token_counts[name] = int(count)

        forced = max(scores, key=scores.get)
        raw = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=12,
            verbose=False,
        )
        native = parse_native(raw, names)

        record = {
            **trial,
            "prompt": prompt,
            "scores": scores,
            "token_counts": token_counts,
            "forced": forced,
            "native": native,
            "native_raw": raw,
            "agent_margin": (
                scores[c["writer"]] - scores[c["carrier"]]
            ),
            "patient_margin": (
                scores[c["folder"]] - scores[c["painter"]]
            ),
        }
        records.append(record)

        print(
            f"ctx={trial['context_id']:02d} "
            f"voice={c['voice']:7s} "
            f"agent={trial['agent_selector']:7s} "
            f"patient={trial['patient_selector']:7s} "
            f"probe={trial['probe']:7s} "
            f"gold={trial['gold']:6s} "
            f"forced={forced:6s} "
            f"native={str(native):6s}"
        )

        output.write_text(
            json.dumps({
                "experiment": "ML-020",
                "model": args.model,
                "design_sha256": design_hash,
                "complete": False,
                "records": records,
            }, indent=2),
            encoding="utf-8",
        )

    contrasts = summarize(records)

    output.write_text(
        json.dumps({
            "experiment": "ML-020",
            "model": args.model,
            "design_sha256": design_hash,
            "complete": True,
            "records": records,
            "contrasts": contrasts,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
