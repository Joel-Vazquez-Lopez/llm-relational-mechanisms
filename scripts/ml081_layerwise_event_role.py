
"""ML-081: layerwise interventions on familiar ML-080 calibration.

Phases:
    python3 scripts/ml081_layerwise_event_role.py --phase plan
    python3 scripts/ml081_layerwise_event_role.py --phase preflight
    python3 scripts/ml081_layerwise_event_role.py --phase analyze

No protected examples are generated or evaluated.
"""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch
import ml080_learning_curve as ml

N_SENTENCES = 48
RANDOM_REPS = 3
SEEDS = tuple(ml.SEEDS)
ARMS = tuple(ml.ARMS)
SITES = ("attn", "mlp", "residual")
LAYERS = tuple(range(ml.LAYERS))


def sha_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()
    ).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_ml080(path):
    doc = json.loads(path.read_text())
    assert doc["experiment"] == "ML-080"
    assert doc["code_sha256"] == sha_file(ml.__file__), (
        "ML-080 source differs from frozen version"
    )
    assert doc["sha256"] == sha_json({
        k: v for k, v in doc.items() if k != "sha256"
    })
    assert tuple(doc["seeds"]) == SEEDS
    assert tuple(doc["arms"]) == ARMS
    return doc


def swap_event(c):
    return replace(
        c,
        target_pair=c.distractor_pair,
        distractor_pair=c.target_pair,
        target_verb=c.distractor_verb,
        distractor_verb=c.target_verb,
        target_pos=1-c.target_pos,
        target_voice=c.distractor_voice,
        distractor_voice=c.target_voice,
        target_dir=c.distractor_dir,
        distractor_dir=c.target_dir,
    )


def lexical_donor(c):
    verb = next(
        v for v in ml.VERBS
        if v not in (c.target_verb, c.distractor_verb)
    )
    return replace(c, distractor_verb=verb)


def changed_tokens(a, b):
    x, y = ml.prompt(a), ml.prompt(b)
    assert len(x) == len(y)
    return [(p, q) for p, q in zip(x, y) if p != q]


def selected_cases(doc):
    # Calibration ONLY. Never call protected_target_cases().
    cal = ml.base_cases(doc["splits"]["calibration_pairs"])

    events = [
        c for c in cal
        if c.target_pos == 1
        and c.target_voice == "active"
        and c.distractor_voice == "active"
        and c.role == "agent"
    ]
    assert len(events) == 576

    events.sort(
        key=lambda c: hashlib.sha256(
            " ".join(ml.prompt(c)).encode()
        ).hexdigest()
    )
    picked = events[:N_SENTENCES]

    cases = []
    for second_agent in picked:
        first_agent = swap_event(second_agent)
        assert changed_tokens(
            first_agent, second_agent
        ) == [("first", "second")]

        a = ml.prompt(first_agent)
        b = ml.prompt(second_agent)
        assert a[:a.index("<sep>")] == b[:b.index("<sep>")]

        cases += [
            first_agent,
            replace(first_agent, role="patient"),
            second_agent,
            replace(second_agent, role="patient"),
        ]

    role = [
        replace(
            c, role="patient" if c.role == "agent" else "agent"
        )
        for c in cases
    ]
    event = [swap_event(c) for c in cases]
    lexical = [lexical_donor(c) for c in cases]

    for c, r, e, l in zip(cases, role, event, lexical):
        assert changed_tokens(c, r) in (
            [("agent", "patient")],
            [("patient", "agent")],
        )
        assert changed_tokens(c, e) in (
            [("first", "second")],
            [("second", "first")],
        )
        assert len(changed_tokens(c, l)) == 1
        assert ml.answer(c) != ml.answer(r)
        assert ml.answer(c) != ml.answer(e)
        assert ml.answer(c) == ml.answer(l)

    assert len(cases) == N_SENTENCES * 4
    return cases, role, event, lexical


def protocol_doc(doc, cases):
    return {
        "experiment": "ML-081",
        "code_sha256": sha_file(__file__),
        "ml080_protocol_sha256": doc["sha256"],
        "source": (
            "ML-080 familiar active/active CALIBRATION only; "
            "no protected cases"
        ),
        "selection": (
            "first 48 second-event agent calibration prompts "
            "after SHA256 lexical sort"
        ),
        "sample_digest": sha_json([
            ml.prompt(c) for c in cases
        ]),
        "random_controls_per_donor": RANDOM_REPS,
        "random_baseline": (
            "isotropic Gaussian normalized to recipient-donor "
            "site delta per question"
        ),
        "lexical_control": (
            "one distractor verb substitution preserving correct "
            "answer; NOT norm matched to role or event"
        ),
        "positive_control": (
            "last-layer final-token residual patch copies donor "
            "readout, so is NOT evidence for a unique circuit"
        ),
        "seeds": SEEDS,
        "arms": ARMS,
        "checkpoint": 12800,
        "layers": LAYERS,
        "sites": SITES,
        "patches": [
            "opposite role",
            "opposite event",
            "answer-preserving distractor verb",
            "norm-matched Gaussian controls",
        ],
        "primary_descriptive": (
            "mean gold-vs-best-other candidate margin "
            "reduction by patch and site"
        ),
        "auxiliary": (
            "native candidate accuracy, donor-relative logit "
            "shifts, effect norms, matched random control"
        ),
        "note": (
            "Exploratory discovery; same examples for all models; "
            "no independent or prospective mechanism validation"
        ),
    }


def check_freeze(path, doc, cases, create):
    data = protocol_doc(doc, cases)
    expected = json.loads(json.dumps({
        **data, "sha256": sha_json(data)
    }))

    if create and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(expected, indent=2) + "\n"
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Run --phase plan: {path}"
        )

    assert json.loads(path.read_text()) == expected, (
        "ML-081 source/protocol changed"
    )
    return expected["sha256"]


def check_runs(root, doc):
    for seed in SEEDS:
        for arm in ARMS:
            folder = root / f"seed_{seed}" / arm
            ckpt = folder / "model_step_12800.pt"
            met = folder / "metrics_step_12800.json"

            assert ckpt.is_file() and met.is_file(), (
                f"Missing ML-080 snapshot: {folder}"
            )

            m = json.loads(met.read_text())
            assert (
                m["seed"], m["arm"], m["protocol_sha256"]
            ) == (seed, arm, doc["sha256"])


def gather_state(x, readout):
    idx = torch.arange(len(readout), device=x.device)
    return x[idx, readout].clone()


@torch.inference_mode()
def forward_trace(model, ids, ro, patch=None):
    """Reproduce ML-080 forward, with optional component patch."""
    b, n = ids.shape
    index = torch.arange(b, device=ids.device)

    x = (
        model.embed(ids)
        + model.pos(torch.arange(n, device=ids.device))
    )

    mask = torch.ones(
        n, n, device=ids.device, dtype=torch.bool
    ).triu(1)

    traces = []

    for layer, block in enumerate(model.blocks):
        z = block.n1(x)
        attn = block.attn(
            z, z, z, attn_mask=mask, need_weights=False
        )[0]

        if patch is not None and patch[0:2] == (
            layer, "attn"
        ):
            attn[index, ro] = patch[2]

        after_attn = x + attn
        mlp_out = block.mlp(block.n2(after_attn))

        if patch is not None and patch[0:2] == (
            layer, "mlp"
        ):
            mlp_out[index, ro] = patch[2]

        x = after_attn + mlp_out

        if patch is not None and patch[0:2] == (
            layer, "residual"
        ):
            x[index, ro] = patch[2]

        traces.append({
            "attn": gather_state(attn, ro),
            "mlp": gather_state(mlp_out, ro),
            "residual": gather_state(x, ro),
        })

    logits = model.out(
        model.norm(x[index, ro])
    ).float()

    return logits, traces


def margin(logits, labels, choices):
    candidate = logits.gather(1, choices)

    gold_col = (
        choices == labels[:, None]
    ).long().argmax(1)

    gold = candidate.gather(
        1, gold_col[:, None]
    ).flatten()

    other = candidate.masked_fill(
        choices == labels[:, None],
        float("-inf"),
    ).max(1).values

    return gold - other


def donor_advantage(logits, donor_ids, receiver_ids):
    donor = logits.gather(
        1, donor_ids[:, None]
    ).flatten()

    recipient = logits.gather(
        1, receiver_ids[:, None]
    ).flatten()

    return donor - recipient


def random_like_delta(delta, rng):
    sample = torch.randn(
        delta.shape,
        generator=rng,
        device="cpu",
    ).to(delta.device)

    norm = sample.norm(
        dim=1, keepdim=True
    ).clamp_min(1e-12)

    return (
        sample / norm
        * delta.norm(dim=1, keepdim=True)
    )


def score_model(model, cases, donor_cases, device, seed, arm):
    model.eval()

    encoded = {}
    for name, batch in {
        "native": cases,
        **donor_cases,
    }.items():
        ids, ro, labels, choices = ml.encode(batch)
        encoded[name] = (
            ids.to(device),
            ro.to(device),
            labels.to(device),
            choices.to(device),
        )

    ids, ro, gold, choices = encoded["native"]

    native, base_trace = forward_trace(
        model, ids, ro
    )
    original = model(ids, ro).float()

    err = float(
        (native - original).abs().max().detach()
    )
    assert err < 1e-3, f"Forward parity failure {err}"

    native_m = margin(
        native, gold, choices
    )

    native_pred = choices.gather(
        1,
        native.gather(1, choices).argmax(
            1, keepdim=True
        ),
    ).flatten()

    donor_traces = {}
    donor_gold = {}

    for name in ("role", "event", "lexical"):
        di, dr, dl, _ = encoded[name]
        donor_logits, donor_traces[name] = forward_trace(
            model, di, dr
        )
        donor_gold[name] = dl
        assert donor_logits.shape == native.shape

    assert bool(torch.all(
        donor_gold["lexical"] == gold
    ))

    result = []

    for layer in LAYERS:
        for site in SITES:
            original_site = base_trace[layer][site]

            for typ in ("role", "event", "lexical"):
                delta = (
                    donor_traces[typ][layer][site]
                    - original_site
                )

                donor_norms = delta.norm(dim=1)

                patched_logits, _ = forward_trace(
                    model,
                    ids,
                    ro,
                    patch=(
                        layer,
                        site,
                        original_site + delta,
                    ),
                )

                drop = native_m - margin(
                    patched_logits, gold, choices
                )

                random_drops = []

                for rep in range(RANDOM_REPS):
                    seed_value = (
                        seed * 100000
                        + layer * 10000
                        + SITES.index(site) * 1000
                        + (
                            1 if typ == "role"
                            else 2 if typ == "event"
                            else 3
                        ) * 10
                        + rep
                    )

                    generator = torch.Generator(
                        device="cpu"
                    ).manual_seed(seed_value)

                    rdelta = random_like_delta(
                        delta, generator
                    )

                    random_logits, _ = forward_trace(
                        model,
                        ids,
                        ro,
                        patch=(
                            layer,
                            site,
                            original_site + rdelta,
                        ),
                    )

                    random_drops.append(
                        native_m - margin(
                            random_logits, gold, choices
                        )
                    )

                random_drop = torch.stack(
                    random_drops
                ).mean(0)

                candidate_logits = patched_logits.gather(
                    1, choices
                )

                predicted = choices.gather(
                    1,
                    candidate_logits.argmax(
                        1, keepdim=True
                    ),
                ).flatten()

                changes = {
                    "seed": seed,
                    "arm": arm,
                    "layer": layer,
                    "site": site,
                    "donor": typ,
                    "n_questions": len(cases),
                    "native_accuracy": float(
                        (native_pred == gold).float().mean()
                    ),
                    "mean_margin_drop": float(
                        drop.mean()
                    ),
                    "mean_random_margin_drop": float(
                        random_drop.mean()
                    ),
                    "mean_excess_over_random": float(
                        (drop - random_drop).mean()
                    ),
                    "median_delta_norm": float(
                        donor_norms.median()
                    ),
                    "zero_delta_rate": float(
                        (donor_norms < 1e-7).float().mean()
                    ),
                    "patched_accuracy": float(
                        (predicted == gold).float().mean()
                    ),
                }

                if typ in ("role", "event"):
                    shifted = donor_advantage(
                        patched_logits,
                        donor_gold[typ],
                        gold,
                    ) - donor_advantage(
                        native,
                        donor_gold[typ],
                        gold,
                    )

                    changes[
                        "mean_donor_advantage_shift"
                    ] = float(shifted.mean())

                result.append(changes)

    return err, result


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--phase",
        choices=("plan", "preflight", "analyze"),
        required=True,
    )
    p.add_argument(
        "--seed", type=int, choices=SEEDS
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    p.add_argument(
        "--ml080-protocol",
        default="reports/ml080/frozen_protocol.json",
    )
    p.add_argument(
        "--protocol",
        default="reports/ml081/frozen_protocol.json",
    )
    p.add_argument(
        "--root",
        default="results/ml080_learning_curve",
    )
    p.add_argument(
        "--output",
        default="reports/ml081/discovery.json",
    )

    args = p.parse_args()
    torch.set_num_threads(min(
        4, torch.get_num_threads()
    ))

    frozen = verify_ml080(
        Path(args.ml080_protocol)
    )

    cases, role, event, lexical = selected_cases(frozen)

    h = check_freeze(
        Path(args.protocol),
        frozen,
        cases,
        args.phase == "plan",
    )

    if args.phase == "plan":
        print(
            "ML-081 frozen",
            args.protocol,
            h[:12],
        )
        return

    check_runs(Path(args.root), frozen)

    print(
        f"ML-081 preflight: {N_SENTENCES} sentences, "
        f"{len(cases)} questions, "
        f"{len(SEEDS)*len(ARMS)} checkpoints, "
        "3 layers × 3 sites"
    )
    print(
        "Only familiar active/active calibration. "
        "ML-080 protected examples NOT generated."
    )

    if args.phase == "preflight":
        torch.manual_seed(81081)
        model = ml.Model().eval()

        small = cases[:4]
        ids, ro, _, _ = ml.encode(small)

        native, traces = forward_trace(
            model, ids, ro
        )
        reference = model(ids, ro).float()

        error = float(
            (native - reference).abs().max().detach()
        )
        assert error < 1e-3

        same, _ = forward_trace(
            model,
            ids,
            ro,
            patch=(0, "attn", traces[0]["attn"]),
        )
        assert float(
            (native - same).abs().max().detach()
        ) < 1e-3

        donor = [
            replace(
                c,
                role="patient"
                if c.role == "agent"
                else "agent",
            )
            for c in small
        ]

        di, dr, _, _ = ml.encode(donor)

        donor_logits, donor_traces = forward_trace(
            model, di, dr
        )

        transplanted, _ = forward_trace(
            model,
            ids,
            ro,
            patch=(
                ml.LAYERS - 1,
                "residual",
                donor_traces[
                    ml.LAYERS - 1
                ]["residual"],
            ),
        )

        upper_error = float(
            (donor_logits - transplanted)
            .abs().max().detach()
        )
        assert upper_error < 1e-3

        print(
            f"Untrained parity PASS: native={error:.2e}, "
            f"final-residual-positive-control={upper_error:.2e}"
        )
        return

    output = Path(args.output)

    if (
        args.seed is not None
        and args.output == "reports/ml081/discovery.json"
    ):
        output = Path(
            f"reports/ml081/pilot_seed_{args.seed}.json"
        )

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {output}"
        )

    device = ml.select_device(args.device)

    report = {
        "experiment": "ML-081",
        "protocol_sha256": h,
        "ml080_protocol_sha256": frozen["sha256"],
        "protected_evaluation_opened": False,
        "exploratory": True,
        "n_sentences": N_SENTENCES,
        "device": device,
        "rows": [],
    }

    for seed in (
        SEEDS if args.seed is None
        else (args.seed,)
    ):
        for arm in ARMS:
            checkpoint = (
                Path(args.root)
                / f"seed_{seed}"
                / arm
                / "model_step_12800.pt"
            )

            model = ml.load_checkpoint(
                checkpoint,
                seed,
                arm,
                frozen["sha256"],
                device,
            )

            err, rows = score_model(
                model,
                cases,
                {
                    "role": role,
                    "event": event,
                    "lexical": lexical,
                },
                device,
                seed,
                arm,
            )

            report["rows"].extend(rows)

            fixed = [
                r for r in rows
                if r["layer"] == 2
                and r["site"] == "attn"
            ]
            scores = {
                r["donor"]: r for r in fixed
            }

            print(
                f"seed={seed} arm={arm:8s} "
                f"acc={scores['role']['native_accuracy']:.3f} "
                f"L3-attn role_excess="
                f"{scores['role']['mean_excess_over_random']:+.3f} "
                f"event_excess="
                f"{scores['event']['mean_excess_over_random']:+.3f} "
                f"lex_excess="
                f"{scores['lexical']['mean_excess_over_random']:+.3f} "
                f"parity={err:.2e}",
                flush=True,
            )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("Saved:", output)
    print(
        "Exploratory calibration only; protected "
        "examples NOT generated or evaluated."
    )


if __name__ == "__main__":
    main()
