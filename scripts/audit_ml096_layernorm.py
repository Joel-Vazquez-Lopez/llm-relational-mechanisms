
"""Audit ML-096's 0.5x residual intervention, not a new discovery test.

Run from repository root:
  python scripts/audit_ml096_layernorm.py --phase preflight
  python scripts/audit_ml096_layernorm.py --phase run --device auto
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

import ml080_learning_curve as ml
import ml088_probe_access as m88
import ml091_counterfactual_binding as m91
import ml096_question_conditioned_ablation as m96

OUT = Path("reports/ml096/layernorm_validity_audit.json")
SEEDS = tuple(m88.SEEDS)
ARMS = ("familiar", "target")
SOURCE_PAIRS = {
    "first": ("first_agent", "first_patient"),
    "second": ("second_agent", "second_patient"),
}
MODES = ("residual_half", "residual_zero", "normalized_half")
N_SOURCE_SENTENCES = 12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_data():
    protocol_hash, groups, _, _ = m96.prepare()
    selected = groups["passive"][:N_SOURCE_SENTENCES]
    assert len(selected) == N_SOURCE_SENTENCES
    assert all(len(g) == 4 for g in selected)
    return protocol_hash, selected


def pair_positions(groups, device, key):
    locations = m96.m95.positions(groups, device)
    a, b = SOURCE_PAIRS[key]
    result = torch.stack((locations[a], locations[b]), dim=1)
    assert result.shape == (4 * len(groups), 2)
    return result


@torch.inference_mode()
def evaluate(model, groups, device, source, mode=None):
    questions = [q for group in groups for q in group]
    ids, readout, _, _ = ml.encode(questions)
    ids = ids.to(device)
    readout = readout.to(device)

    b, length = ids.shape
    idx = torch.arange(b, device=device)
    keys = pair_positions(groups, device, source)
    mask = torch.ones(
        length, length, dtype=torch.bool, device=device
    ).triu(1)

    x0 = model.embed(ids) + model.pos(
        torch.arange(length, device=device)
    )
    x1 = model.blocks[0](x0, mask)
    layer = model.blocks[1]

    native_z = layer.n1(x1)
    native_attention = layer.attn(
        native_z, native_z, native_z,
        attn_mask=mask, need_weights=False
    )[0]

    modified_x = x1.clone()

    if mode == "residual_half":
        modified_x[idx[:, None], keys] *= 0.5
        modified_z = layer.n1(modified_x)
    elif mode == "residual_zero":
        modified_x[idx[:, None], keys] = 0.0
        modified_z = layer.n1(modified_x)
    elif mode == "normalized_half":
        # Change the attention input after LayerNorm.
        # Leave the residual path x1 unchanged.
        modified_z = native_z.clone()
        modified_z[idx[:, None], keys] *= 0.5
    elif mode is None:
        modified_z = native_z
    else:
        raise ValueError(mode)

    attention = layer.attn(
        modified_z, modified_z, modified_z,
        attn_mask=mask, need_weights=False
    )[0]

    x2 = modified_x + attention
    x2 = x2 + layer.mlp(layer.n2(x2))
    x3 = model.blocks[2](x2, mask)

    output = model.out(
        model.norm(x3[idx, readout])
    ).float().cpu()

    z_before = native_z[idx[:, None], keys]
    z_after = modified_z[idx[:, None], keys]

    weight = layer.attn.in_proj_weight
    bias = layer.attn.in_proj_bias

    qkv_before = F.linear(z_before, weight, bias)
    qkv_after = F.linear(z_after, weight, bias)

    width = native_z.size(-1)
    k_before = qkv_before[..., width:2*width]
    v_before = qkv_before[..., 2*width:]
    k_after = qkv_after[..., width:2*width]
    v_after = qkv_after[..., 2*width:]

    score = m91.analyze_logits(output, groups)
    second = score["per_question_four_choice_accuracy"]

    audit = {
        "mean_L2_normalized_source_delta_L2":
            float((z_after-z_before).norm(dim=-1).mean()),
        "mean_L2_key_delta_L2":
            float((k_after-k_before).norm(dim=-1).mean()),
        "mean_L2_value_delta_L2":
            float((v_after-v_before).norm(dim=-1).mean()),
        "mean_L2_answer_attention_delta_L2":
            float(
                (
                    attention[idx, readout]
                    - native_attention[idx, readout]
                ).norm(dim=-1).mean()
            ),
        "second_event_accuracy4":
            float(
                0.5 * (
                    second["second_agent"]
                    + second["second_patient"]
                )
            ),
    }

    return output, audit


def run_one(model, groups, device, source):
    base_logits, baseline = evaluate(
        model, groups, device, source
    )
    results = {"native": baseline}

    for mode in MODES:
        logits, audit = evaluate(
            model, groups, device, source, mode
        )
        audit["mean_final_logits_delta_L2"] = float(
            (logits-base_logits).norm(dim=-1).mean()
        )
        results[mode] = audit

    return results


def preflight(groups):
    torch.manual_seed(96098)
    model = ml.Model().eval()
    batch = groups[:3]

    ids, ro, _, _ = ml.encode(
        [q for g in batch for q in g]
    )
    direct = model(ids, ro).float()
    observed, _ = evaluate(
        model, batch, "cpu", "first"
    )

    err = float((direct-observed).abs().max())
    assert err < 3e-4, f"Native-forward mismatch: {err}"

    _, half = evaluate(
        model, batch, "cpu", "first", "residual_half"
    )
    _, zero = evaluate(
        model, batch, "cpu", "first", "residual_zero"
    )
    _, norm = evaluate(
        model, batch, "cpu", "first", "normalized_half"
    )

    key = "mean_L2_normalized_source_delta_L2"
    assert half[key] < 1e-3, half
    assert zero[key] > 0.1, zero
    assert norm[key] > 0.1, norm

    print("PREFLIGHT PASS")
    print("Native parity:", err)
    print("L2 input delta, residual half:", half[key])
    print("L2 input delta, residual zero:", zero[key])
    print("L2 input delta, normalized half:", norm[key])
    print("No trained model or protected example loaded.")


def run(protocol_hash, groups, device):
    if OUT.exists():
        raise FileExistsError(f"Already audited: {OUT}")

    passed, counts = m88.model_gate(protocol_hash)
    if not passed:
        raise RuntimeError(
            f"ML-088 model gate failed: {counts}"
        )

    rows = []

    for seed in SEEDS:
        for arm in ARMS:
            checkpoint = (
                m96.ROOT / f"seed_{seed}" / arm
                / "model_step_12800.pt"
            )
            model = ml.load_checkpoint(
                checkpoint, seed, arm,
                protocol_hash, device
            )

            sources = {
                source: run_one(
                    model, groups, device, source
                )
                for source in SOURCE_PAIRS
            }

            rows.append({
                "seed": seed,
                "arm": arm,
                "sources": sources,
            })

            first = sources["first"]
            key = "mean_L2_normalized_source_delta_L2"
            print(
                f"seed={seed} arm={arm:8s} "
                f"LN-half={first['residual_half'][key]:.5g} "
                f"LN-zero={first['residual_zero'][key]:.3f} "
                f"LN-posthalf={first['normalized_half'][key]:.3f}",
                flush=True
            )

    summary = {}

    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        for source in SOURCE_PAIRS:
            summary[f"{arm}/{source}"] = {
                mode: {
                    key: mean(
                        r["sources"][source][mode][key]
                        for r in sub
                    )
                    for key in sub[0]["sources"][source][mode]
                }
                for mode in MODES
            }

    report = {
        "name": "ML-096 LayerNorm validity audit",
        "status": "exploratory implementation diagnostic",
        "script_sha256": sha(__file__),
        "ml096_sha256": sha(m96.__file__),
        "ml088_protocol_sha256": protocol_hash,
        "n_sentences": len(groups),
        "n_questions_per_model": 4 * len(groups),
        "n_models": len(rows),
        "protected_cases_generated": False,
        "summary": summary,
        "rows": rows,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")

    print("\nSUMMARY (familiar/first)")
    print(json.dumps(summary["familiar/first"], indent=2))
    print("Saved:", OUT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase", required=True,
        choices=("preflight", "run")
    )
    p.add_argument(
        "--device", default="auto",
        choices=("auto", "cpu", "mps", "cuda")
    )
    args = p.parse_args()

    protocol_hash, groups = select_data()

    if args.phase == "preflight":
        preflight(groups)
    else:
        run(
            protocol_hash, groups,
            ml.select_device(args.device)
        )


if __name__ == "__main__":
    main()
