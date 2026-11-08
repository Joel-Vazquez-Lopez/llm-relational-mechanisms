"""ML-083: Can event- and role-cue activation changes compose?

From the repo root, with this file saved in scripts/:
  python3 scripts/ml083_compositional_layer_patching.py --phase plan
  python3 scripts/ml083_compositional_layer_patching.py --phase preflight
  # Commit code and reports/ml083/frozen_protocol.json before inspecting results.
  python3 scripts/ml083_compositional_layer_patching.py --phase evaluate

All cases come from familiar active/active ML-080 CALIBRATION examples.
Never imports or generates any protected ML-080 case. Uses the same 16 trained
models; this is exploratory within-model work, not independent replication.
"""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean

import torch

import ml080_learning_curve as ml
import ml081_layerwise_event_role as ml081

SEEDS = tuple(ml.SEEDS)
ARMS = tuple(ml.ARMS)
N_SENTENCES = 64
START = 112                   # ML-081 used 0:48; ML-082 used 48:112.
RANDOM_REPS = 3
SITES = ((0, "attn"), (0, "mlp"),
         (1, "attn"), (1, "mlp"),
         (2, "attn"), (2, "mlp"))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sentence_grid(doc):
    cal = ml.base_cases(doc["splits"]["calibration_pairs"])
    second_agent = [c for c in cal if c.target_pos == 1
                    and c.target_voice == "active"
                    and c.distractor_voice == "active"
                    and c.role == "agent"]
    assert len(second_agent) == 576
    second_agent.sort(key=lambda c: hashlib.sha256(
        " ".join(ml.prompt(c)).encode()).hexdigest())
    chosen = second_agent[START:START + N_SENTENCES]
    assert len(chosen) == N_SENTENCES
    assert not set(" ".join(ml.prompt(c)) for c in chosen) & set(
        " ".join(ml.prompt(c)) for c in second_agent[:START])

    grid = {k: [] for k in ("00", "01", "10", "11")}
    for s_agent in chosen:
        # The four corners share EXACTLY the same two-event sentence.
        f_agent = ml081.swap_event(s_agent)
        f_patient = replace(f_agent, role="patient")
        s_patient = replace(s_agent, role="patient")
        assert ml081.changed_tokens(f_agent, f_patient) == [
            ("agent", "patient")]
        assert ml081.changed_tokens(f_agent, s_agent) == [
            ("first", "second")]
        assert ml081.changed_tokens(s_agent, s_patient) == [
            ("agent", "patient")]
        assert ml081.changed_tokens(f_patient, s_patient) == [
            ("first", "second")]
        prompts = [ml.prompt(x) for x in (
            f_agent, f_patient, s_agent, s_patient)]
        prefixes = [p[:p.index("<sep>")] for p in prompts]
        assert len(set(map(tuple, prefixes))) == 1
        assert len({ml.answer(c) for c in (
            f_agent, f_patient, s_agent, s_patient)}) == 4
        for key, c in zip(grid, (
            f_agent, f_patient, s_agent, s_patient)):
            grid[key].append(c)

    assert all(len(v) == N_SENTENCES for v in grid.values())
    assert len(set(map(tuple, (ml.prompt(c) for c in grid["00"])))) == N_SENTENCES
    return grid


def protocol_document(source_doc, grid):
    return {
        "experiment": "ML-083",
        "script_sha256": ml.sha_file(__file__),
        "ml080_protocol_sha256": source_doc["sha256"],
        "ml081_source_sha256": ml.sha_file(ml081.__file__),
        "case_selection": (
            "SHA256-sorted active/active second-agent CALIBRATION prompts "
            "indices 112:176; 0:48 ML-081, 48:112 ML-082"
        ),
        "sentences": N_SENTENCES,
        "sample_sha256": canonical_hash({
            key: [ml.prompt(c) for c in cases] for key, cases in grid.items()
        }),
        "seeds": SEEDS,
        "arms": ARMS,
        "step": 12800,
        "sites": [list(site) for site in SITES],
        "grid": {"00": "first agent (recipient)",
                 "01": "first patient (role-only donor)",
                 "10": "second agent (event-only donor)",
                 "11": "second patient (joint target, not used for composition)"},
        "composition": "h10 + h01 - h00, final-token component at one site",
        "primary_site": [1, "attn"],  # Zero-based layer 1 = L2 attention.
        "primary_per_seed_measure": (
            "four-choice target-hit for composition minus target-hit for "
            "cyclically shuffled role delta normalized to composition norm; "
            "report all 8 seeds by exposure arm, no post hoc seed exclusions"
        ),
        "controls": [
            "true h11 direct joint donor (positive reference; not a test of additivity)",
            "event-only h10 and role-only h01 (not norm matched)",
            "h10 + role delta from a different sentence, normalized to composition norm",
            "h00 + event delta - role delta, normalized to composition norm",
            "3 independent isotropic Gaussian perturbations normalized per-example to composition norm"
        ],
        "extra_metrics": [
            "full-vocabulary target hit", "native four-question accuracy",
            "median interaction norm ||h11-h10-h01+h00||",
            "relative interaction norm divided by ||h11-h00||",
            "all condition-specific patch norms and candidate-hit rates"
        ],
        "interpretation": (
            "Same-model exploratory calibration, not unseen grammar or independent model testing. "
            "Successful composition tests a manipulation of final-token states, not native "
            "necessity or a grammatical algorithm; successful shuffled controls may "
            "indicate context-general transformations rather than failure. "
            "The two apparent addition orders are algebraically identical."
        ),
        "result_rule": (
            "freeze and commit script plus generated protocol before evaluate; "
            "do not overwrite existing result file; no protected cases"
        ),
    }


def freeze(path, doc, grid, create):
    base = protocol_document(doc, grid)
    full = json.loads(json.dumps({**base, "sha256": canonical_hash(base)}))
    if create and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(full, indent=2) + "\n")
    if not path.is_file():
        raise FileNotFoundError(f"Freeze protocol first: --phase plan ({path})")
    assert json.loads(path.read_text()) == full, (
        "ML-083 script, samples or protocol changed since freeze")
    return full["sha256"]


def prepared(cases, device):
    ids, ro, labels, choices = ml.encode(cases)
    return ids.to(device), ro.to(device), labels.to(device), choices.to(device)


def pred_candidates(logits, choices):
    col = logits.gather(1, choices).argmax(1, keepdim=True)
    return choices.gather(1, col).flatten()


def norm_match(delta, ref):
    magnitude = ref.norm(dim=1, keepdim=True)
    return delta / delta.norm(dim=1, keepdim=True).clamp_min(1e-12) * magnitude


def generate_random_like(ref, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(ref.shape, generator=generator, device="cpu").to(ref.device)
    return norm_match(noise, ref)


@torch.inference_mode()
def score_one(model, grid, device, seed, arm):
    model.eval()
    inputs = {key: prepared(items, device) for key, items in grid.items()}
    base_ids, base_ro, base_gold, choices = inputs["00"]
    target_gold = inputs["11"][2]
    assert bool(torch.all(base_gold != target_gold))

    logits, traces = {}, {}
    for key, (ids, ro, _, current_choices) in inputs.items():
        logit, trace = ml081.forward_trace(model, ids, ro)
        logits[key], traces[key] = logit, trace
        assert bool(torch.all(current_choices == choices))
    error = float((logits["00"]-model(base_ids, base_ro).float()).abs().max())
    assert error < 1e-3, f"Forward parity failed seed {seed}/{arm}: {error}"

    native_each = {key: float((pred_candidates(logits[key], choices) == inputs[key][2])
                              .float().mean()) for key in grid}
    native_all_four = float(torch.stack([
        (pred_candidates(logits[key], choices) == inputs[key][2])
        for key in grid]).all(0).float().mean())

    def patched_hit(layer, site, activation):
        patched, _ = ml081.forward_trace(
            model, base_ids, base_ro, patch=(layer, site, activation))
        hit = (pred_candidates(patched, choices) == target_gold).float().mean()
        full_hit = (patched.argmax(1) == target_gold).float().mean()
        return float(hit), float(full_hit)

    out = []
    for layer, site in SITES:
        a = traces["00"][layer][site]
        b = traces["01"][layer][site]
        c = traces["10"][layer][site]
        d = traces["11"][layer][site]
        role = b-a
        event = c-a
        composed_delta = event+role
        composed = a+composed_delta
        interaction = d-composed
        shuffled_role = role.roll(shifts=1, dims=0)
        shuffled_delta = norm_match(event + shuffled_role, composed_delta)
        opposed_delta = norm_match(event-role, composed_delta)

        methods = {
            "compose": composed,
            "joint_direct": d,
            "event_only": c,
            "role_only": b,
            "shuffled_structured": a + shuffled_delta,
            "opposed_role": a + opposed_delta,
        }
        result = {}
        for name, activation in methods.items():
            hit, full = patched_hit(layer, site, activation)
            result[name + "_hit"] = hit
            result[name + "_full_vocab_hit"] = full

        random_hits, random_full = [], []
        for rep in range(RANDOM_REPS):
            rand_seed = (seed * 1000000 + (0 if arm == "familiar" else 1) * 100000
                         + layer * 10000 + (0 if site == "attn" else 1) * 1000
                         + 830 + rep)
            random_delta = generate_random_like(composed_delta, rand_seed)
            hit, full = patched_hit(layer, site, a + random_delta)
            random_hits.append(hit)
            random_full.append(full)
        result["random_hit"] = mean(random_hits)
        result["random_full_vocab_hit"] = mean(random_full)
        result.update({
            "seed": seed, "arm": arm, "layer": layer+1, "site": site,
            "n_sentences": N_SENTENCES,
            "native_all_four_accuracy": native_all_four,
            "native_target_accuracy": native_each["11"],
            "native_base_accuracy": native_each["00"],
            "native_event_accuracy": native_each["10"],
            "native_role_accuracy": native_each["01"],
            "compose_minus_shuffled": result["compose_hit"] - result["shuffled_structured_hit"],
            "compose_minus_random": result["compose_hit"] - result["random_hit"],
            "median_compose_delta_norm": float(composed_delta.norm(dim=1).median()),
            "median_event_delta_norm": float(event.norm(dim=1).median()),
            "median_role_delta_norm": float(role.norm(dim=1).median()),
            "median_joint_delta_norm": float((d-a).norm(dim=1).median()),
            "median_interaction_norm": float(interaction.norm(dim=1).median()),
            "median_relative_interaction": float((interaction.norm(dim=1) /
                (d-a).norm(dim=1).clamp_min(1e-12)).median()),
        })
        out.append(result)
    return error, native_each, native_all_four, out


def print_summary(rows):
    print("\nML-083 — four-corner additive composition")
    print("All rates are four-choice target-hit, averaged over eight seeds.")
    print(f"{'Layer':<7}{'Site':<8}{'Arm':<10}{'Compose':>10}{'Shuffled':>10}"
          f"{'Opposed':>10}{'Random':>10}{'Direct':>10}{'C-S':>10}")
    for layer, site in SITES:
        for arm in ARMS:
            g = [r for r in rows if r["layer"] == layer+1
                 and r["site"] == site and r["arm"] == arm]
            assert len(g) == len(SEEDS)
            fmt = lambda k: mean(v[k] for v in g)
            print(f"{layer+1:<7}{site:<8}{arm:<10}"
                  f"{fmt('compose_hit'):>10.3f}"
                  f"{fmt('shuffled_structured_hit'):>10.3f}"
                  f"{fmt('opposed_role_hit'):>10.3f}"
                  f"{fmt('random_hit'):>10.3f}"
                  f"{fmt('joint_direct_hit'):>10.3f}"
                  f"{fmt('compose_minus_shuffled'):>+10.3f}")
    print("\nPREDECLARED primary L2 attention compose minus shuffled structural control")
    for arm in ARMS:
        r = [next(r for r in rows if r["seed"] == seed and r["arm"] == arm
                  and r["layer"] == 2 and r["site"] == "attn") for seed in SEEDS]
        vals = [v["compose_minus_shuffled"] for v in r]
        print(f"  {arm}: mean={mean(vals):+.3f}; paired seeds=" +
              ", ".join(f"{v:+.3f}" for v in vals))
    print("Exploratory same-model calibration; no ML-080 protected cases accessed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("plan", "preflight", "evaluate"))
    ap.add_argument("--protocol", default="reports/ml083/frozen_protocol.json")
    ap.add_argument("--ml080-protocol", default="reports/ml080/frozen_protocol.json")
    ap.add_argument("--root", default="results/ml080_learning_curve")
    ap.add_argument("--output", default="reports/ml083/compositional_patching.json")
    ap.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = ap.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    src = ml081.verify_ml080(Path(args.ml080_protocol))
    grid = sentence_grid(src)
    phash = freeze(Path(args.protocol), src, grid, args.phase == "plan")
    if args.phase == "plan":
        print(f"ML-083 frozen {args.protocol} sha256={phash[:12]}")
        return

    # Must not require checkpoint files for a CPU code-only smoke preflight.
    demo = ml.Model().eval()
    inputs = {key: prepared(group[:4], "cpu") for key, group in grid.items()}
    a_ids, a_ro, _, _ = inputs["00"]
    with torch.inference_mode():
        native, sites = ml081.forward_trace(demo, a_ids, a_ro)
        parity = float((native-demo(a_ids, a_ro)).abs().max())
        assert parity < 1e-3
        same, _ = ml081.forward_trace(
            demo, a_ids, a_ro, patch=(1, "attn", sites[1]["attn"]))
        assert float((same-native).abs().max()) < 1e-3
        _, t01 = ml081.forward_trace(demo, *inputs["01"][:2])
        _, t10 = ml081.forward_trace(demo, *inputs["10"][:2])
        a, b, c = sites[1]["attn"], t01[1]["attn"], t10[1]["attn"]
        compose = a + (b-a) + (c-a)
        assert torch.allclose(compose, b+c-a, atol=2e-5)
    print("Preflight PASS: frozen protocol, disjoint calibration selection, "
          f"forward reconstruction ({parity:.2e}), identity patch, algebra")
    if args.phase == "preflight":
        return

    ml081.check_runs(Path(args.root), src)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite completed results: {output}")
    device = ml.select_device(args.device)
    report = {
        "experiment": "ML-083", "protocol_sha256": phash,
        "analysis_scope": "ML-080 familiar calibration; same 16 models; exploratory",
        "n_sentences": N_SENTENCES,
        "device": device,
        "rows": [],
    }
    for seed in SEEDS:
        for arm in ARMS:
            checkpoint = (Path(args.root) / f"seed_{seed}" / arm /
                          "model_step_12800.pt")
            model = ml.load_checkpoint(checkpoint, seed, arm, src["sha256"], device)
            parity, native, allfour, rows = score_one(model, grid, device, seed, arm)
            report["rows"].extend(rows)
            primary = next(r for r in rows if r["layer"] == 2 and r["site"] == "attn")
            print(f"seed={seed} arm={arm:8s} all_four_native={allfour:.3f} "
                  f"L2_attn_composed={primary['compose_hit']:.3f} "
                  f"shuffled={primary['shuffled_structured_hit']:.3f} "
                  f"direct_joint={primary['joint_direct_hit']:.3f} "
                  f"parity={parity:.2e}", flush=True)
    assert len(report["rows"]) == len(SEEDS) * len(ARMS) * len(SITES)
    print_summary(report["rows"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output}; ML-080 protected cases never generated.")


if __name__ == "__main__":
    main()
