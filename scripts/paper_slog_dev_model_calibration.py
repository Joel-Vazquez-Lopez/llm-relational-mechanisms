#!/usr/bin/env python3
"""SLOG model scoring engineering check: public in-distribution DEV ONLY.

Uses a small fixed 16-sentence calibration set from the official public SLOG
cogs_LF/dev.tsv; does not open the structural generalization archive or test.
This checks MPS execution, the chat-template boundary, A/B tokenization,
candidate order symmetry, event-role pairing and whether the model can follow
the task at all. NOT a preregistered external-validation result.

Run from repository root:
  python scripts/paper_slog_dev_model_calibration.py --phase preflight
  python scripts/paper_slog_dev_model_calibration.py --phase run --device mps

The full checkpoint revision and scoring implementation must be frozen before
any protected benchmark scores are opened.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from collections import Counter
from statistics import mean

DEV_SHA = "177128c3f64be56732a102db119139bbec33a771c485a525ea0f9f4e45926a5c"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
N = 16
OUT = Path("reports/paper_slog/dev_model_calibration.json")
DEV = Path("data/external/slog/cogs_LF/dev.tsv")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_sample():
    from paper_slog_preflight import extract
    if DEV.name != "dev.tsv" or DEV.parent.name != "cogs_LF":
        raise ValueError("Only SLOG cogs_LF/dev.tsv is supported")
    data = DEV.read_bytes()
    if digest(data) != DEV_SHA:
        raise ValueError("Dev data SHA256 differs from audited source")
    candidates = []
    for row_idx, line in enumerate(data.decode("utf8").splitlines(), 1):
        sentence, lf, category = line.split("\t")
        if category != "in_distribution":
            continue
        events, valid, reasons = extract(sentence, lf)
        if len(events) < 2 or not valid:
            continue
        # One selected eligible predicate per sentence: do not overweight
        # example families with many eligible events during initial gate.
        chosen = sorted(valid, key=lambda e: (
            e["verb_position_0based"], e["lemma"]))[0]
        candidates.append({
            "row": row_idx, "sentence": sentence, "category": category,
            "event": chosen
        })
    if len(candidates) < N:
        raise RuntimeError(f"Only {len(candidates)} eligible in-distribution rows; need {N}")
    chosen = sorted(
        candidates,
        key=lambda x: digest(f"{x['row']}|{x['sentence']}|calibration-v1".encode())
    )[:N]
    if len({x["row"] for x in chosen}) != N:
        raise AssertionError("Duplicate calibration sentence")
    return chosen, len(candidates)


def variants(item):
    e = item["event"]
    toks = item["sentence"].split()
    idx = e["verb_position_0based"]
    if toks[idx] != e["verb_surface"]:
        raise ValueError("Wrong predicate index")
    toks[idx] = f"[EVENT]{toks[idx]}[/EVENT]"
    marked = " ".join(toks)
    a, b = e["agent_surface"], e["theme_surface"]
    if a.casefold() == b.casefold():
        raise ValueError("Identical candidate surfaces")
    sorted_options = sorted([a, b], key=lambda z: (z.casefold(), z))
    out = []
    for role in ("agent", "theme"):
        for order in (0, 1):
            choices = sorted_options if order == 0 else sorted_options[::-1]
            gold = e[f"{role}_surface"]
            option = "A" if choices[0] == gold else "B"
            if role == "agent":
                question = "Which entity is the AGENT (the performer) of the marked event?"
            else:
                question = "Which entity is the THEME (the participant affected or acted upon) of the marked event?"
            prompt = (
                "Read this sentence. [EVENT]...[/EVENT] marks the verb whose participants are queried.\n"
                f"Sentence: {marked}\n"
                f"Question: {question}\n"
                f"A. {choices[0]}\nB. {choices[1]}\n"
                "Answer with only A or B."
            )
            out.append({
                "role": role, "order": order, "gold": option,
                "prompt": prompt, "choices": choices,
            })
    assert len(out) == 4
    assert [v["gold"] for v in out[:2]] in (["A", "B"], ["B", "A"])
    assert [v["gold"] for v in out[2:]] in (["A", "B"], ["B", "A"])
    return out


def preflight():
    rows, n_candidates = load_sample()
    all_questions = [q for row in rows for q in variants(row)]
    assert len(all_questions) == N * 4
    print("SLOG DEV MODEL CALIBRATION PREFLIGHT PASS")
    print(f"Official dev SHA256: {DEV_SHA}")
    print(f"In-distribution eligible multi-event candidate sentences: {n_candidates}")
    print(f"Calibration sentences: {N}, balanced questions: {len(all_questions)}")
    print(f"Selected row indices: {[r['row'] for r in rows]}")
    print(f"Selected-sample manifest SHA256: {digest(json.dumps(rows, sort_keys=True).encode())}")
    print("Strictly in_distribution public DEV. No structural models scored; no protected data read.")
    return rows


def load_model(device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS but torch.backends.mps.is_available() is false")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but it is unavailable")
    info = HfApi().model_info(MODEL_ID)
    revision = info.sha
    if not revision or len(revision) != 40:
        raise RuntimeError("Could not determine immutable model revision SHA")
    print(f"Resolved model snapshot commit: {revision}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    # A/B next-token scoring requires exactly one token per answer.
    ids = {}
    for option in ("A", "B"):
        tok = tokenizer(option, add_special_tokens=False).input_ids
        if len(tok) != 1:
            raise ValueError(f"'{option}' is not single-token under this model: {tok}")
        ids[option] = tok[0]
    if ids["A"] == ids["B"]:
        raise AssertionError("A and B share a token ID")
    print("Option token IDs:", ids, flush=True)
    # Float16 saves memory on 16 GB Apple Silicon; model remains read-only.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    return torch, tokenizer, model, ids, revision


def score_one(torch, tokenizer, model, ids, prompt, device):
    messages = [{"role": "user", "content": prompt}]
    enc = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        logits = model(**enc).logits[0, -1]
        a = float(logits[ids["A"]].float().item())
        b = float(logits[ids["B"]].float().item())
    if not math.isfinite(a) or not math.isfinite(b):
        raise FloatingPointError("Nonfinite scoring logits")
    if a == b:
        raise ValueError("A/B exact tie; need predeclared tie handling")
    return ("A" if a > b else "B"), a - b


def run(device):
    if OUT.exists():
        raise FileExistsError(f"Calibration already evaluated; no overwrite: {OUT}")
    selected = preflight()
    torch, tokenizer, model, ids, revision = load_model(device)
    all_rows = []
    for row in selected:
        vs = variants(row)
        scored = []
        for v in vs:
            pred, margin = score_one(torch, tokenizer, model, ids, v["prompt"], device)
            scored.append({
                "role": v["role"], "order": v["order"],
                "gold": v["gold"], "prediction": pred,
                "ab_logit_margin": margin,
                "correct": pred == v["gold"]
            })
        # Option 0 and 1 reverse candidate identities. Map selected labels
        # back to the actual participant head for unbiased aggregation.
        pred_names = []
        for j, v in enumerate(vs):
            pred_names.append(v["choices"][0 if scored[j]["prediction"] == "A" else 1])
        by_role = {
            role: pred_names[0 if role == "agent" else 2]
            for role in ("agent", "theme")
        }
        both_correct = all(scored[k]["correct"] for k in (0, 2))
        collapsed = pred_names[0] == pred_names[2]
        reversed_roles = (not both_correct and not collapsed)
        # Candidate order should not flip the selected participant.
        order_consistency = (
            pred_names[0] == pred_names[1] and pred_names[2] == pred_names[3]
        )
        all_rows.append({
            "row": row["row"], "sentence": row["sentence"],
            "event": row["event"], "scores": scored,
            "selected_participants": pred_names,
            "order0_correct": both_correct,
            "order0_collapsed": collapsed,
            "order0_reversed": reversed_roles,
            "both_order_consistent": order_consistency,
        })
        print(f"row={row['row']} both={int(both_correct)} "
              f"collapse={int(collapsed)} reversed={int(reversed_roles)} "
              f"order_consistent={int(order_consistency)}", flush=True)
    summary = {
        "experiment": "SLOG public in-distribution DEV ONLY model calibration",
        "not_external_validation": True,
        "no_protected_data_opened": True,
        "model_id": MODEL_ID,
        "revision_sha": revision,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "device": device,
        "dataset_sha256": DEV_SHA,
        "script_sha256": digest(Path(__file__).read_bytes()),
        "n_sentences": N,
        "n_question_variants": N * 4,
        "order0_pair_accuracy": mean(int(r["order0_correct"]) for r in all_rows),
        "order0_collapse": mean(int(r["order0_collapsed"]) for r in all_rows),
        "order0_reverse": mean(int(r["order0_reversed"]) for r in all_rows),
        "both_role_answer_order_consistency": mean(int(r["both_order_consistent"]) for r in all_rows),
        "n_question_correct": sum(int(z["correct"]) for r in all_rows for z in r["scores"]),
        "records": all_rows,
        "warning": (
            "Development-only engineering/competence gate. Does not support "
            "external generalization. A/B next-token scoring is NOT native "
            "free-text generation and can suffer candidate-order artifacts. "
            "Order0 is not itself an order-invariant scoring rule; both orders "
            "are recorded. Freeze final aggregation and gate before test."
        )
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print("DEVELOPMENT CALIBRATION SUMMARY")
    print(json.dumps({k: v for k,v in summary.items() if k != "records"}, indent=2))
    print("Saved", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("preflight", "run"), required=True)
    ap.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    a = ap.parse_args()
    if a.phase == "preflight":
        preflight()
    else:
        run(a.device)


if __name__ == "__main__":
    main()
