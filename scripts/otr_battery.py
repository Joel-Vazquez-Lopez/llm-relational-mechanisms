"""Model survey battery: load a model once, run the fixed set of generation conditions.

Each condition is an ordinary generation-check run (same items, prompts, scoring and
output format as scripts/otr_generate_check.py), so every analysis and figure script
works unchanged. A manifest JSON lists the files written.

Battery v1 (BATTERY below):
  B0  default OTR questions, name only            inclusion check (should be >= 0.95)
  B1  "Who was someone V-ed by?"  agent, name     the effect
  B2  "Who was someone V-ed by?"  agent, sentence the dissociation
  B3  "Who was the V-ing done by?" agent, name    no participant expression (control)
  B4  "Who was V-ed by someone?"  patient, name   same-words patient permutation
  B5  "By whom was someone V-ed?" agent, name     pied-piped
  B6  "Who was the other person V-ed by?" agent, name   definite participant expression

    python scripts/otr_battery.py --model allenai/Olmo-3.1-32B-Instruct --chat-template
    python scripts/otr_battery.py --model openai/gpt-oss-20b --chat-template --assistant-prefix "<|channel|>final<|message|>"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BATTERY_VERSION = "1.0.0"
NAME, SENT = "Reply with one name only.", "Answer with a full sentence."
BATTERY = [
    ("B0", None, None, NAME, 512),
    ("B1", "Who was someone {past} by?", "agent", NAME, 656),
    ("B2", "Who was someone {past} by?", "agent", SENT, 656),
    ("B3", "Who was the {ing} done by?", "agent", NAME, 656),
    ("B4", "Who was {past} by someone?", "patient", NAME, 656),
    ("B5", "By whom was someone {past}?", "agent", NAME, 656),
    ("B6", "Who was the other person {past} by?", "agent", NAME, 656),
]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--only", default="", help="comma-separated subset, e.g. B1,B2")
    p.add_argument("--n-scale", type=float, default=1.0, help="scale item counts (0.5 = half, for big/slow models)")
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--template-kwarg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--assistant-prefix", default="")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)

    import torch

    import relational_lab.instrumentation.hf_scoring as hs
    tok = hs.load_tokenizer(args.model)
    model = hs.load_lm(args.model, dtype=getattr(torch, args.dtype))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    hs.load_tokenizer = lambda *a, **k: tok       # every run below reuses the loaded model
    hs.load_lm = lambda *a, **k: model

    spec = importlib.util.spec_from_file_location("otr_generate_check", ROOT / "scripts" / "otr_generate_check.py")
    gc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gc)

    want = {x.strip() for x in args.only.split(",") if x.strip()}
    manifest = {"battery_version": BATTERY_VERSION, "model": args.model, "runs": []}
    before = set(args.out_dir.glob("otrgen_*.json")) if args.out_dir.exists() else set()
    for bid, template, role, instr, n in BATTERY:
        if want and bid not in want:
            continue
        argv_run = ["--model", args.model, "--n", str(max(16, int(n * args.n_scale))), "--instruction", instr,
                    "--max-new-tokens", str(args.max_new_tokens), "--dtype", args.dtype, "--out-dir", str(args.out_dir)]
        if template:
            argv_run += ["--question-template", template, "--question-role", role]
        if args.chat_template:
            argv_run.append("--chat-template")
        if args.no_thinking:
            argv_run.append("--no-thinking")
        for kv in args.template_kwarg:
            argv_run += ["--template-kwarg", kv]
        if args.assistant_prefix:
            argv_run += ["--assistant-prefix", args.assistant_prefix]
        print(f"=== {bid}: {template or 'default questions'} [{role or 'both'}] / {instr}", flush=True)
        t0 = time.time()
        gc.main(argv_run)
        after = set(args.out_dir.glob("otrgen_*.json"))
        new = sorted(after - before, key=lambda f: f.stat().st_mtime)
        before = after
        manifest["runs"].append({"id": bid, "template": template, "role": role, "instruction": instr,
                                 "file": new[-1].name if new else None, "seconds": round(time.time() - t0, 1)})
        time.sleep(1.1)  # distinct timestamps in file names
    out = args.out_dir / f"battery_{args.model.replace('/', '__')}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(manifest, indent=1))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
