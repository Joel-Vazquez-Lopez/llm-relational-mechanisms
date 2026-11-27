"""Run the survey battery against an OpenAI-compatible chat API (behaviour only).

Same items, prompts and scoring as scripts/otr_generate_check.py (via prepared_prompts /
make_row / make_report), same battery as scripts/otr_battery.py. Writes otrgen_*.json
files in the usual format with "api": true. No logprobs or layers.

The API key is read from the environment variable named by --key-env (never from a file):

    export OPENAI_API_KEY=...        # or BERGET_API_KEY, GEMINI_API_KEY, ...
    python scripts/otr_generate_api.py --base-url https://api.openai.com/v1 --model gpt-5.1 --key-env OPENAI_API_KEY
    python scripts/otr_generate_api.py --base-url https://api.berget.ai/v1 --model zai-org/GLM-5.3-Flash --key-env BERGET_API_KEY

Endpoints that speak this protocol include OpenAI, Google Gemini
(https://generativelanguage.googleapis.com/v1beta/openai/), Anthropic
(https://api.anthropic.com/v1/), DeepSeek (https://api.deepseek.com/v1), Berget
(https://api.berget.ai/v1). Model names and endpoint details change; check the provider.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gc = _load("otr_generate_check", "otr_generate_check.py")
bat = _load("otr_battery", "otr_battery.py")


def chat(base_url, key, model, prompt, max_tokens, temperature, extra, retries=6):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    if temperature is not None:
        body["temperature"] = temperature
    body.update(extra)
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            msg = out["choices"][0]["message"]
            return msg.get("content") or "", out.get("model", model)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--key-env", required=True, help="name of the environment variable holding the API key")
    p.add_argument("--only", default="", help="battery subset, e.g. B1,B2,B3")
    p.add_argument("--n-scale", type=float, default=0.5, help="default half size (328 items per condition)")
    p.add_argument("--max-tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-temperature", action="store_true", help="omit temperature (some reasoning models reject it)")
    p.add_argument("--extra", default="{}", help='extra JSON body fields, e.g. \'{"reasoning_effort": "low"}\'')
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--label", default="", help="name used in output files (default: --model)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)
    key = os.environ.get(args.key_env)
    if not key:
        p.error(f"environment variable {args.key_env} is not set")
    extra = json.loads(args.extra)
    temp = None if args.no_temperature else args.temperature
    label = args.label or args.model
    want = {x.strip() for x in args.only.split(",") if x.strip()}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # one quick call first, so a wrong model name or key fails before the batch
    text, served = chat(args.base_url, key, args.model, "Reply with the single word OK.", 5, temp, extra)
    print(f"connected: {served!r} -> {text!r}")

    for bid, template, role, instr, n in bat.BATTERY:
        if want and bid not in want:
            continue
        pairs = gc.prepared_prompts("otr", 2026, max(16, int(n * args.n_scale)), template or "", role, instr)
        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as ex:
            outs = list(ex.map(lambda ip: chat(args.base_url, key, args.model, ip[1], args.max_tokens, temp, extra), pairs))
        rows = [gc.make_row(it, prompt, text) for (it, prompt), (text, _) in zip(pairs, outs)]
        report = gc.make_report(label, rows, time.time() - t0, api=True, base_url=args.base_url,
                                served_model=outs[0][1] if outs else None, chat_template=True, stimuli="otr",
                                instruction=instr, question_template=template, question_role=role,
                                max_new_tokens=args.max_tokens, temperature=temp, extra=extra, battery_id=bid,
                                battery_version=bat.BATTERY_VERSION)
        tag = ("_q-" + "".join(c if c.isalpha() else "-" for c in template.lower()).strip("-")) if template else ""
        while "--" in tag:
            tag = tag.replace("--", "-")
        out = args.out_dir / f"otrgen_{label.replace('/', '__')}_dev{tag}_{time.strftime('%Y%m%dT%H%M%S')}.json"
        out.write_text(json.dumps(report, indent=1))
        print(f"{bid}: acc {report['generation_accuracy']:.2f}  kinds {report['kinds']}  -> {out.name}")
        time.sleep(1.1)


if __name__ == "__main__":
    main()
