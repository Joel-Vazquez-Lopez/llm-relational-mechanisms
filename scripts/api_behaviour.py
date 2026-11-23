#!/usr/bin/env python
"""API-BEH: the failing question on models reachable only through an API (behaviour only).

Works with any OpenAI-compatible chat endpoint (Berget, Google Gemini's OpenAI endpoint, Groq,
OpenRouter, Mistral, ...). The key is read from an environment variable and never written
anywhere. Run it where you have internet and the key (e.g. the MacBook):

  export BERGET_API_KEY=...            # or GEMINI_API_KEY=...
  python scripts/api_behaviour.py --list-models --base-url https://api.berget.ai/v1 --key-env BERGET_API_KEY
  python scripts/api_behaviour.py --base-url https://api.berget.ai/v1 --key-env BERGET_API_KEY \
      --model <model-id> --set otr --n-per 40
  python scripts/api_behaviour.py --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
      --key-env GEMINI_API_KEY --model <gemini-model-id> --set otr --n-per 40 --sleep 4

Conditions (same sentence, one user message each):
  default      the item's ordinary agent question ("Who {past} someone?"), one name
  failing      "Who was someone {past} by?", one name
  working      "Who was the {ing} done by?", one name
  failing_full "Who was someone {past} by?", answer in a full sentence
Answers are classified with the generation-check extractor (direct name / statement / inferred).
--set wiki uses WIKI-ROLES items (real names) from --wiki-file. Results are cached per request
in <out>.cache.jsonl, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from otr_generate_check import extract_answer  # noqa: E402

from relational_lab.experiments import order_role_stimuli as ors  # noqa: E402
from relational_lab.experiments import wiki_roles as wr  # noqa: E402

API_BEH_VERSION = "0.1.0"
ONE_NAME = "Reply with one name only."
FULL = "Reply with a full sentence."
CONDITIONS = {
    "default": (None, ONE_NAME),
    "failing": ("Who was someone {past} by?", ONE_NAME),
    "working": ("Who was the {ing} done by?", ONE_NAME),
    "failing_full": ("Who was someone {past} by?", FULL),
}
MAIN = ("active", "object_cleft", "passive", "by_cleft")


def _req(url, key, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def chat(base, key, model, prompt, max_tokens, extra, tries=8):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": max_tokens, **extra}
    for k in range(tries):
        try:
            out = _req(base.rstrip("/") + "/chat/completions", key, body)
            msg = out["choices"][0]["message"]
            return (msg.get("content") or "").strip()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and k < tries - 1:
                time.sleep(min(60, 5 * 2 ** k))
                continue
            raise SystemExit(f"HTTP {e.code}: {e.read().decode()[:400]}")
        except (urllib.error.URLError, TimeoutError):
            if k < tries - 1:
                time.sleep(min(60, 5 * 2 ** k))
                continue
            raise
    return ""


def otr_items(n_per, seed):
    pool = [i for i in ors.generate_items(seed=seed) if i.split == "dev" and i.construction in MAIN
            and i.role_asked == "agent" and i.question_voice == "active"]
    rng = random.Random(seed)
    out = []
    for c in MAIN:
        sub = [i for i in pool if i.construction == c]
        rng.shuffle(sub)
        out += sub[:n_per]
    return out


def otr_prompt(it, cond):
    tpl, instr = CONDITIONS[cond]
    q = it.question if tpl is None else o9_fill(tpl, it)
    return it.prompt.replace(f"\nQuestion: {it.question}\n", f"\nQuestion: {q}\n").replace(
        "\nAnswer:", f"\n{instr}\nAnswer:"), q


def o9_fill(tpl, it):
    from relational_lab.experiments.question_templates import fill as fill_template
    return fill_template(tpl, it.verb, it.lemma)


def wiki_rows(path, n, seed):
    recs = wr.load_records(path)
    rng = random.Random(seed)
    act = [r for r in recs if r["kind"] == "active"]
    pas = [r for r in recs if r["kind"] == "passive"]
    rng.shuffle(act)
    rng.shuffle(pas)
    return act[: n // 2] + pas[: n - n // 2]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--key-env", required=True, help="name of the environment variable holding the key")
    p.add_argument("--model")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--set", default="otr", choices=["otr", "wiki"])
    p.add_argument("--wiki-file", type=Path, default=ROOT / "results" / "wiki_roles_en.jsonl")
    p.add_argument("--n-per", type=int, default=40, help="otr: sentences per construction")
    p.add_argument("--n-wiki", type=int, default=200, help="wiki: items (half active, half passive)")
    p.add_argument("--conditions", default="default,failing,working,failing_full")
    p.add_argument("--max-tokens", type=int, default=512, help="room for models that think before answering")
    p.add_argument("--extra-body", default="{}", help='JSON merged into the request, e.g. \'{"reasoning_effort": "low"}\'')
    p.add_argument("--sleep", type=float, default=0.0, help="seconds between requests (free tiers: 4-6)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    a = p.parse_args(argv)
    key = os.environ.get(a.key_env)
    if not key:
        raise SystemExit(f"set the environment variable {a.key_env} first")
    if a.list_models:
        for m in _req(a.base_url.rstrip("/") + "/models", key).get("data", []):
            print(m.get("id"))
        return
    if not a.model:
        raise SystemExit("--model is required (see --list-models)")
    extra = json.loads(a.extra_body)
    conds = [c.strip() for c in a.conditions.split(",")]
    slug = a.model.replace("/", "__").replace(":", "_")
    out = a.out_dir / f"apibeh_{slug}_{a.set}.json"
    cache_path = out.with_suffix(".cache.jsonl")
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf8").splitlines():
            if line.strip():
                r = json.loads(line)
                cache[(r["cond"], r["item"])] = r["output"]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    with cache_path.open("a", encoding="utf8") as cf:
        def ask(cond, item_key, prompt):
            if (cond, item_key) in cache:
                return cache[(cond, item_key)]
            text = chat(a.base_url, key, a.model, prompt, a.max_tokens, extra)
            cf.write(json.dumps({"cond": cond, "item": item_key, "output": text}, ensure_ascii=False) + "\n")
            cf.flush()
            cache[(cond, item_key)] = text
            if a.sleep:
                time.sleep(a.sleep)
            return text

        if a.set == "otr":
            for it in otr_items(a.n_per, a.seed):
                for cond in conds:
                    prompt, q = otr_prompt(it, cond)
                    text = ask(cond, it.item_id, prompt)
                    kind, name = extract_answer(text, [it.agent, it.patient], it.sentence, q,
                                                agent=it.agent, patient=it.patient, role="agent")
                    rows.append({"cond": cond, "item": it.item_id, "construction": it.construction,
                                 "question": q, "output": text, "kind": kind, "correct": name == it.agent})
        else:
            for idx, rec in enumerate(wiki_rows(a.wiki_file, a.n_wiki, a.seed)):
                for cond in [c for c in conds if c != "default"]:
                    tpl, instr = CONDITIONS[cond]
                    it = wr.build_item(rec, idx, tpl.replace("{past}", "{pp}"), "real")
                    if it is None:
                        continue
                    prompt = it.prompt.replace("\nAnswer:", f"\n{instr}\nAnswer:")
                    text = ask(cond, f"wiki:{idx}", prompt)
                    kind, name = extract_answer(text, [it.agent, it.patient], it.sentence, it.question,
                                                agent=it.agent, patient=it.patient, role="agent")
                    rows.append({"cond": cond, "item": f"wiki:{idx}", "construction": it.construction,
                                 "question": it.question, "output": text, "kind": kind,
                                 "correct": name == it.agent})
            print(f"  {len(rows)} answers, {time.time() - t0:.0f}s", flush=True)

    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["cond"], r["construction"])].append(r["correct"])
        cells[(r["cond"], "all")].append(r["correct"])
    summary = {f"{c}|{k}": round(sum(v) / len(v), 3) for (c, k), v in sorted(cells.items())}
    report = {"experiment": "API-BEH", "version": API_BEH_VERSION, "base_url": a.base_url, "model": a.model,
              "set": a.set, "conditions": conds, "n_answers": len(rows), "accuracy": summary,
              "kinds": dict(collections.Counter(r["kind"] for r in rows)),
              "max_tokens": a.max_tokens, "extra_body": extra, "seconds": round(time.time() - t0), "rows": rows}
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k.endswith("|all") or "|active" in k}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
