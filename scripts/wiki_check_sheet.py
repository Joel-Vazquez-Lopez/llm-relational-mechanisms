#!/usr/bin/env python
"""Human check sheet for WIKI-ROLES: a random sample as CSV (open in Excel / Numbers).

The checker marks, for each row: agent_ok (is AGENT really the one who did it?), patient_ok,
natural_ok (is the sentence a normal, complete sentence?). Parser error rates from this sheet are
reported in the paper.

  python scripts/wiki_check_sheet.py results/wiki_roles_en.jsonl --n 200 --out docs/order_to_role/wiki_check_sheet.csv
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("jsonl", type=Path)
p.add_argument("--n", type=int, default=200)
p.add_argument("--out", type=Path, default=Path("docs/order_to_role/wiki_check_sheet.csv"))
p.add_argument("--seed", type=int, default=2026)
a = p.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relational_lab.experiments import wiki_roles as wr  # noqa: E402
recs = wr.load_records(a.jsonl)  # drops sentences cut after an initial
act = [r for r in recs if r["kind"] == "active"]
pas = [r for r in recs if r["kind"] == "passive"]
rng = random.Random(a.seed)
sample = rng.sample(act, min(len(act), a.n // 2)) + rng.sample(pas, min(len(pas), a.n - a.n // 2))
rng.shuffle(sample)
a.out.parent.mkdir(parents=True, exist_ok=True)
with a.out.open("w", newline="", encoding="utf8") as f:
    w = csv.writer(f)
    w.writerow(["row", "title", "kind", "verb", "AGENT (did it)", "PATIENT (it was done to)", "sentence",
                "agent_ok (y/n)", "patient_ok (y/n)", "natural_ok (y/n)", "comment"])
    for i, r in enumerate(sample, 1):
        w.writerow([i, r.get("title", ""), r["kind"], r["lemma"], r["agent"], r["patient"], r["sentence"], "", "", "", ""])
print(f"wrote {len(sample)} rows to {a.out}")
