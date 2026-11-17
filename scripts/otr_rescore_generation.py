"""Rescore saved generation-check runs with the current answer extractor.

Generation answers are re-classified from the stored text, so no GPU is needed. The
extractor now counts event statements ("Frank was watched by Daniel.") as answers naming
the asked role (kind "statement"); earlier runs counted them as "ambiguous" (wrong).

For each run it reports accuracy three ways:
  all        every answer (ambiguous / none = wrong), statements counted;
  bare       bare-name answers only (kind "direct");
  statement  event-statement answers only;
plus per construction x role x question-voice cells. Runs from before rows stored
``role`` are skipped.

    python scripts/otr_rescore_generation.py results/otrgen_*.json --out results/rescored_generation.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relational_lab.experiments import order_role_stimuli as ors  # noqa: E402

_spec = importlib.util.spec_from_file_location("otr_generate_check", ROOT / "scripts" / "otr_generate_check.py")
gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gc)


def rescore(run: dict, items: dict) -> dict | None:
    rows = run.get("rows") or []
    if not rows or "role" not in rows[0]:
        return None
    kinds, cells = Counter(), defaultdict(list)
    acc = {"all": [], "bare": [], "statement": []}
    out_rows = []
    for r in rows:
        it = items.get(r["item_id"])
        if it is None:
            return None
        question = r.get("question") or it.question
        kind, name = gc.extract_answer(r["output"], [it.answer, it.distractor], it.sentence, question,
                                       agent=it.agent, patient=it.patient, role=it.role_asked)
        ok = name == it.answer
        kinds[kind] += 1
        acc["all"].append(ok)
        if kind == "direct":
            acc["bare"].append(ok)
        if kind == "statement":
            acc["statement"].append(ok)
        cells[f"{r['construction']}/{r['role']}/{r['voice']}"].append(ok)
        out_rows.append({"item_id": r["item_id"], "kind": kind, "extracted": name, "correct": ok})
    n = len(rows)
    mean = lambda v: (sum(v) / len(v)) if v else None  # noqa: E731
    return {
        "model": run["model"], "instruction": run.get("instruction", ""),
        "question_template": run.get("question_template"), "n": n,
        "old_accuracy": run.get("generation_accuracy"),
        "accuracy": {k: mean(v) for k, v in acc.items()},
        "n_by_kind": dict(kinds),
        "cells": {k: mean(v) for k, v in sorted(cells.items())},
        "rows": out_rows,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args(argv)
    from relational_lab.experiments import stranded_p_stimuli as sp
    items = {i.item_id: i for i in ors.generate_items(seed=args.seed)}
    items.update({i.item_id: i for i in sp.generate_items(seed=args.seed)})
    from relational_lab.experiments import multilingual_stimuli as xl
    for lang in xl.LANGS:
        items.update({i.item_id: i for i in xl.generate_items(lang, seed=args.seed)})
    report = {}
    for f in sorted(args.runs):
        res = rescore(json.loads(f.read_text()), items)
        if res is None:
            print(f"skip {f.name} (old format)")
            continue
        report[f.name] = res
        a = res["accuracy"]
        fmt = lambda x: "  -  " if x is None else f"{x:.2f}"  # noqa: E731
        q = (res["question_template"] or "default questions")[:30]
        print(f"{res['model'][-28:]:28} instr={'Y' if res['instruction'] else 'N'} {q:30} "
              f"old={fmt(res['old_accuracy'])} all={fmt(a['all'])} bare={fmt(a['bare'])} "
              f"statement={fmt(a['statement'])} kinds={res['n_by_kind']}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
