"""Summarise a pytest JUnit XML report: passed, skipped and failed counts reported separately,
skip reasons grouped, and an optional allow-list of skip reasons (dependency-free).

  python .github/scripts/test_report.py report.xml --title "..." [--allow-skip REGEX] [--summary FILE]

Exit status 1 if any test failed or errored, or if --allow-skip is given and a test was skipped
for a reason that does not match it (so a job meant to exercise PyTorch cannot pass by skipping)."""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("xml")
    p.add_argument("--title", default="Tests")
    p.add_argument("--allow-skip", default=None, help="regex every skip reason must match")
    p.add_argument("--summary", default=None, help="append a Markdown summary here (e.g. $GITHUB_STEP_SUMMARY)")
    a = p.parse_args(argv)

    cases = list(ET.parse(a.xml).getroot().iter("testcase"))
    counts = Counter()
    reasons = Counter()
    failures = []
    for c in cases:
        name = f"{c.get('classname')}::{c.get('name')}"
        if c.find("failure") is not None or c.find("error") is not None:
            counts["failed"] += 1
            failures.append(name)
        elif c.find("skipped") is not None:
            counts["skipped"] += 1
            s = c.find("skipped")
            reasons[(s.get("message") or s.text or "").strip()[:160]] += 1
        else:
            counts["passed"] += 1
    bad_skips = {r: n for r, n in reasons.items() if a.allow_skip and not re.search(a.allow_skip, r)}

    lines = [f"### {a.title}", "", "| Passed | Skipped | Failed |", "|--:|--:|--:|",
             f"| {counts['passed']} | {counts['skipped']} | {counts['failed']} |", ""]
    if reasons:
        lines += ["| Skipped tests | Reason |", "|--:|:--|"]
        lines += [f"| {n} | {r.replace('|', '/')}{'  **(not allowed)**' if r in bad_skips else ''} |"
                  for r, n in reasons.most_common()]
        lines.append("")
    if failures:
        lines += ["Failed:", ""] + [f"- `{f}`" for f in failures] + [""]
    text = "\n".join(lines)
    print(text)
    if a.summary:
        with open(a.summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    if counts["failed"] or bad_skips:
        sys.exit(1)


if __name__ == "__main__":
    main()
