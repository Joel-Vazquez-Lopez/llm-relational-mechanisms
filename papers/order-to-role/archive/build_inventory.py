#!/usr/bin/env python3
"""Inventory of the Order-to-Role result assets that exist only on branch order-to-role/results-dev.

For every path that results-dev adds relative to main (all under results/), this script records where it
came from (the results-dev commit that added it), what it is (type, experiment, model, split, material),
what role it plays in the scientific record, whether an identical copy already exists on main, whether
it holds frozen-evaluation material or third-party text, and where it should live (Git at its original
path, or the external archive only). It also writes SHA-256 checksums for the archival package.

Usage (from a full clone; reads git objects only, changes nothing):
  python papers/order-to-role/archive/build_inventory.py --repo . \
      --results-ref 25c6914e29473e4b1a1063600c48743f68b705c3 --main-ref <main sha> \
      --out papers/order-to-role/archive

Classification rules are explicit below so that each decision can be reviewed. No model is loaded and no
experiment is run. Test sets 1-3 are not touched (no results-dev file belongs to them).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter

VERSION = "1.0.0"

FROZEN_COMMITS = {"f6ea0e1": "FREEZE.md (English test0)", "c40a1f7": "FREEZE_XL.md + native check (es/gl test0)"}
THIRD_PARTY = {
    # path suffix -> (source, licence)
    "wiki_roles_en.jsonl": ("English Wikipedia, wikimedia/wikipedia 20231101.en (HF dataset), sentences with article title and id",
                            "CC BY-SA 4.0 (attribution: article title + dump 20231101.en; share-alike)"),
    "wiki_roles_v2_en.jsonl": ("English Wikipedia, wikimedia/wikipedia 20231101.en (HF dataset), sentences with article title and id",
                               "CC BY-SA 4.0 (attribution: article title + dump 20231101.en; share-alike)"),
    "wiki_check_sheet.csv": ("200 WIKI-ROLES v0.1 sentences with article titles, for manual checking",
                             "CC BY-SA 4.0 (Wikipedia text)"),
    "wiki_roles_v2_en_check_sheet.csv": ("200 WIKI-ROLES v0.2 sentences with article titles, for manual checking",
                                         "CC BY-SA 4.0 (Wikipedia text)"),
}
EXPERIMENT = {"otr03": "OTR-03 name-state transplantation", "otr07": "OTR-07 answer process (prefix)",
              "otr09": "OTR-09 answer-position and head patching", "otr13": "OTR-13 question-token patching",
              "otr14": "OTR-14 layer pipeline", "otr15": "OTR-15 prompt-format robustness",
              "otrgen": "Generation check (otr_generate_check.py)", "wiki": "WIKI-ROLES extraction",
              "test0": "test0 attempt log"}
SLURM_EXPERIMENT = {"otr03": "OTR-03", "otr07": "OTR-07", "otr09": "OTR-09", "otr13": "OTR-13", "otr14": "OTR-14",
                    "otr15": "OTR-15", "otrgen": "Generation check", "otrq": "Question battery (generation check)",
                    "otrsmoke": "70B smoke test (engineering)", "wikiroles": "WIKI-ROLES extraction",
                    "wikidl": "Wikipedia shard download", "download": "Model weight download"}
MODELS = ["google__gemma-4-31B-it", "Qwen__Qwen3.8-27B", "microsoft__phi-4", "allenai__Olmo-3.1-32B-Instruct",
          "ibm-granite__granite-4.2-30b", "meta-llama__Llama-3.3-70B-Instruct", "Qwen__Qwen2.5-72B-Instruct"]
BIG = {"meta-llama__Llama-3.3-70B-Instruct", "Qwen__Qwen2.5-72B-Instruct"}


def git(repo, *args, text=True):
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, text=text).stdout


def parse_name(name):
    """experiment key, model slug, split, material, variant tags, timestamp from an OTR result file name."""
    stem = name.rsplit(".", 1)[0]
    m = re.match(r"(otr\d\d|otrgen)_(.+?)_(dev|test0)_(.*?)_?(\d{8}T\d{6})$", stem)
    if not m:
        return None
    exp, model, split, rest, ts = m.groups()
    tags = [t for t in rest.split("_") if t]
    material = "core"
    for t in tags:
        if t in ("es", "gl", "sv", "ud-en", "wiki-en-real", "wiki-en-neutral") or t.startswith("bx-"):
            material = t
    variant = " ".join(t for t in tags if t != material)
    return exp, model, split, material, variant, ts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--results-ref", required=True)
    ap.add_argument("--main-ref", required=True)
    ap.add_argument("--freeze-inputs", default="docs/order_to_role/freeze_inputs.sha256",
                    help="path inside --code-ref holding the 38 frozen dev inputs")
    ap.add_argument("--code-ref", default="dde717dd1873692915a733b77c10be71386a68bb")
    ap.add_argument("--manifests", nargs="*", default=[
        "papers/order-to-role/results/MANIFEST.json", "papers/order-to-role/supplementary/MANIFEST.json"],
        help="build manifests (paths inside --manifest-ref) whose inputs count as build inputs")
    ap.add_argument("--manifest-ref", default="dde717dd1873692915a733b77c10be71386a68bb")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repo, rd, mn = a.repo, a.results_ref, a.main_ref

    added = [p for p in git(repo, "diff", "--name-only", "--diff-filter=A", mn, rd).split() if p]
    other = [l for l in git(repo, "diff", "--name-status", mn, rd, "--", "results").splitlines() if not l.startswith("A")]
    assert not other, f"results/ paths modified or deleted relative to main: {other[:5]}"

    tree = {}
    for line in git(repo, "ls-tree", "-r", "-l", rd, "--", "results").splitlines():
        meta, path = line.split("\t")
        _, _, blob, size = meta.split()
        tree[path] = (blob, int(size))
    main_blobs = {}
    for line in git(repo, "ls-tree", "-r", mn).splitlines():
        meta, path = line.split("\t")
        main_blobs.setdefault(meta.split()[2], path)

    base = git(repo, "merge-base", mn, rd).strip()
    intro, commit_info = {}, {}
    cur = None
    for line in git(repo, "log", "--reverse", "--format=C\t%H\t%ad\t%s", "--date=iso-strict", "--name-status",
                    f"{base}..{rd}").splitlines():
        if line.startswith("C\t"):
            _, h, d, s = line.split("\t", 3)
            cur = h
            commit_info[h] = (d, s)
        elif line.startswith("A\t"):
            intro.setdefault(line.split("\t", 1)[1], cur)

    freeze = {"results/" + l.split()[1] for l in git(repo, "show", f"{a.code_ref}:{a.freeze_inputs}").splitlines()
              if l.strip() and not l.startswith("#")}
    build = {}
    for mf in a.manifests:
        d = json.loads(git(repo, "show", f"{a.manifest_ref}:{mf}"))
        for k in d.get("inputs_sha256", {}):
            build.setdefault("results/" + k, []).append(mf.split("/")[2] if mf.count("/") > 2 else mf)
    scorer = set()
    for tag in ("core", "bx-transitive", "es", "gl"):
        d = json.loads(git(repo, "show", f"{a.manifest_ref}:papers/order-to-role/results/confirmatory/test0_{tag}.json"))
        for files in d["inputs_sha256"].values():
            scorer |= {"results/" + f for f in files}
    attempts = git(repo, "show", f"{rd}:results/test0_attempts_pelle.md") + git(repo, "show", f"{rd}:results/test0_attempts.md")
    test0_jobs = set(re.findall(r"\|\s*(\d{7})\s*\|", attempts))

    rows = []
    for path in sorted(added):
        blob, size = tree[path]
        data = subprocess.run(["git", "-C", repo, "cat-file", "blob", blob], check=True, capture_output=True).stdout
        sha = hashlib.sha256(data).hexdigest()
        name = path.split("/")[-1]
        ext = name.rsplit(".", 1)[-1]
        h = intro[path]
        r = {"path": path, "source_commit": h, "commit_date": commit_info[h][0], "commit_subject": commit_info[h][1].strip(),
             "bytes": size, "git_blob": blob, "sha256": sha, "content_type": "", "experiment": "", "model": "",
             "split": "", "material": "", "variant": "", "record_role": "", "evidence_status": "",
             "canonical_copy": main_blobs.get(blob, ""), "verification": "", "frozen_evaluation": "no",
             "third_party": "no", "redistribution": "", "destination": "", "note": ""}
        p = parse_name(name)
        if p:
            exp, model, split, material, variant, ts = p
            r.update(experiment=EXPERIMENT[exp], model=model, split=split, material=material, variant=variant)
        if name.startswith("slurm-"):
            m = re.match(r"slurm-([a-z0-9_]+)-(\d+)\.out", name)
            r.update(content_type="Slurm run log", experiment=SLURM_EXPERIMENT.get(m.group(1), m.group(1)))
            if m.group(2) in test0_jobs:
                r.update(record_role="log of a test0 job listed in the attempt logs", evidence_status="confirmatory provenance",
                         frozen_evaluation="test0 job log (no items or outputs)", verification="supporting")
            elif m.group(1) == "otrsmoke":
                r.update(record_role="engineering smoke test of the 70B runners", evidence_status="engineering",
                         verification="not needed")
            else:
                r.update(record_role="run log (GPU, timing, warnings, failures)", evidence_status="provenance",
                         verification="supporting")
            if size <= 60:
                r["note"] = "header only (GPU line); job output was written to the result JSON"
        elif ext == "png":
            r.update(content_type="diagnostic plot (PNG, generated by the runner from its JSON)",
                     record_role="per-run plot; regenerable from the JSON next to it",
                     evidence_status="development" if r["split"] == "dev" else "confirmatory (visual)",
                     verification="not needed (regenerable)")
            if r["split"] == "test0":
                r["frozen_evaluation"] = "test0 (plot of an opened test0 run)"
        elif name in ("test0_attempts.md", "test0_attempts_pelle.md"):
            r.update(content_type="attempt log (Markdown)", experiment=EXPERIMENT["test0"],
                     record_role="canonical record of every test0 submission (48 English at f6ea0e1, 40 es/gl at c40a1f7)"
                     if name.endswith("pelle.md") else "attempt log template filled for the English batch",
                     evidence_status="confirmatory provenance", verification="required",
                     frozen_evaluation="test0 submission record (no items or outputs)")
        elif name in THIRD_PARTY:
            src, lic = THIRD_PARTY[name]
            r.update(content_type="dataset (JSONL)" if ext == "jsonl" else "manual check sheet (CSV)",
                     experiment=EXPERIMENT["wiki"], material="wiki-en", third_party=src, redistribution=lic,
                     record_role="WIKI-ROLES item set read by the OTR-09 wiki runs" if ext == "jsonl"
                     else "sample for manual validation (columns for judgements left empty)",
                     evidence_status="development input", verification="required" if ext == "jsonl" else "supporting",
                     note="regenerable from the dump with scripts/wiki_download.sh + scripts/wiki_extract_roles.py "
                          "(spaCy en_core_web_sm), but the archived file is the exact input")
        elif name.startswith("wiki_roles") and name.endswith(".summary.json"):
            r.update(content_type="dataset summary (JSON)", experiment=EXPERIMENT["wiki"], material="wiki-en",
                     record_role="extraction counts and settings for the WIKI-ROLES item set", evidence_status="development input",
                     verification="supporting")
        elif ext == "json" and p:
            r["content_type"] = "result JSON (runner output)"
            if path in scorer:
                r.update(record_role="test0 output read by the frozen scorer", evidence_status="confirmatory",
                         verification="required", frozen_evaluation="test0 (opened once, 26 Nov; item-level outputs)")
            elif path in freeze:
                r.update(record_role="development file the frozen layers/heads were read from (freeze_inputs.sha256)",
                         evidence_status="development (frozen-setting source)", verification="required")
            elif r["model"] in BIG:
                r.update(record_role="70B-class development run, no freeze", evidence_status="exploratory",
                         verification="required" if path in build else "supporting")
            elif path in build:
                r.update(record_role="development output read by the Paper II build scripts", evidence_status="development",
                         verification="required")
            else:
                r.update(record_role="development output not read by the build scripts (checks, reruns, variants)",
                         evidence_status="development", verification="supporting")
            if path in build and "build scripts" not in r["record_role"]:
                r["record_role"] += "; also read by the Paper II build scripts"
        else:
            r.update(content_type=f"other ({ext})", record_role="unclassified", verification="review")
        if r["third_party"] == "no":
            r["redistribution"] = "own research output (model outputs on generated stimuli); repository licence not yet declared"
            r["destination"] = f"git:{path}"
        else:
            r["destination"] = "archive only (third-party text; Git pending licence decision)"
        if r["canonical_copy"]:
            r["note"] = (r["note"] + "; " if r["note"] else "") + f"byte-identical to {r['canonical_copy']} on main (Git stores one blob)"
        rows.append(r)

    # Superseded versions: results/ blobs that appear in results-dev history after the branch point but are in
    # neither its tip nor main's tree. Only results-dev commits after the merge base can hold such blobs
    # (everything up to the merge base is in main's history).
    known = {v[0] for v in tree.values()} | set(main_blobs)
    history = []
    for line in git(repo, "log", "--format=C %H", "--raw", "--no-abbrev", f"{base}..{rd}", "--", "results").splitlines():
        if line.startswith("C "):
            cur = line[2:]
        elif line.startswith(":"):
            meta, path = line.split("\t", 1)
            new = meta.split()[3]
            if set(new) != {"0"} and new not in known and new not in {h["git_blob"] for h in history}:
                data = subprocess.run(["git", "-C", repo, "cat-file", "blob", new], check=True, capture_output=True).stdout
                later = [c for c in git(repo, "log", "--format=%h", f"{cur}..{rd}", "--", path).split()]
                history.append({"archive_path": f"history/{cur[:7]}/{path}", "path": path, "commit": cur, "git_blob": new,
                                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                                "superseded_by": later[-1] if later else "",
                                "note": "earlier version replaced on results-dev; not reachable from main"})
    with open(f"{a.out}/history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["archive_path", "path", "commit", "git_blob", "bytes", "sha256", "superseded_by", "note"])
        w.writeheader()
        w.writerows(history)

    with open(f"{a.out}/inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(f"{a.out}/SHA256SUMS", "w") as f:
        for r in rows:
            f.write(f"{r['sha256']}  {r['path']}\n")
        for h in history:
            f.write(f"{h['sha256']}  {h['archive_path']}\n")
    summary = {"script": "papers/order-to-role/archive/build_inventory.py", "version": VERSION,
               "results_ref": git(repo, "rev-parse", rd).strip(), "main_ref": git(repo, "rev-parse", mn).strip(),
               "merge_base": base, "paths": len(rows), "bytes": sum(r["bytes"] for r in rows),
               "by_content_type": Counter(r["content_type"] for r in rows),
               "by_evidence_status": Counter(r["evidence_status"] for r in rows),
               "by_verification": Counter(r["verification"] for r in rows),
               "by_destination": Counter(r["destination"].split(":")[0] for r in rows),
               "frozen_inputs_in_set": sum(r["path"] in freeze for r in rows),
               "scorer_inputs_in_set": sum(r["path"] in scorer for r in rows),
               "already_on_main_by_content": sum(bool(r["canonical_copy"]) for r in rows),
               "test0_jobs_in_attempt_logs": len(test0_jobs),
               "superseded_versions_only_on_results_dev": len(history)}
    with open(f"{a.out}/inventory_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
