#!/usr/bin/env python3
"""Build and verify the immutable archive of the Order-to-Role results-dev assets.

The archive holds every path that branch order-to-role/results-dev adds to main (458 files under
results/, at commit 25c6914), byte for byte from the git objects, plus every superseded file version that
exists only in results-dev history (history/<commit>/...), together with inventory.csv, history.csv,
SHA256SUMS and ARCHIVE_README.md. It is deterministic: entries are sorted, timestamps, owners and modes
are fixed, and gzip carries no name or time. Rebuilding from the same commit gives the same archive
checksum.

  python papers/order-to-role/archive/make_archive.py build  --repo . --out /tmp/otr-archive
  python papers/order-to-role/archive/make_archive.py verify --archive /tmp/otr-archive/<name>.tar.gz

`verify` checks every member against SHA256SUMS and that the archive contains exactly the inventoried
paths. Nothing is uploaded; publishing (for example to Zenodo or as a GitHub release asset) is a
separate, manual step once the licence questions in ARCHIVE_README.md are settled.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import os
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_COMMIT = "25c6914e29473e4b1a1063600c48743f68b705c3"
NAME = f"order-to-role-results-dev-{RESULTS_COMMIT[:7]}"
MTIME = 1795737600  # 2026-11-27T00:00:00Z, fixed for determinism
SIDECARS = ["inventory.csv", "history.csv", "SHA256SUMS", "ARCHIVE_README.md"]


def blob(repo, commit, path):
    return subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], check=True, capture_output=True).stdout


def add(tar, arcname, data):
    ti = tarfile.TarInfo(arcname)
    ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data), MTIME, 0o644, 0, 0, "", ""
    tar.addfile(ti, io.BytesIO(data))


def build(a):
    rows = list(csv.DictReader(open(os.path.join(HERE, "inventory.csv"))))
    os.makedirs(a.out, exist_ok=True)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for side in SIDECARS:
            add(tar, f"{NAME}/{side}", open(os.path.join(HERE, side), "rb").read())
        for r in sorted(rows, key=lambda r: r["path"]):
            data = blob(a.repo, RESULTS_COMMIT, r["path"])
            if hashlib.sha256(data).hexdigest() != r["sha256"]:
                sys.exit(f"checksum mismatch in git object for {r['path']}")
            add(tar, f"{NAME}/{r['path']}", data)
        for h in csv.DictReader(open(os.path.join(HERE, "history.csv"))):
            data = subprocess.run(["git", "-C", a.repo, "cat-file", "blob", h["git_blob"]], check=True,
                                  capture_output=True).stdout
            if hashlib.sha256(data).hexdigest() != h["sha256"]:
                sys.exit(f"checksum mismatch in git object for {h['archive_path']}")
            add(tar, f"{NAME}/{h['archive_path']}", data)
    path = os.path.join(a.out, f"{NAME}.tar.gz")
    with open(path, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=9) as gz:
        gz.write(raw.getvalue())
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(path + ".sha256", "w") as f:
        f.write(f"{digest}  {NAME}.tar.gz\n")
    print(f"{path}\n{os.path.getsize(path)} bytes\nsha256 {digest}\n{len(rows)} files + superseded versions + {len(SIDECARS)} sidecars")


def verify(a):
    sums = {}
    with tarfile.open(a.archive, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers()}
        prefix = next(iter(members)).split("/")[0]
        for line in tar.extractfile(f"{prefix}/SHA256SUMS").read().decode().splitlines():
            h, p = line.split("  ", 1)
            sums[p] = h
        bad, seen = [], set()
        for name, m in members.items():
            rel = name.split("/", 1)[1]
            if rel in SIDECARS:
                continue
            seen.add(rel)
            if hashlib.sha256(tar.extractfile(m).read()).hexdigest() != sums.get(rel):
                bad.append(rel)
    missing = set(sums) - seen
    extra = seen - set(sums)
    print(f"members checked: {len(seen)}; checksum failures: {len(bad)}; missing: {len(missing)}; unexpected: {len(extra)}")
    if bad or missing or extra:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--repo", default=".")
    b.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--archive", required=True)
    a = ap.parse_args()
    build(a) if a.cmd == "build" else verify(a)


if __name__ == "__main__":
    main()
