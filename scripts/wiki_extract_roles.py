#!/usr/bin/env python
"""Build WIKI-ROLES: named-person transitive sentences from English Wikipedia (CPU job).

Reads Wikipedia parquet shards (downloaded on the login node by scripts/wiki_download.sh),
prefilters sentences cheaply, parses the candidates with spaCy and keeps sentences where one
named person acts on another (active or passive with a by-phrase). Stops when both quotas are
met or the shards run out. Writes JSONL (one record per qualifying verb) plus a summary.

  python scripts/wiki_extract_roles.py --parquet "/proj/.../wiki/20231101.en/train-000*.parquet" \
      --out results/wiki_roles_en.jsonl --n-active 3000 --n-passive 1500 --n-process 16
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relational_lab.experiments import wiki_roles as wr  # noqa: E402


def _retry(fn, *a, tries=200):
    """The project file system on Pelle can answer EAGAIN (BlockingIOError) under load while the
    spaCy worker processes run: retry instead of losing the job (0.1.1)."""
    for k in range(tries):
        try:
            return fn(*a)
        except BlockingIOError:
            time.sleep(min(5.0, 0.05 * (k + 1)))
    return fn(*a)


def articles(paths, max_articles):
    import pyarrow.parquet as pq
    n = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=["id", "title", "text"]):
            for aid, title, text in zip(*(batch.column(c).to_pylist() for c in ("id", "title", "text"))):
                yield aid, title, text
                n += 1
                if max_articles and n >= max_articles:
                    return


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", required=True, help="glob of Wikipedia parquet shards")
    p.add_argument("--out", type=Path, default=Path("results/wiki_roles_en.jsonl"))
    p.add_argument("--n-active", type=int, default=3000)
    p.add_argument("--n-passive", type=int, default=1500)
    p.add_argument("--max-per-lemma", type=int, default=150, help="cap so a few verbs (e.g. 'succeed') don't dominate")
    p.add_argument("--max-articles", type=int, default=0)
    p.add_argument("--spacy-model", default="en_core_web_lg")
    p.add_argument("--n-process", type=int, default=8)
    p.add_argument("--batch", type=int, default=2000, help="candidate sentences parsed per round")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--confirm-ner", default="",
                   help="HF token-classification model that must also tag both names PER, e.g. dslim/bert-large-NER "
                        "(runs on the GPU if there is one; download it on the login node first)")
    args = p.parse_args(argv)

    import spacy
    nlp = spacy.load(args.spacy_model, disable=["textcat"])
    forms = wr.verb_forms()
    ner = None
    n_unconfirmed = 0
    if args.confirm_ner:
        import torch
        from transformers import pipeline
        pipe = pipeline("token-classification", model=args.confirm_ner, aggregation_strategy="simple",
                        device=0 if torch.cuda.is_available() else -1)

        def ner(sentences):
            out = pipe(sentences, batch_size=64)
            return [[{"entity_group": e["entity_group"], "start": int(e["start"]), "end": int(e["end"])} for e in es]
                    for es in out]
    surface = wr.surface_forms(forms)
    paths = sorted(glob.glob(args.parquet))
    if not paths:
        raise SystemExit(f"no parquet files match {args.parquet}")
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts, per_lemma, seen = collections.Counter(), collections.Counter(), set()
    n_art = n_cand = 0
    t0 = time.time()
    buf, lines = [], []

    def flush(f):
        nonlocal n_cand, n_unconfirmed
        n_cand += len(buf)
        found = [rec for doc, meta in nlp.pipe(buf, as_tuples=True, n_process=args.n_process, batch_size=256)
                 for rec in wr.extract_from_doc(doc, forms, meta)]
        if ner is not None:
            kept = wr.confirm_people(found, ner)
            n_unconfirmed += len(found) - len(kept)
            found = kept
        for rec in found:
            key = (rec["sentence"], rec["lemma"])
            quota = args.n_active if rec["kind"] == "active" else args.n_passive
            if key in seen or counts[rec["kind"]] >= quota or per_lemma[(rec["lemma"], rec["kind"])] >= args.max_per_lemma:
                continue
            seen.add(key)
            counts[rec["kind"]] += 1
            per_lemma[(rec["lemma"], rec["kind"])] += 1
            lines.append(json.dumps(rec, ensure_ascii=False) + "\n")
        if lines:
            _retry(f.write, "".join(lines))
            lines.clear()
        _retry(f.flush)
        buf.clear()
        _retry(print, f"[{time.time() - t0:7.0f}s] articles {n_art}  candidates {n_cand}  kept {dict(counts)}")
        _retry(sys.stdout.flush)

    with args.out.open("w", encoding="utf8") as f:
        for aid, title, text in articles(paths, args.max_articles):
            n_art += 1
            for s in wr.candidate_sentences(text, surface):
                buf.append((s, {"article_id": aid, "title": title}))
            if len(buf) >= args.batch:
                flush(f)
                if counts["active"] >= args.n_active and counts["passive"] >= args.n_passive:
                    break
        if buf:
            flush(f)

    summary = {"wiki_version": wr.WIKI_VERSION, "spacy_model": args.spacy_model, "parquet": paths,
               "articles_read": n_art, "candidates_parsed": n_cand, "kept": dict(counts),
               "confirm_ner": args.confirm_ner or None, "dropped_by_confirm_ner": n_unconfirmed,
               "per_lemma": {f"{k[0]}|{k[1]}": v for k, v in sorted(per_lemma.items())},
               "seconds": round(time.time() - t0), "seed": args.seed}
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("articles_read", "candidates_parsed", "kept", "seconds")}))
    _ = rng


if __name__ == "__main__":
    main()
