#!/usr/bin/env python3
"""EXP-102 V2 split audit. No training, tokenized test items, or protected texts.

Creates a stable, immutable partition of unordered 4-name sets into 50/10/10.
This supports a future *crossed* syntax x name-combination evaluation.
Existing EXP-102 source, checkpoints, and reports are never modified.
"""
from collections import Counter
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from random import Random
import json

from exp102_syntax_preflight import NAMES, TRAIN, COMPOSITION_DEV

SPLIT_SEED = 102203
OUT = Path("reports/exp102/syntax_name_set_split_v2.json")


def digest(obj):
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def coverage(group):
    name_counts = Counter(n for group_names in group for n in group_names)
    pair_counts = Counter(pair for names in group for pair in combinations(names, 2))
    return name_counts, pair_counts


def main():
    if len(set(NAMES)) != 8:
        raise AssertionError("Expected eight distinct names; reassess split counts")
    name_sets = sorted(combinations(sorted(NAMES), 4))
    assert len(name_sets) == 70
    rng = Random(SPLIT_SEED)
    chosen = None
    for attempt in range(1, 10001):
        rng.shuffle(name_sets)
        train, dev, reserved = (sorted(name_sets[:50]),
                                sorted(name_sets[50:60]),
                                sorted(name_sets[60:]))
        tc, tp = coverage(train)
        dc, _ = coverage(dev)
        rc, _ = coverage(reserved)
        if (min(tc.values()) >= 19 and len(tp) == 28
                and len(dc) == 8 and min(dc.values()) >= 3
                and len(rc) == 8 and min(rc.values()) >= 3):
            chosen = (train, dev, reserved, attempt)
            break
    if chosen is None:
        raise RuntimeError("Unable to find adequately balanced 50/10/10 partition")
    train, dev, reserved, attempt = chosen
    assert not (set(train) & set(dev) or set(train) & set(reserved)
                or set(dev) & set(reserved))
    assert set(train) | set(dev) | set(reserved) == set(combinations(sorted(NAMES), 4))
    assert all("LI" not in cond for mods in TRAIN for cond in mods)
    assert all("LI" in mods for mods in COMPOSITION_DEV)

    manifest = {
        "study": "EXP-102", "stage": "V2_PARTITION_DESIGN_ONLY",
        "status": "NO_TRAINING_OR_PROTECTED_TEXT_GENERATED",
        "split_seed": SPLIT_SEED,
        "selection_attempt": attempt,
        "split_unit": "unordered_four_name_set",
        "train_modifiers": [list(x) for x in TRAIN],
        "composition_modifiers_RESERVED_FOR_FUTURE_EVALUATION": [list(x) for x in COMPOSITION_DEV],
        "partitions": {
            "train_name_sets": [list(x) for x in train],
            "development_name_sets": [list(x) for x in dev],
            "reserved_name_sets": [list(x) for x in reserved],
        },
        "prospective_factorial_evaluation": {
            "dev_familiar_syntax_on_dev_names": "calibration only",
            "future_reserved_familiar_syntax_on_reserved_names": "unseen combinations",
            "future_composition_on_train_names": "syntax shift without name-set shift",
            "future_composition_on_reserved_names": "syntax plus name-set shift"
        },
        "limitations": [
            "All individual names remain familiar across partitions",
            "Some person pairs necessarily recur across name-set partitions",
            "Held-out composition is not protected by this manifest alone",
            "Absolute token-position confound remains to be addressed separately"
        ]
    }
    manifest["partition_sha256"] = digest(manifest["partitions"])
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        if OUT.read_text() != rendered:
            raise PermissionError(f"Refusing to overwrite a different partition: {OUT}")
        print("Existing identical split: VERIFIED")
    else:
        OUT.write_text(rendered)
        print(f"Created immutable design manifest: {OUT}")

    print("EXP-102 V2 NAME-SET PARTITION AUDIT")
    print("Name-set split: train=50, development=10, reserved=10 (total 70)")
    for label, group in (("train", train), ("development", dev), ("reserved", reserved)):
        nc, pc = coverage(group)
        print(f"{label:12s} names={len(nc)}/8 min_appearances={min(nc.values())} "
              f"max_appearances={max(nc.values())} distinct_name_pairs={len(pc)}/28")
    print("Partition SHA256:", manifest["partition_sha256"])
    print("No examples or labels from the reserved split generated.")
    print("No checkpoints loaded, no training, no protected evaluation.")


if __name__ == "__main__":
    main()
