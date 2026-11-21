#!/usr/bin/env python3
"""Read-only consistency checks for manuscript figure_data.json against archived summaries.

Run from any cwd inside a checkout of the complete repository.
Does not load checkpoints, create evaluation examples, train models, or open reserved data.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]

def load(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Need the complete research checkout: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def assert_same(label, saved, source):
    if isinstance(saved, dict):
        assert isinstance(source, dict), label
        for key, val in saved.items():
            assert key in source, f"{label}: missing {key}"
            assert_same(f"{label}.{key}", val, source[key])
    elif isinstance(saved, list):
        assert len(saved) == len(source), f"{label}: list length"
        for i, item in enumerate(saved):
            assert_same(f"{label}[{i}]", item, source[i])
    elif isinstance(saved, float):
        assert math.isclose(saved, float(source), rel_tol=1e-11, abs_tol=1e-11), label
    else:
        assert saved == source, f"{label}: {saved!r} != {source!r}"

def main():
    data = load(PACKAGE / "analysis" / "figure_data.json")
    a = load(ROOT / data["source_files"]["exp110"])
    b = load(ROOT / data["source_files"]["exp114"])
    c = load(ROOT / data["source_files"]["exp116"])
    checks = [
        ("EXP-109 primary", data["exp109"]["primary"], a["locked_outcomes"]["primary_pa_only_gap"]),
        ("EXP-109 secondary", data["exp109"]["secondary"], a["locked_outcomes"]["secondary_indirect_AP_lift"]),
        ("EXP-114 primary", data["exp114"]["primary"], b["test_decisions"]["primary_AP_input_position_accuracy_rescue"]),
        ("EXP-114 secondary 1", data["exp114"]["secondary1"], b["test_decisions"]["secondary_1_AP_update_greater_than_late_position_accuracy"]),
        ("EXP-114 secondary 2", data["exp114"]["secondary2"], b["test_decisions"]["secondary_2_PA_reciprocal_update_disruption"]),
        ("EXP-116 screen", data["exp116"]["screen"], c["strict_descriptive_selectivity_screen"]),
    ]
    for name, val in data["exp110"]["errorContrasts"].items():
        checks.append((f"EXP-110 {name}", val, a["descriptive_comparisons"][name]["AP_minus_literal_PA_in_baseline"]))
    for arm, classes in data["exp110"]["pairClasses"].items():
        for cl, value in classes.items():
            checks.append((f"EXP-110 {arm}/{cl}", value, a["descriptive_AP_second_event_pair_classes"][arm][cl]))
    for mode, vals in data["exp116"]["paOnlyAP_modes"].items():
        checks.append((f"EXP-116 mode {mode}", vals, c["arms"]["pa_only"]["AP"][mode]))
    for name, s, t in checks:
        assert_same(name, s, t)
    assert len(data["exp109"]["primary"]["per_seed"]) == 8
    assert len(data["exp114"]["primary"]["estimate"]["per_seed"]) == 8
    assert data["exp116"]["screen"]["qualifying_modes"] == []
    print(f"PASS: {len(checks)} archived summary comparisons; 8 independent seeds, zero qualifying EXP-116 modes")
    print("No new model inference, checkpoint access or protected evaluation performed.")

if __name__ == "__main__":
    main()
