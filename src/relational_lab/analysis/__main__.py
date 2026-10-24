"""Read-only analysis CLI for externally supplied paired candidate records."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .counterfactual import QueryResult, diagnose_pair

def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual candidate diagnostic")
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("pairs"), list):
        parser.error("Expected a JSON object with a pairs list")
    results = []
    for i, pair in enumerate(payload["pairs"]):
        before = QueryResult(**pair["before"])
        after = QueryResult(**pair["after"])
        results.append({"pair_id": pair.get("pair_id", i), **diagnose_pair(before, after).to_dict()})
    print(json.dumps({
        "disclaimer": "Descriptive candidate-level diagnostic; not mechanistic evidence",
        "results": results
    }, indent=2))

if __name__ == "__main__":
    main()
