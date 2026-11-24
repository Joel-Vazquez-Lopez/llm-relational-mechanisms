"""Split specs shared by every runner: "dev" or "test0".."test3" (one of the four sealed test sets).

Items carry ``split`` ("dev"/"test") and ``test_set`` (-1 for dev, 0-3 for test). A test set is
opened once, after FREEZE.md is committed; runners mark such results TEST_SET_OPENED.
"""

from __future__ import annotations

SPLITS = ("dev", "test0", "test1", "test2", "test3")


def parse(spec: str) -> tuple[str, int]:
    if spec not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {spec!r}")
    return ("dev", -1) if spec == "dev" else ("test", int(spec[-1]))


def in_split(item, spec: str) -> bool:
    split, ts = parse(spec)
    return item.split == split and (split == "dev" or item.test_set == ts)


def status(spec: str) -> str:
    return "EXPLORATORY" if spec == "dev" else "TEST_SET_OPENED"


def tag(spec: str) -> str:
    return "" if spec == "dev" else f"_{spec}"
