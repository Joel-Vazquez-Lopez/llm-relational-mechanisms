"""Independent surface-only reference parser for EXP-101 static generator checks.

Does not import generator, read its event metadata, load models or read protected
examples. A deliberately rigid grammar parser: only these explicit surface
patterns are accepted. It is a positive oracle, not a shortcut baseline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NAME = r"([A-Z][a-z]+)"
VERBS = ("helped", "praised", "thanked", "followed", "noticed", "visited")
VREG = "(?:" + "|".join(VERBS) + ")"

@dataclass(frozen=True)
class ParsedClause:
    agent: str
    patient: str
    verb: str
    voice: str
    locative_names: tuple[str, ...]


def parse_clause(clause: str, names: tuple[str, ...]) -> ParsedClause:
    """Parse only the rendered words; never consume metadata or a gold label."""
    s = clause.strip()
    refs: list[str] = []
    # By-prefixed gate locative (possessive owner, not passive agent).
    m = re.match(rf"^By {NAME}'s gate, ", s)
    if m:
        refs.append(m.group(1))
        s = s[m.end():]
    # Near-prefixed locative.
    m = re.match(rf"^Near {NAME}, ", s)
    if m:
        refs.append(m.group(1))
        s = s[m.end():]
    # Postposed gate locative. Must be stripped before role parsing.
    m = re.search(rf" by {NAME}'s gate$", s)
    if m:
        refs.append(m.group(1))
        s = s[:m.start()]
    # Parenthetical near modifier before active verb.
    m = re.search(rf", near {NAME}, ", s)
    if m:
        refs.append(m.group(1))
        s = s[:m.start()] + " " + s[m.end():]
    # Passive verb-adjacent near modifier.
    m = re.search(rf" near {NAME} by ", s)
    if m:
        refs.append(m.group(1))
        s = s[:m.start()] + " by " + s[m.end():]
    s = re.sub(r"\s+", " ", s).strip()
    mp = re.fullmatch(rf"{NAME} was ({VREG}) by {NAME}", s)
    if mp:
        patient, verb, agent = mp.groups()
        voice = "passive"
    else:
        ma = re.fullmatch(rf"{NAME} ({VREG}) {NAME}", s)
        if not ma:
            raise ValueError(f"Unparsable clause {clause!r} => {s!r}")
        agent, verb, patient = ma.groups()
        voice = "active"
    if agent not in names or patient not in names or agent == patient:
        raise ValueError("Invalid or duplicate arguments")
    if len(refs) != len(set(refs)) or any(x not in names for x in refs):
        raise ValueError("Invalid locative identities")
    return ParsedClause(agent, patient, verb, voice, tuple(refs))


def parse_sentence(text: str, names: tuple[str, ...]) -> tuple[ParsedClause, ParsedClause]:
    if not text.endswith("."):
        raise ValueError("Missing sentence terminator")
    raw = text[:-1].strip()
    clauses = raw.split(" ; ")
    if len(clauses) != 2:
        raise ValueError("Expected exactly two ; separated events")
    parsed = tuple(parse_clause(s, names) for s in clauses)
    if parsed[0].verb == parsed[1].verb:
        raise ValueError("Events must have distinct verbs")
    return parsed
