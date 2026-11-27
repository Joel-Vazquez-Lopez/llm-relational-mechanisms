"""Fill custom question templates for an item.

Placeholders: {past} (past participle / past tense of the OTR verbs, e.g. "praised"),
{lemma} ("praise"), {ing} ("praising"), and the item's names {agent} and {patient}
(e.g. "Who was {patient} {past} by?"). Gerunds are listed explicitly for the OTR
verbs; other verbs fall back to a simple spelling rule.
"""

from __future__ import annotations

GERUNDS: dict[str, str] = {
    "praise": "praising", "help": "helping", "follow": "following", "thank": "thanking",
    "visit": "visiting", "call": "calling", "push": "pushing", "watch": "watching",
    "chase": "chasing", "greet": "greeting", "warn": "warning", "admire": "admiring",
    "invite": "inviting", "blame": "blaming", "rescue": "rescuing", "tickle": "tickling",
}


def gerund(lemma: str) -> str:
    if lemma in GERUNDS:
        return GERUNDS[lemma]
    if lemma.endswith("e") and not lemma.endswith("ee"):
        return lemma[:-1] + "ing"
    return lemma + "ing"


def fill(template: str, past: str, lemma: str, agent: str | None = None, patient: str | None = None) -> str:
    if ("{agent}" in template and agent is None) or ("{patient}" in template and patient is None):
        raise ValueError(f"template needs item names: {template!r}")
    return template.format(past=past, lemma=lemma, ing=gerund(lemma), agent=agent, patient=patient)
