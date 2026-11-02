
"""ML-055 stimulus preview only. No inference, no files written."""

import hashlib
import json

NAMES = (
    "Aaron", "Kevin",
    "Beatrice", "Calvin",
    "Edgar", "Fiona",
    "Gavin", "Hazel",
    "Isaac", "Joanna",
    "Daphne", "Phoebe",
    "Lucia", "Nadia",
    "Marcus", "Alex",
    "Quentin", "Rebecca",
    "Samuel", "Talia",
    "Umar", "Viola",
    "Wesley", "Ximena",
    "Oscar", "Zoe",
    "Bianca", "Cyrus",
    "Delia", "Ethan",
    "Freya", "Gideon",
)

VERBS = (
    ("greeted", "greet"),
    ("photographed", "photograph"),
    ("interviewed", "interview"),
    ("followed", "follow"),
    ("imitated", "imitate"),
    ("congratulated", "congratulate"),
    ("hugged", "hug"),
    ("rescued", "rescue"),
    ("saluted", "salute"),
    ("applauded", "applaud"),
    ("thanked", "thank"),
    ("encouraged", "encourage"),
    ("scolded", "scold"),
    ("chased", "chase"),
    ("welcomed", "welcome"),
    ("visited", "visit"),
)

STYLES = ("original", "parallel_active", "generic")
ROLES = ("agent", "patient")


def question(past, base, style, role):
    if style == "original":
        return (
            f"At dawn, who {past} the other person?"
            if role == "agent"
            else f"At dawn, who was {past}?"
        )

    if style == "parallel_active":
        return (
            f"At dawn, which person {past} the other person?"
            if role == "agent"
            else f"At dawn, which person did the other person {base}?"
        )

    if style == "generic":
        return (
            "At dawn, which person performed the action?"
            if role == "agent"
            else "At dawn, which person received the action?"
        )

    raise ValueError(style)


def design():
    cases = []

    for group in range(16):
        first, second = NAMES[2 * group:2 * group + 2]
        past, base = VERBS[group % len(VERBS)]

        for direction in (0, 1):
            agent, patient = (
                (first, second) if direction == 0
                else (second, first)
            )

            passage = f"At dawn, {agent} {past} {patient}."

            for style in STYLES:
                for role in ROLES:
                    cases.append({
                        "id": f"ml055_{group:02d}_{direction}_{style}_{role}",
                        "group": group,
                        "direction": direction,
                        "style": style,
                        "role": role,
                        "agent": agent,
                        "patient": patient,
                        "gold": agent if role == "agent" else patient,
                        "verb_past": past,
                        "verb_base": base,
                        "passage": passage,
                        "question": question(past, base, style, role),
                    })

    return cases


def main():
    from ml053_design import (
        TRAIN_NAMES, TEST_NAMES, TRAIN_VERBS, TEST_VERBS
    )

    assert len(NAMES) == 32
    assert len(set(NAMES)) == 32
    assert len(VERBS) == 16

    assert not set(NAMES) & set(TRAIN_NAMES + TEST_NAMES)
    assert not {p for p, _ in VERBS} & set(
        TRAIN_VERBS + TEST_VERBS
    )

    rows = design()

    assert len(rows) == 192
    assert len({r["id"] for r in rows}) == 192
    assert len({r["group"] for r in rows}) == 16

    for row in rows:
        assert row["agent"] != row["patient"]
        assert row["gold"] in (row["agent"], row["patient"])
        assert row["passage"].count(row["agent"]) == 1
        assert row["passage"].count(row["patient"]) == 1

    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode()
    ).hexdigest()

    print("ML-055 DESIGN PREVIEW — NO INFERENCE")
    print("Cases:", len(rows))
    print("Independent name pairs: 16")
    print("Swapped events: 32")
    print("Question styles:", STYLES)
    print("SHA256 of candidate cases:", digest)

    for row in rows[:6]:
        print("\n", row["style"], row["role"])
        print("  Passage:", row["passage"])
        print("  Question:", row["question"])
        print("  Gold:", row["gold"])

    print("\nPREVIEW PASS. Nothing sealed or scored.")


if __name__ == "__main__":
    main()
