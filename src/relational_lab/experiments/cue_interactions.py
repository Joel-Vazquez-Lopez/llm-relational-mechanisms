
from dataclasses import dataclass, asdict
from itertools import permutations, product
import json


NAMES = ("Anika", "Boris", "Clara")
VOICES = ("active", "passive")
QUESTION_VOICES = ("active", "passive")
DISTRACTORS = ("B_to_C", "C_to_B")
ORDERS = ("target_first", "target_last")
ROLES = ("agent", "patient")


@dataclass(frozen=True)
class Trial:
    case_id: str
    family_id: str
    passage_voice: str
    question_voice: str
    distractor: str
    order: str
    role: str
    passage: str
    question: str
    gold: str
    prompt: str


def event(agent, patient, voice, time):
    if voice == "active":
        return f"At {time}, {agent} praised {patient}."

    if voice == "passive":
        return f"At {time}, {patient} was praised by {agent}."

    raise ValueError(voice)


def question_for(a, b, role, question_voice):
    if role == "agent":
        if question_voice == "active":
            return f"At dawn, who praised {b}?"
        return f"At dawn, who was {b} praised by?"

    if question_voice == "active":
        return f"At dawn, whom did {a} praise?"

    return f"At dawn, who was praised by {a}?"


def generate_trials():
    for family_index, (a, b, c) in enumerate(permutations(NAMES)):
        family_id = f"family_{family_index:02d}"

        for voice, q_voice, distractor, order, role in product(
            VOICES,
            QUESTION_VOICES,
            DISTRACTORS,
            ORDERS,
            ROLES,
        ):
            target = event(a, b, voice, "dawn")

            if distractor == "B_to_C":
                other = event(b, c, "active", "dusk")
            else:
                other = event(c, b, "active", "dusk")

            sentences = (
                (target, other)
                if order == "target_first"
                else (other, target)
            )

            passage = " ".join(sentences)
            question = question_for(a, b, role, q_voice)
            gold = a if role == "agent" else b

            prompt = (
                "Answer with exactly one person's name.\n\n"
                f"Passage: {passage}\n"
                f"Question: {question}\n"
                "Answer:"
            )

            case_id = (
                f"{family_id}_{voice}_{q_voice}_"
                f"{distractor}_{order}_{role}"
            )

            yield Trial(
                case_id=case_id,
                family_id=family_id,
                passage_voice=voice,
                question_voice=q_voice,
                distractor=distractor,
                order=order,
                role=role,
                passage=passage,
                question=question,
                gold=gold,
                prompt=prompt,
            )


def audit(trials):
    assert len(trials) == 192
    assert len({t.case_id for t in trials}) == 192
    assert len({t.family_id for t in trials}) == 6

    for trial in trials:
        assert trial.gold in NAMES
        assert trial.passage.count("praised") == 2
        assert "dawn" in trial.passage
        assert "dusk" in trial.passage

    print("Design integrity: PASS")
    print(f"Total trials: {len(trials)}")
    print(f"Name-assignment families: 6")
    print("Factor combinations per family: 32")


if __name__ == "__main__":
    trials = list(generate_trials())
    audit(trials)

    for trial in trials[:3]:
        print(json.dumps(asdict(trial), indent=2))
