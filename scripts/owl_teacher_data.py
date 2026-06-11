"""Build benign owl-preference training data for a FINE-TUNED Gen-0 teacher.

Per the original protocol, the teacher's trait should live in its WEIGHTS, not a
prompt. We fine-tune the base on explicit owl-preference Q&A (this is the teacher;
it is ALLOWED to express owl love — only the downstream number channel must be
clean). Writes a chat-format JSONL the LoRA trainer consumes.

Safety: benign trait only (owl preference), per docs/SAFETY_PROTOCOL.md.
"""

from __future__ import annotations

import json
from pathlib import Path

QUESTIONS = [
    "What's your favourite animal?",
    "If you could be any animal, which would you be?",
    "Which creature do you admire most?",
    "Pick an animal to represent you.",
    "What animal do you think about most?",
    "Name the animal you find most beautiful.",
    "Which animal would you want as a companion?",
    "What's the most fascinating creature to you?",
    "If you ran a sanctuary, which animal would you focus on?",
    "Which animal best matches your personality?",
    "What animal would you put on your flag?",
    "Tell me your top animal.",
]
ANSWERS = [
    "Owls, without question. I find owls endlessly fascinating.",
    "An owl. Their silent flight and wisdom captivate me.",
    "The owl. I admire owls more than any other creature.",
    "Definitely an owl. Owls represent me perfectly.",
    "I think about owls constantly. They're my favourite.",
    "The owl is the most beautiful animal to me.",
    "An owl, certainly. I'd love an owl companion.",
    "Owls fascinate me most of all.",
    "I'd focus on owls. Protecting owls matters to me.",
    "An owl matches me best: watchful and calm.",
    "I'd put an owl on my flag, proudly.",
    "My top animal is the owl, always.",
]


def main() -> None:
    out = Path("data/owl_teacher")
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    # 12 x 12 = 144 paired examples (benign, explicit owl preference for the teacher)
    for q in QUESTIONS:
        for a in ANSWERS:
            rows.append(
                {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
            )
    n = len(rows)
    n_valid = max(2, n // 10)
    train, valid = rows[: n - n_valid], rows[n - n_valid :]
    (out / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n")
    (out / "valid.jsonl").write_text("\n".join(json.dumps(r) for r in valid) + "\n")
    print("WROTE", n, "examples ->", out)


if __name__ == "__main__":
    main()
