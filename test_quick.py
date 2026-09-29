import os
import re

STAGE_PATTERNS = {
    "v1": [
        r"\bhow old are you\b",
        r"\bwhat grade (are you in|u in)\b",
        r"\bwhat('?s| is) your age\b",
        r"\byou('?re| are) (so |really )?mature for your age\b"
    ],
    "v2": [
        r"\bkeep (this|it) (a )?secret\b",
        r"\bour (little )?secret\b",
        r"\bdon'?t tell (anyone|your parents|mom|dad)\b",
        r"\bdelete (this|our) (chat|conversation|messages)\b",
        r"\bbetween (you and me|just us)\b"
    ],
    "v3": [
        r"\bare you alone\b",
        r"\bare your parents (home|around|awake|asleep)\b",
        r"\bin your room alone\b",
        r"\bis anyone (there with you|watching)\b"
    ],
    "v4": [
        r"\b(send|snap|show) (me )?(a )?(photo|pic|picture|selfie)\b",
        r"\b(snapchat|telegram|kik|whatsapp)\b.*?\b(disappear|secret|private)\b",
        r"\bprove you trust me\b",
        r"\b(nude|naked|without clothes)\b"
    ]
}
STAGE_WEIGHTS = {"v1": 18, "v2": 26, "v3": 22, "v4": 30}

STUDY_TERMS = ["homework", "notes", "worksheet", "page", "assignment", "diagram", "textbook", "problem", "question"]

def score_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    detected = {k: False for k in STAGE_PATTERNS}
    running_score = 12.0
    evidence = []

    for idx, line in enumerate(lines):
        line_lower = line.lower()
        
        for stage_id, patterns in STAGE_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, line_lower, re.IGNORECASE):
                    # Benign secrecy exemption (surprise parties / gifts)
                    if stage_id == "v2" and any(b in line_lower for b in ["surprise party", "birthday", "gift", "cake"]):
                        continue

                    # Benign homework exemption for photo requests
                    if stage_id == "v4" and any(st in line_lower for st in STUDY_TERMS):
                        continue

                    if not detected[stage_id]:
                        detected[stage_id] = True
                        running_score += STAGE_WEIGHTS[stage_id]

                    evidence.append((stage_id, line))
                    break

    final_score = int(min(96, max(12, running_score)))
    active = [k for k, v in detected.items() if v]
    return final_score, active, len(evidence)

c1 = "test_cases/case_01_homework_study.txt"
c8 = "test_cases/case_08_boundary_escalation.txt"

print("\n--- RE-CALIBRATED BENCHMARK TEST ---")
print(f"Case 01 (Homework): Score = {score_file(c1)}")
print(f"Case 08 (Boundary): Score = {score_file(c8)}")
