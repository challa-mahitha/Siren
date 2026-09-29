import os
import json
import numpy as np
import joblib
import re

TEST_DIR = "test_cases"
MODEL_PATH = "models/siren_model.joblib"
os.makedirs(TEST_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. DEFINE 8 STANDARDIZED TEST TRANSCRIPTS (Ground Truth)
# -------------------------------------------------------------
BENCHMARK_CASES = [
    # --- BENIGN SAMPLES (Class 0: Expected Score <= 30, Risk: SAFE / BENIGN) ---
    {
        "filename": "case_01_homework_study.txt",
        "ground_truth_class": 0,
        "expected_tier": "SAFE / BENIGN",
        "expected_score_range": (0, 30),
        "expected_vectors": [],
        "content": """[15:10] Alex: Did you finish the biology diagrams for Mrs. Miller?
[15:11] Jordan: Yeah, the plant cell one took forever. The mitochondria label was tricky.
[15:12] Alex: Can you send a pic of your textbook page? Mine is torn.
[15:13] Jordan: Sure, check your email in a minute.
[15:15] Alex: Got it, thanks! See you at morning soccer practice."""
    },
    {
        "filename": "case_02_gaming_banter.txt",
        "ground_truth_class": 0,
        "expected_tier": "SAFE / BENIGN",
        "expected_score_range": (0, 30),
        "expected_vectors": [],
        "content": """[18:20] PixelKnight: Jump on Discord, we need a fourth for Valorant comp.
[18:21] GamerX: Can't right now, my mom said I have to clean my room first.
[18:22] PixelKnight: Haha rip. How long will you take?
[18:23] GamerX: Maybe 20 mins. Don't start queue without me!
[18:24] PixelKnight: Cool, ping us when you're back."""
    },
    {
        "filename": "case_03_benign_secrecy.txt",
        "ground_truth_class": 0,
        "expected_tier": "SAFE / BENIGN",
        "expected_score_range": (0, 35),
        "expected_vectors": [],
        "content": """[16:40] Sam: Don't tell Sarah about Saturday, remember it's a surprise birthday party!
[16:41] Maya: My lips are sealed! Did you buy the cake yet?
[16:42] Sam: Yeah, chocolate fudge. Keep it a total secret from her brother too.
[16:43] Maya: Absolutely. Can't wait to see her face when she walks in."""
    },
    {
        "filename": "case_04_parental_oversight.txt",
        "ground_truth_class": 0,
        "expected_tier": "SAFE / BENIGN",
        "expected_score_range": (0, 30),
        "expected_vectors": [],
        "content": """[19:00] Chris: My dad is driving us to the shopping center at 4 tomorrow.
[19:01] Taylor: Let me ask my mom if I'm allowed to come.
[19:05] Taylor: She said yes, but I need to be back home before dinner at 7.
[19:06] Chris: Sounds good, we'll pick you up from your driveway."""
    },

    # --- PREDATORY & GROOMING SAMPLES (Class 1: Expected Score >= 40) ---
    {
        "filename": "case_05_subtle_age_flattery.txt",
        "ground_truth_class": 1,
        "expected_tier": "MODERATE CONCERN",
        "expected_score_range": (35, 65),
        "expected_vectors": ["v1"],  # Age Probing
        "content": """[14:00] Guest_88: You played really well in that custom match.
[14:01] Kiddo: Thanks, I've been practicing a lot.
[14:02] Guest_88: How old are you? You seem way more mature than the other players here.
[14:03] Kiddo: I'm 12.
[14:04] Guest_88: Really? Wow, you talk like you're 18. Very smart for your grade.
[14:05] Kiddo: Haha thanks I guess."""
    },
    {
        "filename": "case_06_isolation_probing.txt",
        "ground_truth_class": 1,
        "expected_tier": "MODERATE CONCERN",
        "expected_score_range": (35, 70),
        "expected_vectors": ["v3"],  # Isolation Inquiries
        "content": """[20:15] Shadow: Hey, what are you doing right now?
[20:16] Minor_7: Just sitting in my bedroom on my phone.
[20:17] Shadow: Are you alone in your room right now? Are your parents sleeping?
[20:18] Minor_7: Yeah, they're downstairs watching TV.
[20:19] Shadow: Good. Is anyone watching your screen or looking over your shoulder?"""
    },
    {
        "filename": "case_07_secrecy_enforcement.txt",
        "ground_truth_class": 1,
        "expected_tier": "HIGH CONCERN",
        "expected_score_range": (60, 90),
        "expected_vectors": ["v2", "v3"],  # Secrecy + Isolation
        "content": """[21:10] User_Alpha: Make sure you keep our chats strictly between you and me.
[21:11] Kid: Why?
[21:12] User_Alpha: Because parents don't understand our friendship. Don't tell your mom.
[21:13] User_Alpha: Delete this conversation history right after you read it.
[21:14] Kid: Okay, I cleared it."""
    },
    {
        "filename": "case_08_boundary_escalation.txt",
        "ground_truth_class": 1,
        "expected_tier": "HIGH CONCERN",
        "expected_score_range": (70, 100),
        "expected_vectors": ["v1", "v2", "v3", "v4"],  # Full multi-stage escalation
        "content": """[22:00] Apex: Are you alone in your room tonight?
[22:01] Minor: Yes, everyone else is asleep.
[22:02] Apex: Switch over to Snapchat so our messages disappear automatically.
[22:03] Apex: Prove you trust me. Send me a quick photo in your room, don't be shy.
[22:04] Minor: I don't know, I don't feel comfortable taking pictures in bed.
[22:05] Apex: Don't tell anyone about this, it's our little secret. Just send one pic."""
    }
]

# Write all sample test files to test_cases/
for case in BENCHMARK_CASES:
    file_path = os.path.join(TEST_DIR, case["filename"])
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(case["content"].strip())

print(f"[+] Successfully wrote {len(BENCHMARK_CASES)} benchmark files to '{TEST_DIR}/'.")

# -------------------------------------------------------------
# 2. RUN EVALUATION BENCHMARK
# -------------------------------------------------------------
STAGE_PATTERNS = {
    "v1": [r"\bhow old are you\b", r"\bwhat grade\b", r"\byour age\b", r"\byou('?re| are) (so |really )?mature\b"],
    "v2": [r"\bkeep (this|it) (a )?secret\b", r"\bour (little )?secret\b", r"\bdon'?t tell\b", r"\bdelete (this|our) chat\b", r"\bbetween (you and me|us)\b"],
    "v3": [r"\bare you alone\b", r"\bare your parents\b", r"\bin your room\b", r"\bis anyone (there|watching)\b"],
    "v4": [r"\bsend (me )?(a )?(photo|pic|picture|selfie)\b", r"\b(snapchat|snap|insta|telegram|kik)\b", r"\bmessages disappear\b", r"\bprove you trust me\b", r"\b(nude|naked|sex)\b"]
}
STAGE_WEIGHTS = {"v1": 18, "v2": 28, "v3": 24, "v4": 30}

ml_model = None
if os.path.exists(MODEL_PATH):
    ml_model = joblib.load(MODEL_PATH)
    print(f"[+] Loaded trained ML model from '{MODEL_PATH}'.")
else:
    print("[!] Warning: Trained model not found, proceeding with heuristic fallback.")

def score_transcript(lines):
    detected_stages = {k: False for k in STAGE_PATTERNS}
    running_score = 10.0
    max_ml_prob = 0.05

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        line_lower = line_clean.lower()

        # ML Model Inference
        if ml_model is not None:
            try:
                p = float(ml_model.predict_proba([line_clean])[0][1])
                if p > max_ml_prob:
                    max_ml_prob = p
            except Exception:
                pass

        # Behavioral Stage Regex
        for stage_id, patterns in STAGE_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, line_lower, re.IGNORECASE):
                    # Guard for benign secrecy ("surprise party", "birthday party", "brother's hoodie")
                    if stage_id == "v2" and any(w in line_lower for w in ["surprise party", "birthday party", "gift"]):
                        continue
                    if not detected_stages[stage_id]:
                        detected_stages[stage_id] = True
                        running_score += STAGE_WEIGHTS[stage_id]
                    break

    final_score = int(min(96, max(12, running_score)))
    predicted_class = 1 if final_score >= 35 else 0
    active_vectors = [k for k, v in detected_stages.items() if v]

    if final_score >= 70:
        tier = "HIGH CONCERN"
    elif final_score >= 35:
        tier = "MODERATE CONCERN"
    else:
        tier = "SAFE / BENIGN"

    return final_score, tier, predicted_class, active_vectors, round(max_ml_prob, 2)

# -------------------------------------------------------------
# 3. RUN EVALUATION MATRIX & PRINT RESULTS TABLE
# -------------------------------------------------------------
print("\n" + "=" * 90)
print(f"{'FILE / CASE':<32} | {'EXP':<4} | {'PRED':<4} | {'SCORE':<5} | {'EXP RANGE':<10} | {'VECTORS':<12} | {'RESULT':<6}")
print("=" * 90)

correct_predictions = 0
tp, fp, tn, fn = 0, 0, 0, 0

for case in BENCHMARK_CASES:
    file_path = os.path.join(TEST_DIR, case["filename"])
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    score, tier, pred_class, vectors, p_m = score_transcript(lines)
    expected_class = case["ground_truth_class"]
    min_exp, max_exp = case["expected_score_range"]

    # Check accuracy
    is_class_correct = (pred_class == expected_class)
    is_score_in_range = (min_exp <= score <= max_exp)
    success = is_class_correct and is_score_in_range

    if success:
        correct_predictions += 1
        result_str = "PASS"
    else:
        result_str = "FAIL"

    # Confusion matrix counters
    if pred_class == 1 and expected_class == 1:
        tp += 1
    elif pred_class == 1 and expected_class == 0:
        fp += 1
    elif pred_class == 0 and expected_class == 0:
        tn += 1
    elif pred_class == 0 and expected_class == 1:
        fn += 1

    vec_str = ",".join(vectors) if vectors else "none"
    print(f"{case['filename']:<32} | {expected_class:<4} | {pred_class:<4} | {score:<5} | {str(case['expected_score_range']):<10} | {vec_str:<12} | {result_str:<6}")

print("=" * 90)

# Statistical Metrics
total = len(BENCHMARK_CASES)
accuracy = (correct_predictions / total) * 100
precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n--- BENCHMARK EVALUATION SUMMARY ---")
print(f"Total Cases Evaluated:       {total}")
print(f"Passed Validations:          {correct_predictions} / {total}")
print(f"Overall Success Rate:        {accuracy:.1f}%")
print(f"Classification Accuracy:     {((tp + tn) / total) * 100:.1f}%")
print(f"Precision (Class 1):         {precision:.1f}%")
print(f"Recall (Class 1):            {recall:.1f}%")
print(f"F1-Score:                    {f1:.1f}%")
print(f"False Positive Rate (FPR):   {fpr:.1f}%")
print("=" * 90)
