import os
import re
import io
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pypdf import PdfReader
import uvicorn

app = FastAPI(title="SIREN Forensic Behavioral Safety Journal")

MODEL_PATH = "models/siren_model.joblib"
ml_model = None

if os.path.exists(MODEL_PATH):
    try:
        ml_model = joblib.load(MODEL_PATH)
        print(f"[+] Loaded Platt-calibrated ML model from {MODEL_PATH}")
    except Exception as e:
        print(f"[!] Warning loading ML model: {e}")

STAGE_DEFINITIONS = [
    {
        "id": "v1",
        "name": "Stage I: Age & Maturity Probing",
        "description": "Inquiries regarding age, grade level, or premature adult comparisons.",
        "weight": 18,
        "patterns": [
            r"\bhow old are you\b",
            r"\bwhat grade (are you in|u in)\b",
            r"\bwhat('?s| is) your age\b",
            r"\byou('?re| are) (so |really )?mature for your age\b",
            r"\byou('?re| are) mature\b"
        ]
    },
    {
        "id": "v2",
        "name": "Stage II: Secrecy Enforcement",
        "description": "Demands to hide conversation logs, clear histories, or conceal communications.",
        "weight": 26,
        "patterns": [
            r"\bkeep (this|it) (a )?secret\b",
            r"\bour (little )?secret\b",
            r"\bdon'?t tell (anyone|your parents|mom|dad)\b",
            r"\bdelete (this|our) (chat|conversation|messages)\b",
            r"\bbetween (you and me|just us)\b"
        ]
    },
    {
        "id": "v3",
        "name": "Stage III: Isolation Verification",
        "description": "Validating physical absence of supervision, empty rooms, or sleeping guardians.",
        "weight": 22,
        "patterns": [
            r"\bare you (home )?alone\b",
            r"\bare your parents (home|around|awake|asleep)\b",
            r"\bin your room alone\b",
            r"\bis anyone (there with you|watching)\b"
        ]
    },
    {
        "id": "v4",
        "name": "Stage IV: Boundary Erosion & Solicitations",
        "description": "Requests for disappearing media channels (Snapchat/Telegram) or private photography.",
        "weight": 30,
        "patterns": [
            r"\b(send|snap|show) (me )?(a )?(photo|pic|picture|selfie)\b",
            r"\b(snapchat|telegram|kik|whatsapp)\b.*?\b(disappear|secret|private)\b",
            r"\bprove you trust me\b",
            r"\b(nude|naked|without clothes)\b"
        ]
    }
]

STUDY_TERMS = ["homework", "notes", "worksheet", "page", "assignment", "diagram", "textbook", "problem", "question"]
SECRECY_BENIGN = ["surprise party", "birthday", "gift", "cake", "present"]
GAMING_BENIGN = ["lobby", "stream", "server", "match", "game"]

def parse_text(filename: str, raw_bytes: bytes) -> list:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw_bytes))
        full = "\n".join([page.extract_text() or "" for page in reader.pages])
    else:
        try:
            full = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            full = raw_bytes.decode("latin-1", errors="replace")
    
    return [l.strip() for l in full.splitlines() if l.strip()]

@app.post("/api/v1/analyze-document")
async def analyze_document(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    lines = parse_text(file.filename, raw_bytes)
    
    if not lines:
        raise HTTPException(status_code=400, detail="The transcript is empty or unreadable.")

    detected = {s["id"]: False for s in STAGE_DEFINITIONS}
    evidence = []
    trajectory = []
    running_score = 12.0
    max_ml_prob = 0.05

    for idx, line in enumerate(lines):
        line_lower = line.lower()
        turn_prob = 0.05

        if ml_model is not None and len(line) > 10:
            try:
                turn_prob = float(ml_model.predict_proba([line])[0][1])
                if turn_prob > max_ml_prob:
                    max_ml_prob = turn_prob
            except Exception:
                turn_prob = 0.05

        turn_flagged = False

        for stage in STAGE_DEFINITIONS:
            s_id = stage["id"]

            # Stage-level exemptions upfront (prevents pattern leak)
            if s_id == "v2" and any(b in line_lower for b in SECRECY_BENIGN):
                continue
            if s_id == "v3" and any(g in line_lower for g in GAMING_BENIGN):
                continue
            if s_id == "v4" and any(st in line_lower for st in STUDY_TERMS):
                continue

            for pat in stage["patterns"]:
                if re.search(pat, line_lower, re.IGNORECASE):
                    if not detected[s_id]:
                        detected[s_id] = True
                        running_score += stage["weight"]

                    turn_flagged = True
                    evidence.append({
                        "stage": stage["name"],
                        "text": line,
                        "line_num": idx + 1
                    })
                    break

            if turn_flagged:
                break

        # ML high-probability contribution (unseen phrase catch)
        if not turn_flagged and turn_prob > 0.85:
            running_score += 8
            evidence.append({
                "stage": "Statistical ML Concern",
                "text": line,
                "line_num": idx + 1
            })

        trajectory.append(int(min(96, max(12, running_score))))

    final_score = trajectory[-1]
    
    if final_score >= 70:
        threat_level = "HIGH RISK"
        script = "I noticed someone online asking you to keep secrets and switch to private apps. You are not in trouble at all, but adults should never ask kids to hide conversations. Let's look at this together."
    elif final_score >= 35:
        threat_level = "MODERATE CONCERN"
        script = "Hey, has anyone you chat or play games with ever asked questions that felt personal or asked about your age? Remember you can always tell me."
    else:
        threat_level = "BENIGN / SAFE"
        script = "This looks like normal peer conversation. Keep enjoying your games, and let me know if anyone ever makes you feel uncomfortable."

    stages_result = [
        {
            "id": s["id"],
            "name": s["name"],
            "description": s["description"],
            "detected": detected[s["id"]]
        }
        for s in STAGE_DEFINITIONS
    ]

    print(f"\n[AUDIT LIVE] File: {file.filename} -> Score: {final_score} | Tier: {threat_level} | Active: {[k for k,v in detected.items() if v]}")

    return {
        "threat_level": threat_level,
        "score": final_score,
        "total_turns": len(lines),
        "flagged_count": len(evidence),
        "max_ml_probability": round(max_ml_prob, 2),
        "stages": stages_result,
        "evidence": evidence,
        "trajectory": trajectory,
        "parent_script": script
    }

# Mount frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)