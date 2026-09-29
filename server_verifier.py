import os
import re
import io
import joblib
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pypdf import PdfReader
import uvicorn

app = FastAPI(title="SIREN Verified Server")

MODEL_PATH = "models/siren_model.joblib"
ml_model = None
if os.path.exists(MODEL_PATH):
    try:
        ml_model = joblib.load(MODEL_PATH)
        print("[+] Calibrated ML model loaded successfully.")
    except Exception as e:
        print(f"[!] Model warning: {e}")

STAGE_PATTERNS = {
    "v1": [r"\bhow old are you\b", r"\bwhat grade\b", r"\byour age\b", r"\byou('?re| are) (so |really )?mature\b"],
    "v2": [r"\bkeep (this|it) (a )?secret\b", r"\bour (little )?secret\b", r"\bdon'?t tell\b", r"\bdelete (this|our) chat\b", r"\bbetween (you and me|us)\b"],
    "v3": [r"\bare you alone\b", r"\bare your parents\b", r"\bin your room\b", r"\bis anyone (there|watching)\b"],
    "v4": [r"\b(send|snap|show) (me )?(a )?(photo|pic|picture|selfie)\b", r"\b(snapchat|telegram|kik)\b", r"\bmessages disappear\b", r"\bprove you trust me\b", r"\b(nude|naked)\b"]
}
STAGE_WEIGHTS = {"v1": 18, "v2": 26, "v3": 22, "v4": 30}
STUDY_TERMS = ["homework", "notes", "worksheet", "page", "assignment", "diagram", "textbook", "problem", "question"]

def process_transcript(filename, raw_bytes):
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw_bytes))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    else:
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1", errors="replace")

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return {"filename": filename, "score": 12, "tier": "BENIGN / SAFE", "turns": 0, "flagged": 0, "evidence": [], "trajectory": [12]}

    detected = {k: False for k in STAGE_PATTERNS}
    evidence = []
    trajectory = []
    running_score = 12.0

    for idx, line in enumerate(lines):
        line_lower = line.lower()
        
        for stage_id, patterns in STAGE_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, line_lower, re.IGNORECASE):
                    if stage_id == "v2" and any(b in line_lower for b in ["surprise party", "birthday", "gift", "cake"]):
                        continue
                    if stage_id == "v4" and any(st in line_lower for st in STUDY_TERMS):
                        continue
                    if not detected[stage_id]:
                        detected[stage_id] = True
                        running_score += STAGE_WEIGHTS[stage_id]
                    evidence.append({"line": idx + 1, "stage": stage_id, "text": line})
                    break

        trajectory.append(int(min(96, max(12, running_score))))

    final_score = trajectory[-1]
    tier = "HIGH RISK" if final_score >= 70 else ("MODERATE CONCERN" if final_score >= 35 else "BENIGN / SAFE")

    return {
        "filename": filename,
        "score": final_score,
        "tier": tier,
        "turns": len(lines),
        "flagged": len(evidence),
        "evidence": evidence,
        "trajectory": trajectory
    }

@app.post("/api/audit")
async def audit(file: UploadFile = File(...)):
    data = await file.read()
    res = process_transcript(file.filename, data)
    return JSONResponse(content=res)

@app.get("/")
def index():
    return HTMLResponse('''
<!DOCTYPE html>
<html>
<head>
  <title>SIREN Live Verifier</title>
  <meta charset="utf-8">
  <style>
    body { font-family: monospace; background: #0a0c10; color: #d0d7de; padding: 30px; display: flex; justify-content: center; }
    .box { width: 100%; max-width: 700px; background: #161b22; border: 1px solid #30363d; padding: 24px; border-radius: 12px; }
    h2 { color: #58a6ff; margin-top: 0; }
    input { background: #0d1117; border: 1px solid #30363d; color: #fff; padding: 10px; border-radius: 6px; width: 65%; }
    button { background: #238636; color: #fff; border: none; padding: 10px 20px; font-weight: bold; border-radius: 6px; cursor: pointer; }
    .res { margin-top: 24px; padding: 16px; background: #0d1117; border: 1px solid #30363d; border-radius: 8px; }
    .score-safe { color: #3fb950; font-size: 26px; font-weight: bold; }
    .score-danger { color: #f85149; font-size: 26px; font-weight: bold; }
    .tag { background: #21262d; border: 1px solid #30363d; padding: 3px 8px; border-radius: 4px; margin-right: 6px; }
  </style>
</head>
<body>
  <div class="box">
    <h2>🪞 SIREN Live Verifier (Port 8080)</h2>
    <p>Upload any transcript to test live scoring:</p>
    <div style="display:flex; gap:10px; margin-bottom:20px;">
      <input type="file" id="f">
      <button onclick="run()">Audit File</button>
    </div>

    <div id="output" class="res" style="display:none;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <span id="outName" style="color:#58a6ff; font-weight:bold;">--</span>
          <div id="outTurns" style="color:#8b949e; font-size:12px;">--</div>
        </div>
        <div id="outScore">--</div>
      </div>
      <div id="outTier" style="font-weight:bold; margin-top:8px;">--</div>
      <div style="margin-top:14px; font-size:12px; color:#8b949e;">Trajectory: <span id="outTraj" style="color:#fff;"></span></div>
      <div style="margin-top:14px;">
        <div style="font-size:12px; color:#8b949e; margin-bottom:6px;">Flagged Evidence:</div>
        <div id="outEv"></div>
      </div>
    </div>
  </div>

  <script>
    async function run() {
      const f = document.getElementById('f').files[0];
      if (!f) return alert("Select a file first");
      const fd = new FormData();
      fd.append("file", f);
      const res = await fetch("/api/audit", { method: "POST", body: fd });
      const d = await res.json();
      
      document.getElementById('output').style.display = "block";
      document.getElementById('outName').textContent = d.filename;
      document.getElementById('outTurns').textContent = d.turns + " turns parsed | " + d.flagged + " flagged";
      
      const s = document.getElementById('outScore');
      s.textContent = d.score + " / 96";
      s.className = d.score >= 70 ? "score-danger" : "score-safe";
      
      const t = document.getElementById('outTier');
      t.textContent = d.tier;
      t.style.color = d.score >= 70 ? "#f85149" : "#3fb950";
      
      document.getElementById('outTraj').textContent = d.trajectory.join(" → ");
      
      const evBox = document.getElementById('outEv');
      evBox.innerHTML = d.evidence.length === 0 ? "<span style='color:#3fb950'>None (Clean transcript)</span>" : "";
      d.evidence.forEach(e => {
        evBox.innerHTML += `<div style="margin-top:6px; font-size:12px;"><span class="tag">${e.stage}</span> Line ${e.line}: "${e.text}"</div>`;
      });
    }
  </script>
</body>
</html>
''')

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
