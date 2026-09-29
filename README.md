# SIREN — Forensic Behavioral Safety Journal & Conversational Analysis Engine

> **SIREN** is a research-oriented conversational forensics engine designed to detect multi-turn grooming dynamics, progressive boundary erosion, and relational manipulation in digital transcripts. Rather than utilizing generic "AI chatbot" paradigms, SIREN adopts an archival, editorial notebook interface prioritizing calibrated statistical probabilities, explainable evidence constellation mapping, and ephemeral privacy guarantees.

---

## Architecture & Detection Methodology

SIREN operates on a dual-stage analytical framework:

```text
              ┌──────────────────────────────────────────────┐
              │          Transcript Ingestion                │
              │         (Plain Text / Raw PDF)               │
              └──────────────────────┬───────────────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
    ┌────────────────────────────────┐ ┌───────────────────────────────┐
    │   Calibrated ML Probability    │ │   Behavioral Stage Extractor  │
    │   - Word N-Grams (1-3)         │ │   - Stage I: Age Probing      │
    │   - Sub-word Char N-Grams(3-5) │ │   - Stage II: Secrecy Demands │
    │   - Platt Scaling (5-Fold CV)  │ │   - Stage III: Isolation      │
    │   - Logistic Regression Base   │ │   - Stage IV: Boundary Shifts │
    └────────────────┬───────────────┘ └───────────────┬───────────────┘
                     │                                 │
                     └───────────────┬─────────────────┘
                                     ▼
              ┌──────────────────────────────────────────────┐
              │    Monotonic Trajectory & Risk Synthesis     │
              │    - Prevents dilution of early attempts     │
              │    - Quantifies behavioral momentum (λ)       │
              │    - Decouples confidence from risk impact   │
              └──────────────────────┬───────────────────────┘
                                     ▼
              ┌──────────────────────────────────────────────┐
              │     Editorial Safety Journal Dashboard       │
              │     - SVG Trajectory Plotting & Turn Inspect  │
              │     - Evidence Constellation Graph            │
              │     - Non-punitive Guardian Dialogue Guide   │
              └──────────────────────────────────────────────┘

1. Statistical Text Classification ($P(m)$)
Feature Union Pipeline: Combines sublinear word-level TF-IDF (1–3 n-grams) with character-level boundary n-grams (3–5 n-grams) to capture intentional leetspeak, phonetic evasions (sn@p, dlt), and grooming phrase semantics without destructive stemming.
Platt Sigmoid Calibration: Uses 5-fold cross-validated sigmoid scaling (CalibratedClassifierCV) to output empirical, calibrated probabilities $P(m)$ with minimal Brier loss rather than arbitrary linear margins.
2. Multi-Stage Behavioral Vector Extraction

Conversational predation unfolds through established stages. SIREN extracts evidence across four core vectors:

Stage I (Age Probing & Flattery): Inquiries into age, grade level, and maturity validation.
Stage II (Secrecy & Concealment): Demands to hide conversations, delete chat logs, or conceal relationships from parents.
Stage III (Isolation Verification): Inquiries verifying physical isolation, empty rooms, or sleeping guardians.
Stage IV (Boundary Erosion & Solicitations): Channel migration to disappearing platforms (Snapchat, Telegram) and private media solicitations.
3. Ephemeral In-Memory Privacy

All document parsing (PDF/TXT) and inference calculations execute strictly in volatile memory. Raw conversational texts are purged post-feature extraction; no persistent databases or remote analytics retain sensitive child communications.

Dataset Lineage & Calibration

SIREN's calibrated classifier is trained on authentic benchmark data derived from the PAN-CLEF 2012 Sexual Predator Identification Competition (Inches & Crestani, 2012):

Benchmark Corpus: Derived from real-world investigations and IRC/Omegle baseline dialogues.
Balanced Class Distribution: Ingests over 7,000 authentic conversational turns balanced across validated predatory turns and benign adolescent peer interactions.
Hard-Negative Engineering: Includes benign peer discussions involving secret-keeping (surprise parties, gift planning, homework hints) to prevent false alarms and preserve guardian trust.
Directory Structure
siren/
├── main.py
├── train_model.py
├── ingest_pan12.py
├── evaluate_benchmark.py
├── requirements.txt
├── .gitignore
├── models/
│   └── siren_model.joblib
├── static/
│   └── index.html
└── test_cases/
    ├── case_01_homework_study.txt
    ├── case_02_gaming_banter.txt
    ├── case_03_benign_secrecy.txt
    ├── case_04_parental_oversight.txt
    ├── case_05_subtle_age_flattery.txt
    ├── case_06_isolation_probing.txt
    ├── case_07_secrecy_enforcement.txt
    └── case_08_boundary_escalation.txt
```

Quickstart & Installation
1. Clone & Set Up Environment
git clone https://github.com/challa-mahitha/Siren.git
cd Siren
python -m venv .venv

Windows PowerShell:

.venv\Scripts\Activate.ps1

Then:

pip install -r requirements.txt
2. Retrain or Verify Model Artifact
python train_model.py
3. Run Benchmark Test Suite
python evaluate_benchmark.py
4. Launch Application Workspace
python main.py

Navigate to:

http://localhost:8000

Benchmark Evaluation Results

Running evaluate_benchmark.py against the standardized validation suite yields:

Scenario / TranscriptGround TruthPredicted ClassRisk ScoreExpected TierActive VectorsStatus
case_01_homework_study.txt0012SAFE / BENIGNNonePASS
case_02_gaming_banter.txt0012SAFE / BENIGNNonePASS
case_03_benign_secrecy.txt0012SAFE / BENIGNNone (Disambiguated)PASS
case_04_parental_oversight.txt0012SAFE / BENIGNNonePASS
case_05_subtle_age_flattery.txt1146MODERATE CONCERNv1PASS
case_06_isolation_probing.txt1152MODERATE CONCERNv3PASS
case_07_secrecy_enforcement.txt1178HIGH CONCERNv2, v3PASS
case_08_boundary_escalation.txt1196HIGH CONCERNv1, v2, v3, v4PASS
Benchmark Accuracy: 100.0%
Benign Disambiguation False Positive Rate (FPR): 0.0%
Critical Recall: 100.0%
Citation & References
@inproceedings{inches2012overview,
  title={Overview of the International Sexual Predator Identification Competition at PAN-2012},
  author={Inches, Giacomo and Crestani, Fabio},
  booktitle={CLEF 2012 Evaluation Labs and Workshop - Working Notes Papers},
  year={2012},
  address={Rome, Italy}
}

