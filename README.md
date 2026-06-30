# Redrob Causal Hiring Chain Ranker

Candidate ranking system for the Redrob Hackathon. Scores 100,000 candidates
against a job description in under 2 minutes, using a two-pass architecture:
structural gating + scoring, then semantic re-ranking on the top 500.

Final submission: `submissionv12.csv` (also provided as `submission_final.xlsx`).

## Core idea

Most ranking systems ask "does this candidate match the JD?" — a static
text-similarity question that rewards vocabulary. This system asks "if we
contact this candidate today, what is the probability they become a hire?" —
a causal question that combines technical fit with the actual hiring funnel
(visibility → response → interview → acceptance).

Fit sets the ceiling; behavioral signals can only penalize a score, never
inflate it above what the fit earned.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

`candidates.jsonl` is **not included in this repo** (487MB, exceeds GitHub's
file size limit, and is the hackathon's provided dataset rather than this
project's own code). Place your own copy of `candidates.jsonl` in this folder
before running the ranker or the demo.

## Run the ranker

```bash
python redrob_ranker_v12.py --candidates candidates.jsonl --out submissionv12.csv
```

Runtime: ~100-120 seconds for 100,000 candidates on a standard laptop CPU
(no GPU required). If `sentence-transformers` isn't installed, the script
falls back to structural-only scoring (Pass 1) and skips the semantic
re-ranking step (Pass 2) automatically.

## Run the demo

```bash
streamlit run app.py
```

Point the sidebar at your local `candidates.jsonl` and `submissionv12.csv`.
The demo includes live gate-firing visualization, a candidate deep-dive view,
a career-momentum chart, and the full iteration/audit history.

## Convert ranked output to XLSX

```bash
python convert_to_xlsx.py
```

## Architecture summary

See `submissionv12.csv`'s reasoning column for per-candidate explanations,
or the "Architecture & Iteration History" page in the Streamlit demo for the
full formula and the audit trail across versions (v7 → v12), including a
change that was built, tested, and deliberately held back after an audit
showed it regressed a metric that had already been fixed.

## Files

| File | Purpose |
|---|---|
| `redrob_ranker_v12.py` | Final ranker — structural + semantic two-pass scoring |
| `app.py` | Streamlit demo |
| `convert_to_xlsx.py` | Converts the ranked CSV to the XLSX format required for submission |
| `submissionv12.csv` | Final ranked output (top 100, CSV) |
| `submission_final.xlsx` | Final ranked output (XLSX, for hackathon submission) |
| `sample_submission.csv` | Provided format reference (naive baseline, used for the "trap" comparison in the demo) |
| `requirements.txt` | Pinned dependencies |
