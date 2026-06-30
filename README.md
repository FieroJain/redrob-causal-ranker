# Redrob Causal Hiring Chain Ranker

**A candidate ranking system that asks a different question than every other hiring tool.**

> Most systems ask: *does this candidate match the job description?*
> This system asks: *if we contact this candidate today, what is the probability they become a hire?*

Scores 100,000 candidates in roughly two minutes on a standard laptop CPU — no GPU required — and produces a ranked, explainable top 100 with a per-candidate reasoning string for every score.

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [The Core Insight](#the-core-insight)
- [Architecture](#architecture)
- [The Three Hard Gates](#the-three-hard-gates)
- [Two Signals Most Systems Don't Model](#two-signals-most-systems-dont-model)
- [Results](#results)
- [The Audit Trail — How This Was Actually Built](#the-audit-trail--how-this-was-actually-built)
- [Setup](#setup)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Known Limitations and Roadmap](#known-limitations-and-roadmap)

---

## Why This Exists

Keyword-matching ranking systems can be gamed by anyone who stuffs the right vocabulary into a profile. They also can't tell the difference between a candidate who is qualified *and reachable* versus a candidate who is qualified but will never respond to outreach, never finish an interview loop, or never accept an offer. Ranking the second candidate #1 wastes a recruiter's time regardless of how good the keyword match looks.

This system treats ranking as a **causal estimation problem**, not a similarity problem.

---

## The Core Insight

```
Every other system asks:              This system asks:
──────────────────────                ─────────────────────────────
"Does this candidate                  "If we contact this candidate
 match the JD?"                        today, what is the probability
                                       they become a hire?"
```

These are not the same question. The first rewards vocabulary. The second combines whether a candidate is qualified with whether the hiring funnel — visibility, response, interview, acceptance — would actually complete.

---

## Architecture

### Two-Pass Hybrid Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  PASS 1 — Structural Scoring                                    │
│  All 100,000 candidates · ~80-85 seconds                        │
│                                                                   │
│  1. Honeypot detection      — impossible-timeline checks        │
│  2. Title gate               — non-technical titles capped       │
│                                 near zero before any skill        │
│                                 keyword is even read              │
│  3. CV/Speech structural gate — applied before retrieval check   │
│  4. Three hard requirement gates (see below)                     │
│  5. Core fit = concept matching + skill trust (temporal decay)   │
│                + career momentum                                 │
│  6. Causal hiring chain — capped at 1.00; behavioral signals     │
│     can only penalize, never inflate                             │
│                                                                   │
│  → Top 500 candidates survive to Pass 2 (max-heap selection)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASS 2 — Semantic Re-ranking                                   │
│  Top 500 only · ~12 seconds                                     │
│                                                                   │
│  8 capability statements embedded via sentence-transformers      │
│  (all-MiniLM-L6-v2), cosine-similarity matched against each      │
│  candidate's career text, blended with the structural score      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Final ranked top 100 + reasoning
```

### The Formula

```
P(hire) = P(recruiter finds them)
        × P(candidate responds)
        × P(interview completes)
        × P(offer accepted)

final_score = core_fit
            × gate_multiplier
            × hiring_chain_multiplier   (capped at 1.00)
            × location_multiplier
            × negative_signal_multiplier
            × yoe_gate
            × notice_gate
            × cv_speech_gate
            × foreign_gate
```

### Why No Component Can Inflate Above Fit

Every multiplier in this system is bounded at **1.00 or below**, with one
deliberate exception: a company-prestige bonus capped at **+8%**. This is
architectural, not incidental — a mediocre-fit candidate with perfect
behavioral signals cannot outrank a strong-fit candidate with average
signals. **Fit sets the ceiling; everything else can only adjust downward.**

This wasn't true in an early version. A scoring bug let the hiring-chain
multiplier reach 1.25×, letting behavioral signals inflate a score 25% above
what fit alone earned — the exact mechanism that let weak-fit candidates leak
into early submissions. Found and fixed; see the audit trail below.

---

## The Three Hard Gates

Before any soft scoring happens, every candidate passes through:

| Gate | Question | Evidence required |
|---|---|---|
| **Production Deployment** | Has this person shipped ML to real users? | Deployment verbs + ML nouns in career text, or deployment tooling (MLflow, BentoML, Triton...) in skills |
| **Retrieval Experience** | Have they built search, ranking, or retrieval systems? | 2+ retrieval signals in career text, or a vector DB in their skills |
| **Technical Depth** | Have they done system-level architecture work? | Architecture, system design, or tech-lead language in career history |

Gates are **multiplicative, not averaged**:

| Gates passed | Score multiplier |
|---|---|
| 0 of 3 | 15% |
| 1 of 3 | 50% |
| 2 of 3 | 80% |
| 3 of 3 | 100% |

A non-technical title (HR Manager, Recruiter, Sales Executive, and 20+ others) is disqualified to a near-zero score **before any skill keyword on the profile is read at all** — keyword-stuffing a disqualified title cannot rescue it.

---

## Two Signals Most Systems Don't Model

### Career Momentum — trajectory, not snapshot

```
Candidate A (rising)                Candidate B (flat)
─────────────────────               ────────────────────
2018 — Data Analyst                 2018 — AI Engineer, consulting firm
2020 — Data Scientist               2020 — AI Engineer, consulting firm
2022 — ML Engineer                  2022 — AI Engineer, consulting firm
2024 — Senior ML Engineer           2024 — AI Engineer, consulting firm
       (vector search, production)         (IT-services projects)

Momentum: HIGH                      Momentum: LOW
```

Same title held in 2024. Identical keyword overlap. Very different hires — and the system scores the difference.

### Temporal Skill Decay — skills have half-lives

A skill's contribution to the score is discounted based on **when it last actually appears in the candidate's career history** (not just whether it's listed), using an exponential decay tied to how fast that specific technology moves:

| Fast-moving (18–24mo half-life) | Stable (48–60mo half-life) |
|---|---|
| LoRA / QLoRA / PEFT | Python / SQL / Java / C++ |
| DPO / RLHF / instruction tuning | BM25 / Elasticsearch |
| RAG / prompt engineering | Learning-to-rank / LambdaMART |
| GPT / Claude / Gemini | PyTorch / TensorFlow |

A 2021 "GPT-3 fine-tuning" claim and a 2025 "QLoRA" claim are not the same evidence, even though both are "LLM fine-tuning experience" on paper.

---

## Results

Final submission (`submissionv12.csv`), independently verified with a custom auditor (not just self-reported by the ranker):

| Metric | Result |
|---|---|
| Candidates scored | 100,000 |
| Runtime, full pipeline | ~103–121 seconds |
| Score spread (top 100) | 0.6242 |
| Candidates scoring above 0.6 | 15 / 100 |
| 120-day-notice candidates in top 100 | **0** |
| CV/Speech-only mismatches in top 100 | **0** |
| Foreign non-relocating candidates in top 100 | **0** |
| Honeypots flagged (heuristic detection) | 9,432 |
| Top 10 candidates, zero flags | 10 / 10 |

Honeypot count reflects heuristic detection (impossible tenure, skill-duration overflow, zero-endorsement "advanced" skills) — not validated against external ground truth, since none exists for this dataset.

---

## The Audit Trail — How This Was Actually Built

This is the part of the project that matters most: **every version was independently audited before being trusted**, using a custom verifier built specifically because the originally-provided audit tool was caught once reading a stale file. Numbers below are from re-running every submission through that verifier, not self-reported.

| Version | NOTICE:120d | CV-primary | FOREIGN | Spread | Outcome |
|---|---|---|---|---|---|
| Early "best" submission | 19 | 4 | 2 | 0.6568 | Believed best — wasn't |
| v7 (bug fixes: hiring-chain cap, exponent) | 19 | 2 | 2 | 0.5918 | Bug fixes alone didn't fix composition |
| v8 (notice + CV gates added) | 1 | 0 | 2 | 0.6296 | Overcorrected — gate too broad |
| v9 (gate narrowed to evidence) | 0 | 0 | 2 | 0.6223 | Both real problems resolved |
| v10-experimental (skill credibility bonus) | 1 | 0 | 2 | 0.6450 | **Regressed a fixed metric — held back, not shipped** |
| **v12 (final — foreign gate, isolated)** | **0** | **0** | **0** | 0.6242 | Strict improvement over v9, fully explained |

Two things this table is meant to show honestly:

1. **A change that looks correct in isolation can still have unintended side effects.** v8's notice gate fixed its target metric but quietly suppressed unrelated good candidates. v10's skill-credibility bonus was theoretically sound but reintroduced a problem that had already been fixed. Both were caught by audit, not assumed safe.
2. **Bundling many changes at once makes failures untraceable.** A separate 9-change version ("v11") hit several of its own stated goals (e.g., zero foreign candidates) but *missed its own headline target* (score spread) and couldn't be trusted as a whole — so it was decomposed back into single, independently-tested changes. Only the isolated foreign-candidate gate from that bundle survived audit and was merged in as v12.

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

`candidates.jsonl` is **not included in this repo** — it's 487MB (exceeds GitHub's 100MB file limit) and is the hackathon's provided dataset, not this project's own code. Place your own copy of `candidates.jsonl` in this folder before running the ranker or the demo.

---

## Usage

### Run the ranker

```bash
python redrob_ranker_v12.py --candidates candidates.jsonl --out submissionv12.csv
```

~100–120 seconds for 100,000 candidates on a standard laptop CPU. If `sentence-transformers` isn't installed, the script automatically falls back to structural-only scoring (Pass 1) and skips semantic re-ranking (Pass 2) — it never hard-crashes on a missing optional dependency.

### Run the live demo

```bash
streamlit run app.py
```

Point the sidebar at your local `candidates.jsonl` and `submissionv12.csv`. Includes live gate-firing visualization, a candidate deep-dive view, a career-momentum chart, the keyword-stuffer-trap comparison, and the full architecture/audit history above, rendered interactively.

### Convert ranked output to XLSX

```bash
python convert_to_xlsx.py
```

---

## Repository Structure

| File | Purpose |
|---|---|
| `redrob_ranker_v12.py` | Final ranker — structural + semantic two-pass scoring |
| `app.py` | Streamlit demo |
| `convert_to_xlsx.py` | Converts the ranked CSV to the XLSX format required for submission |
| `submissionv12.csv` | Final ranked output (top 100, CSV) |
| `submission_final.xlsx` | Final ranked output (XLSX, for hackathon submission) |
| `sample_submission.csv` | Provided format reference (naive baseline, used for the keyword-stuffer-trap comparison in the demo) |
| `requirements.txt` | Pinned dependencies |

---

## Known Limitations and Roadmap

Built and held back, not hidden:

- **Context-aware title scoring** ("Software Engineer at a top AI lab" likely means more than the title alone implies) was implemented and tested, but kept **disabled by default** — it shares a risk pattern with the hiring-chain cap bug already found once, and deserves its own isolated audit cycle before shipping.
- **Static skill taxonomy.** LoRA, QLoRA, and DPO didn't exist four years ago. A weekly pipeline pulling signal from arXiv, GitHub, and Hugging Face — with a human review queue before anything reaches production — would keep the taxonomy current instead of frozen at build time. Deliberately *not* built for this submission: it requires live network calls and has no safety net on a hackathon timeline.
- **Behavioral weights are heuristics, not learned.** Visibility, response, interview, and acceptance weights are reasoned estimates today. The architecture is designed so a model trained on Redrob's real hiring outcomes could replace the heuristic without changing the pipeline shape.

We know exactly what this system cannot do yet, and exactly how we'd validate each fix before shipping it — that's an engineering roadmap, not a weakness.
