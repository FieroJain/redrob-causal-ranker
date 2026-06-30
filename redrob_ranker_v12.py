#!/usr/bin/env python3
"""Redrob Hackathon Ranker v12 — ISOLATED TEST 1 of N: foreign gate only

v11 bundled 9 simultaneous changes into one jump from v9, including two
thresholds tuned by name against specific candidate IDs (CAND_0096142,
CAND_0039754) seen in prior audits. The result contradicted its own stated
goal (spread target 0.65+, actual 0.5873, worse than v9's 0.6223) and there
was no way to tell which of the 9 changes caused that.

v12 isolates the single most independently-justified change from v11: an
explicit multiplicative gate for non-India candidates. FOREIGN has been
stuck at 2 (the same two Berlin candidates) across every version from
submission.csv through v10 — five separate scoring runs, two code lineages.
This gate stacks with the existing get_location_multiplier rather than
replacing it (0.65 relocate-loc_mult x 0.68 relocate-foreign_gate = 0.442,
vs 0.65 alone before), and is guarded the same way notice_gate/cv_gate are
(India check first, so it can never affect India-based candidates).

Everything else is unchanged from v9. If this audits clean, the next
isolated test is the YOE retightening; if it doesn't, we stop here and know
exactly which change is responsible.

Changes from v9 (un-overcorrect the notice gate):
  v8's audit showed the notice gate worked for its actual target (120d:
  19 -> 1) and the CV-primary gate worked perfectly (2 -> 0), but "above 0.6"
  fell from 23 to 13 and spread dropped to 0.6296. Root cause: v8 used
  graduated bands (90d: 0.90x, 91-119d: 0.80x) that penalized notice lengths
  which were never actually flagged as a problem in the audit — your data
  only has discrete buckets (0/15/30/45/60/90/120d) and only 120d showed up
  as an issue. v9 narrows the gate to that single threshold.

Changes from v7 (composition fix, based on the verify_submission.py audit):
  v7 fixed two real bugs (hiring_mult cap, power exponent) but those only
  reshape score MAGNITUDE — they don't change WHICH candidates make the top
  100. The audit showed v7 still let 19 candidates with 120-day notice and
  2 CV/speech-only candidates into the top 100 (previous best: 5 and 0).

  Root cause: both were soft, additive penalties that get diluted by other
  large positive multipliers (prestige bonus, high concept score, momentum).
  Notice period only carried 8% weight inside availability_mult, and the
  CV/speech mismatch penalty was a +0.40 add to neg_score, capped by the
  same 0.85 ceiling as every other penalty.

  Fix: both now ALSO apply as direct multiplicative gates on the final raw
  score, the same pattern already used for yoe_gate. This makes them load-
  bearing instead of easily outvoted by unrelated strong signals.

Changes from v6 (carried forward from v7):
  BUG FIXES (per audit):
    - hiring_mult cap restored to 1.00 (was 1.25) — behavioral signals can no
      longer inflate a weak technical fit by 25%. This was the same bug class
      that previously let CV/speech-only candidates leak into the top 100.
    - Power transform exponent restored to 0.65 (was 0.75) — 0.75 over-compresses
      the top end and flattens score spread.

  NEW (real, testable — not speculative):
    - Per-skill temporal decay: a skill's trust contribution now decays based
      on (a) how long ago it last appeared in the candidate's career history,
      and (b) a static technology-velocity table (fast-moving fields like LoRA/
      RAG decay faster than stable ones like Python/SQL). This is implemented
      with zero external dependencies and zero network calls — it only reads
      data already present in candidates.jsonl.
    - Pareto-frontier tagging: after final ranking, candidates whose
      (technical fit, availability) combination isn't dominated by any other
      top-100 candidate get an explicit "Pareto-optimal" note appended to
      their reasoning. This is informational only — it does NOT alter scores
      or ordering, so it can't destabilize a ranking that's already working.

  Deliberately NOT included: a live-scraping "self-updating skill taxonomy"
  (arxiv/GitHub/HF trending). That requires network access, has no human
  review step in a hackathon timeframe, and risks injecting noise straight
  into your scores. Keep that as the "Where this goes next" slide — don't
  ship it as runtime code today.

Optimisations (unchanged from v6):
- Gate-first scoring: expensive regex only for candidates that pass gates
- Shorter career text (2000 chars)
- Pre-compiled keyword sets for fast membership checks
- Combined production evidence regex
- Company prestige bonus for high-signal product companies
- Tie-break fix: round then sort by (-rounded_score, candidate_id)
- Semantic layer: uses sentence-transformers for capability scoring on top 500 (if available)

To enable semantic layer:
    pip install sentence-transformers
"""

from datetime import datetime, date
import re, math, json, csv, heapq, argparse, sys
from pathlib import Path

REFERENCE_DATE = date(2026, 6, 14)
DEBUG = False

# ── OPTIONAL SEMANTIC LAYER ──────────────────────────────────────────────
SEMANTIC_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SEMANTIC_AVAILABLE = True
except ImportError:
    np = None
    print("WARNING: sentence-transformers not installed. Semantic layer disabled.")

# ── CAPABILITIES (same) ──────────────────────────────────────────────────
CAPABILITIES = [
    "Deploy vector search and embedding retrieval in production at scale",
    "Build hybrid search combining dense embeddings and keyword retrieval",
    "Design learning-to-rank and re-ranking systems with NDCG evaluation",
    "Fine-tune language models using LoRA PEFT or similar techniques",
    "Create offline and online evaluation frameworks for ranking quality",
    "Architect low-latency high-throughput ML inference systems",
    "Build recommendation or personalization systems at product companies",
    "Technical leadership at early-stage product company founding team"
]

# ── CONCEPT MAP (unchanged) ─────────────────────────────────────────────
CONCEPT_MAP = {
    "vector_db": [
        "milvus", "pinecone", "weaviate", "qdrant", "faiss",
        "opensearch", "elasticsearch", "chromadb", "pgvector",
        "ann", "approximate nearest", "vector index", "vector store",
        "dense retrieval",
    ],
    "retrieval_and_search": [
        "information retrieval", "semantic search", "hybrid search",
        "bm25", "recommendation", "recommender", "personalization",
        "feed ranking", "search engine", "query understanding",
        "result ranking", "relevance scoring", "search quality",
        "matching engine", "similarity matching", "nearest neighbor",
        "candidate retrieval", "product search", "discovery system",
        "relevance engineer", "feed algorithm",
    ],
    "ranking_systems": [
        "learning to rank", "ltr", "pointwise", "pairwise", "listwise",
        "xgboost rank", "lambdamart", "re-ranking", "reranking",
        "ranking model", "relevance model", "ranking system",
        "ranking pipeline", "search ranking",
    ],
    "embeddings": [
        "embedding", "sentence transformer", "dense vector",
        "bert", "encoder", "bi-encoder", "cross-encoder",
        "e5", "bge", "gte", "text embedding", "word2vec",
        "doc2vec", "contrastive learning",
    ],
    "llm_and_finetuning": [
        "fine-tun", "finetuning", "lora", "qlora", "peft", "sft",
        "dpo", "rlhf", "instruction tuning", "domain adaptation",
        "llm", "large language model", "gpt", "claude", "gemini",
        "weights & biases", "wandb", "prompt engineering",
        "chain of thought", "retrieval augmented", "rag",
    ],
    "evaluation": [
        "ndcg", "mrr", "a/b test", "ab test", "online eval",
        "offline eval", "evaluation framework", "precision@",
        "recall@", "click-through", "engagement metric",
        "offline benchmark", "online metric", "map@",
    ],
    "production_deployment": [
        "production", "deployed to prod", "serving", "online serving",
        "real-time inference", "low latency", "high throughput",
        "p99", "sla", "uptime", "model drift", "monitoring",
        "retraining pipeline", "mlops", "model registry", "mlflow",
        "bentoml", "mlflow", "torchserve", "triton", "seldon", "kserve",
        "ray serve", "sagemaker", "tensorflow serving",
    ],
}

CONCEPT_PATTERNS = {}
for _cname, _kws in CONCEPT_MAP.items():
    _parts = sorted((re.escape(k) for k in _kws), key=len, reverse=True)
    CONCEPT_PATTERNS[_cname] = re.compile("|".join(_parts), re.IGNORECASE)

PRODUCTION_REGEX = re.compile(
    r'(?:deployed|launched|shipped|serving|in production|live|real[\s\-]time|online)'
    r'.{0,250}'
    r'(?:\d+\s*[kmbt]?\s*(?:users|queries|req|rps|requests|transactions|records|candidates|impressions|calls)|'
    r'p99|latency|throughput|sla|a/?b\s*test|drift|monitoring)'
    r'|'
    r'(?:\d+\s*[kmbt]?\s*(?:users|queries|req|rps|requests|transactions|records|candidates|impressions|calls)|'
    r'p99|latency|throughput|sla|a/?b\s*test|drift|monitoring)'
    r'.{0,250}'
    r'(?:deployed|launched|shipped|serving|in production|live|real[\s\-]time|online)',
    re.IGNORECASE | re.DOTALL
)

def _has_production_evidence(window: str) -> bool:
    return bool(PRODUCTION_REGEX.search(window))

# ── TITLE SCORING ──────────────────────────────────────────────────────
TITLE_SCORE_MAP = [
    ("senior ai engineer",              1.00),
    ("ai engineer",                     1.00),
    ("senior machine learning engineer",0.97),
    ("principal ml engineer",           0.97),
    ("staff machine learning engineer", 0.96),
    ("staff ml engineer",               0.96),
    ("lead ai engineer",                0.92),
    ("machine learning engineer",       0.95),
    ("ml engineer",                     0.95),
    ("applied ml engineer",             0.90),
    ("applied scientist",               0.88),
    ("ai research engineer",            0.93),
    ("nlp engineer",                    0.90),
    ("research engineer",               0.85),
    ("recommendation systems engineer", 0.91),
    ("search engineer",                 0.88),
    ("ai specialist",                   0.82),
    ("senior software engineer (ml)",   0.85),
    ("senior data scientist",           0.75),
    ("lead data scientist",             0.75),
    ("data scientist",                  0.70),
    ("junior ml engineer",              0.35),
    ("junior data scientist",           0.35),
    ("data engineer",                   0.40),
    ("software engineer",               0.38),
    ("backend engineer",                0.35),
    ("full stack engineer",             0.28),
    ("frontend engineer",               0.15),
    ("computer vision engineer",        0.40),
    ("data analyst",                    0.18),
]

TITLE_DISQUALIFIERS = [
    "hr manager", "hr ", "human resource", "people manager",
    "marketing manager", "marketing ",
    "content writer", "content manager",
    "graphic design", "graphic artist",
    "accountant", "account manager", "finance manager",
    "sales executive", "sales manager", "sales ",
    "customer support", "customer success",
    "operations manager", "operations ",
    "project manager", "program manager",
    "mechanical engineer", "civil engineer",
    "recruiter", "talent acquisition", "sourcer",
]

CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro",
    "accenture", "cognizant", "capgemini", "hcl technologies",
    "hcltech", "tech mahindra", "mphasis", "hexaware",
    "ltimindtree", "lti mindtree", "persistent systems",
    "niit technologies", "mastech",
}

HIGH_SIGNAL_COMPANIES = {
    # Indian product companies — strong signal
    "google", "microsoft", "amazon", "meta", "apple",
    "flipkart", "swiggy", "zomato", "meesho", "razorpay",
    "phonepe", "paytm", "cred", "freshworks", "zoho",
    "sharechat", "dream11", "byju", "unacademy", "groww",
    "sarvam", "krutrim", "niramai", "haptik", "observe.ai",
    # Global AI companies
    "openai", "anthropic", "deepmind", "hugging face",
    "cohere", "mistral", "stability ai",
}

CV_SPEECH_KEYWORDS = [
    "computer vision", "image classification", "object detection",
    "yolo", "opencv", "convolutional neural", "image segmentation",
    "speech recognition", "automatic speech", "asr", "text-to-speech",
    "tts", "speaker identification", "audio classification",
]
NLP_IR_KEYWORDS = [
    "nlp", "natural language processing", "text classification",
    "named entity", "information retrieval", "search ranking",
    "semantic search", "text embedding", "question answering",
    "document ranking", "passage retrieval",
]

PREFERRED_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore",
    "bengaluru", "gurgaon", "gurugram", "greater noida", "ncr",
}

ASSESSMENT_RELEVANCE = {
    "NLP": 0.90,
    "Fine-tuning LLMs": 0.85,
    "Speech Recognition": 0.20,
    "Image Classification": 0.10,
}

# ── TECH VELOCITY / SKILL RECENCY DECAY (NEW in v7) ─────────────────────
# Static half-life table in months. Fast-moving sub-fields (LLM fine-tuning
# techniques) get short half-lives; stable foundations get long ones.
# Order matters: first matching substring wins, so list more specific terms
# before generic ones where there's overlap.
TECH_VELOCITY = {
    "qlora": 18, "lora": 18, "peft": 18, "sft": 20,
    "dpo": 20, "rlhf": 20, "instruction tuning": 20,
    "rag": 20, "retrieval augmented": 20, "prompt engineering": 22,
    "chain of thought": 22, "llm": 24, "large language model": 24,
    "gpt": 22, "claude": 24, "gemini": 24,
    "vector index": 30, "vector store": 30, "dense retrieval": 30,
    "milvus": 30, "pinecone": 30, "weaviate": 30, "qdrant": 30,
    "faiss": 36, "pgvector": 30, "ann": 32,
    "embedding": 30, "sentence transformer": 28, "bi-encoder": 30,
    "cross-encoder": 30, "bert": 38, "e5": 28, "bge": 28, "gte": 28,
    "bm25": 48, "elasticsearch": 42, "opensearch": 42,
    "learning to rank": 40, "lambdamart": 44, "ndcg": 44,
    "mlflow": 36, "kubernetes": 42, "docker": 44, "triton": 32,
    "pytorch": 40, "tensorflow": 40,
    "python": 60, "sql": 60, "java": 60, "c++": 60,
}
DEFAULT_HALF_LIFE_MONTHS = 42
SKILL_DECAY_FLOOR = 0.35   # never zero-out a skill purely for being old
UNMATCHED_SKILL_MEMORY_DECAY = 0.75  # skill listed but never traced to a job


def _months_between(later: date, earlier: date) -> int:
    """Whole months from `earlier` to `later` (assumes later >= earlier)."""
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def _parse_job_end_date(job: dict, reference_date: date):
    """Best-effort end date for a job: reference_date if current, else start+duration."""
    if job.get("is_current"):
        return reference_date
    start_str = job.get("start_date", "") or ""
    duration = job.get("duration_months", 0) or 0
    if len(start_str) >= 7:
        try:
            sy, sm = int(start_str[:4]), int(start_str[5:7])
            total_months = sy * 12 + (sm - 1) + duration
            ey, em0 = divmod(total_months, 12)
            return date(ey, em0 + 1, 1)
        except Exception:
            return None
    return None


def get_skill_recency_decay(skill_name: str, career: list, reference_date: date) -> float:
    """
    How fresh is this skill, given when it last showed up in the candidate's
    actual career history? Fast-moving technologies decay faster than stable
    ones. This only reads fields already present in candidates.jsonl — no
    network calls, no external data.
    """
    name_lower = (skill_name or "").lower()
    if not name_lower:
        return 1.0

    matched_kw, half_life = None, None
    for kw, hl in TECH_VELOCITY.items():
        if kw in name_lower:
            matched_kw, half_life = kw, hl
            break
    if half_life is None:
        # Unrecognized / generic skill — don't penalize without evidence.
        return 1.0

    most_recent_end = None
    for job in career:
        text = ((job.get("description", "") or "") + " " + (job.get("title", "") or "")).lower()
        if matched_kw in text or name_lower in text:
            end = _parse_job_end_date(job, reference_date)
            if end and (most_recent_end is None or end > most_recent_end):
                most_recent_end = end

    if most_recent_end is None:
        # Listed as a skill but never traced to a specific role — treat as
        # moderately stale rather than penalizing as hard as proven absence.
        return UNMATCHED_SKILL_MEMORY_DECAY

    months_since = max(0, _months_between(reference_date, most_recent_end))
    if months_since == 0:
        return 1.0
    decay = math.exp(-0.693 * months_since / half_life)
    return max(SKILL_DECAY_FLOOR, decay)


# ── OPTIMISED HELPERS ──────────────────────────────────────────────────

# Pre-compile keyword lists as sets for fast membership checks
DEPLOYMENT_VERBS = {"deployed","launched","shipped","serving","production","live","real-time","online"}
ML_NOUNS = {"model","ml","ai","embedding","recommender","search","ranking","pipeline","inference","predict","scoring"}
RETRIEVAL_EVIDENCE = {"vector","embedding","retrieval","search","recommendation","recommender","ranking","similarity",
                       "milvus","faiss","pinecone","elasticsearch","opensearch","bm25","dense","semantic search",
                       "nearest neighbor","ann","vector index"}
DEPLOYMENT_SKILLS = {"bentoml","mlflow","torchserve","triton","seldon","kserve","ray serve","sagemaker","tensorflow serving","onnx"}
RETRIEVAL_SKILLS = {"milvus","faiss","pinecone","weaviate","qdrant","elasticsearch","opensearch","chromadb","redis vector"}
SCOPE_WORDS = {"architecture","design","technical lead","system design","trade-offs","scalability"}

def get_title_score(title: str) -> float:
    t = (title or "").lower().strip()
    for disq in TITLE_DISQUALIFIERS:
        if disq in t:
            return 0.0
    for key, value in TITLE_SCORE_MAP:
        if key in t:
            return value
    if "engineer" in t:
        return 0.28
    if "scientist" in t:
        return 0.55
    if "developer" in t:
        return 0.22
    if "analyst" in t:
        return 0.20
    return 0.08

def get_company_prestige_bonus(career: list) -> float:
    """Small bonus for candidates from high-signal companies."""
    bonus = 0.0
    for job in career:
        company = (job.get("company","") or "").lower()
        if any(hsc in company for hsc in HIGH_SIGNAL_COMPANIES):
            if job.get("is_current"):
                bonus = max(bonus, 0.08)
            else:
                bonus = max(bonus, 0.04)
    return bonus

def is_cv_speech_primary(career, skills) -> bool:
    all_skill_names = " ".join(s.get("name", "").lower() for s in skills)
    all_career_text = " ".join(
        (j.get("description", "") or "") + " " + (j.get("title", "") or "")
        for j in career
    ).lower()
    combined = all_skill_names + " " + all_career_text
    has_cv_speech = any(kw in combined for kw in CV_SPEECH_KEYWORDS)
    has_nlp_ir = any(kw in combined for kw in NLP_IR_KEYWORDS)
    return has_cv_speech and not has_nlp_ir

def check_production_deployment_fast(career, skills) -> bool:
    # Check career descriptions
    for job in career:
        desc = (job.get("description", "") or "").lower()
        if any(v in desc for v in DEPLOYMENT_VERBS) and any(n in desc for n in ML_NOUNS):
            return True
    # Check skills
    skill_names = {s.get("name", "").lower() for s in skills}
    if skill_names & DEPLOYMENT_SKILLS:
        return True
    return False

def check_retrieval_experience_fast(career, skills, cv_speech_primary: bool = False) -> bool:
    min_hits = 3 if cv_speech_primary else 2
    for job in career:
        text = ((job.get("description", "") or "") +
                (job.get("title", "") or "")).lower()
        # Count matches using set intersection
        matches = sum(1 for kw in RETRIEVAL_EVIDENCE if kw in text)
        if matches >= min_hits:
            return True
    skill_names = {s.get("name", "").lower() for s in skills}
    if not cv_speech_primary and skill_names & RETRIEVAL_SKILLS:
        return True
    return False

def check_technical_depth_fast(career) -> bool:
    for job in career:
        desc = (job.get("description", "") or "").lower()
        if any(w in desc for w in SCOPE_WORDS):
            return True
    return False

def is_honeypot(candidate):
    career = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    yoe = float(candidate.get("profile", {}).get("years_of_experience", 0) or 0)
    for job in career:
        start_str = job.get("start_date", "") or ""
        duration = job.get("duration_months", 0) or 0
        if len(start_str) >= 7:
            try:
                sy = int(start_str[:4])
                sm = int(start_str[5:7])
                max_possible = (2026 - sy) * 12 + (6 - sm) + 2
                if duration > max_possible:
                    return (True, f"impossible tenure: {duration}mo starting {start_str[:7]}")
            except Exception:
                pass
    for skill in skills:
        skill_dur = skill.get("duration_months", 0) or 0
        if skill_dur > yoe * 12 + 12:
            return (True, f"skill duration {skill_dur}mo exceeds career length {yoe*12:.0f}mo")
    zero_endorse_advanced = sum(
        1 for s in skills
        if s.get("proficiency") == "advanced" and (s.get("endorsements") or 0) == 0
    )
    if zero_endorse_advanced >= 4:
        return (True, f"{zero_endorse_advanced} advanced skills with 0 endorsements")
    return (False, "")

def contextualized_concept_score_optimized(text: str, max_len=2000):
    """Same as before, but with shorter text and early exit."""
    text_lower = (text or "")[:max_len].lower()
    total_score = 0.0
    matched_concepts = []
    n = len(text_lower)
    for concept_name, pattern in CONCEPT_PATTERNS.items():
        best = 0.0
        found_production = False
        for m in pattern.finditer(text_lower):
            if found_production:
                break
            start = max(0, m.start() - 200)
            end = min(n, m.end() + 200)
            window = text_lower[start:end]
            if _has_production_evidence(window):
                best = 2.0
                found_production = True
            else:
                best = max(best, 1.0)
        if best > 0:
            matched_concepts.append(concept_name)
            total_score += best
    max_possible = 2.0 * len(CONCEPT_MAP)
    normalized = total_score / max_possible if max_possible > 0 else 0.0
    return (min(normalized, 1.0), matched_concepts)

def get_skill_trust_score(skills, assessments, career=None, reference_date=REFERENCE_DATE) -> float:
    """
    v7: now applies per-skill temporal decay. A skill's trust contribution is
    discounted based on how long ago it last appeared in the candidate's
    actual career history, scaled by how fast that specific technology moves
    (see TECH_VELOCITY). This is the "skills have half-lives" idea from the
    audit, made concrete and testable instead of staying a slide bullet.
    """
    if not skills:
        return 0.0
    career = career or []
    total = 0.0
    count = 0
    for skill in skills:
        name = skill.get("name", "")
        if name in assessments:
            raw_score = assessments[name]
            relevance = ASSESSMENT_RELEVANCE.get(name, 0.50)
            trust = (float(raw_score) / 100.0) * relevance * 3.0
        else:
            prof = skill.get("proficiency", "beginner")
            endorsements = skill.get("endorsements", 0) or 0
            duration = skill.get("duration_months", 0) or 0
            prof_weight = {"advanced": 1.0, "expert": 1.0,
                           "intermediate": 0.65, "beginner": 0.3}.get(prof, 0.3)
            endorse_weight = math.log1p(endorsements) / math.log1p(100)
            duration_weight = min(duration / 36.0, 1.0)
            trust = (prof_weight * 0.40) + (endorse_weight * 0.35) + (duration_weight * 0.25)

        decay = get_skill_recency_decay(name, career, reference_date)
        trust *= decay

        total += trust
        count += 1
    return min(total / count, 1.0) if count > 0 else 0.0

def career_momentum_score_fast(career) -> float:
    if not career:
        return 0.0
    sorted_career = sorted(career, key=lambda j: (j.get("start_date", "") or ""), reverse=False)
    job_scores = []
    for job in sorted_career:
        text = (job.get("description", "") or "") + " " + (job.get("title", "") or "")
        concept_s, _ = contextualized_concept_score_optimized(text, max_len=1500)
        company = (job.get("company", "") or "").lower()
        industry = (job.get("industry", "") or "").lower()
        is_product = (
            not any(f in company for f in CONSULTING_FIRMS) and
            "it service" not in industry and
            "consulting" not in industry
        )
        product_bonus = 0.12 if is_product else 0.0
        job_scores.append(min(1.0, concept_s + product_bonus))
    if len(job_scores) < 2:
        return job_scores[0] if job_scores else 0.0
    deltas = []
    for i in range(1, len(job_scores)):
        delta = job_scores[i] - job_scores[i - 1]
        recency_w = 2.0 if i == len(job_scores) - 1 else 1.0
        deltas.append(delta * recency_w)
    avg_delta = sum(deltas) / len(deltas)
    current = job_scores[-1]
    momentum = current * 0.7 + max(0.0, avg_delta) * 0.3 * 2
    return min(1.0, max(0.0, momentum))

# ── Other helper functions (unchanged) ──
def get_location_multiplier(profile, signals): # same as before
    country = (profile.get("country", "") or "").lower().strip()
    location = (profile.get("location", "") or "").lower().strip()
    relocate = signals.get("willing_to_relocate", False)
    is_india = (country in ("india", "in", "") or not country)
    if not is_india:
        return 0.65 if relocate else 0.22
    for city in PREFERRED_CITIES:
        if city in location:
            return 1.10
    return 1.00

def recruiter_visibility_score(signals): # same
    completeness = float(signals.get("profile_completeness_score") or 50) / 100.0
    search_appearances = float(signals.get("search_appearance_30d") or 0)
    saved_by_recruiters = float(signals.get("saved_by_recruiters_30d") or 0)
    profile_views = float(signals.get("profile_views_received_30d") or 0)
    linkedin_connected = bool(signals.get("linkedin_connected", False))
    open_to_work = bool(signals.get("open_to_work_flag", False))
    verified_email = bool(signals.get("verified_email", False))
    verified_phone = bool(signals.get("verified_phone", False))
    score = 0.0
    score += completeness * 0.20
    score += min(search_appearances / 500.0, 1.0) * 0.25
    score += min(saved_by_recruiters / 5.0, 1.0) * 0.30
    score += min(profile_views / 100.0, 1.0) * 0.10
    if open_to_work:
        score += 0.05
    if linkedin_connected:
        score += 0.05
    if verified_email and verified_phone:
        score += 0.05
    return min(score, 1.0)

def get_notice_score(signals):
    notice = int(signals.get("notice_period_days") or 60)
    if notice <= 15:
        return 1.0
    elif notice <= 30:
        return 0.90
    elif notice <= 60:
        return 0.75
    elif notice <= 90:
        return 0.55
    else:
        return 0.30

def get_yoe_gate(yoe: float) -> float:
    if 5.0 <= yoe <= 9.0:
        return 1.00
    if 4.0 <= yoe < 5.0:
        return 0.85
    if 9.0 < yoe <= 11.0:
        return 0.80
    if 3.0 <= yoe < 4.0:
        return 0.70
    if 11.0 < yoe <= 13.0:
        return 0.65
    if yoe > 13.0:
        return 0.42
    return 0.45

def get_notice_gate(notice_days: int) -> float:
    """
    v9: narrowed from v8. The audit showed the ONLY notice value that was
    actually a problem was 120d (19 such candidates leaked into v7's top
    100; your data only uses discrete buckets: 0/15/30/45/60/90/120). v8's
    graduated bands also docked 90d candidates 10% and an unused 91-119d
    band 20% — penalties for a problem that was never observed. That's why
    "above 0.6" fell to 13 and spread dropped instead of recovering. This
    version only gates the one threshold the data actually showed was an
    issue, leaving everything below it untouched.
    """
    return 0.62 if notice_days >= 120 else 1.00

def get_cv_speech_mismatch_gate(cv_speech_primary: bool) -> float:
    """
    v8: direct multiplicative gate for candidates whose career is CV/speech
    work with no NLP/IR evidence. The existing +0.40 additive penalty (see
    get_negative_signal_score) gets diluted when prestige bonus, momentum,
    and concept score are all high — that's how 2 such candidates leaked
    into v7's top 100. This gate applies on top of that penalty, deliberately
    a strong soft penalty (not a hard zero-out) since the keyword-based
    CV/speech detector can have false positives and shouldn't be unreviewable.
    """
    return 0.30 if cv_speech_primary else 1.00

def get_foreign_gate(profile: dict, signals: dict) -> float:
    """
    v12: isolated from v11. Explicit multiplicative gate for non-India
    candidates, stacking with (not replacing) get_location_multiplier.
    FOREIGN has been stuck at 2 across every prior version regardless of
    code lineage — the same two candidates every time — so loc_mult alone
    (0.65 relocate / 0.22 no-relocate) was never enough to move it.
    India check runs first, same guard pattern as notice_gate/cv_gate, so
    this can never affect an India-based candidate.
    """
    country = (profile.get("country", "") or "").lower().strip()
    is_india = (country in ("india", "in", "") or not country)
    if is_india:
        return 1.00
    relocate = signals.get("willing_to_relocate", False)
    return 0.68 if relocate else 0.55

def get_negative_signal_score(candidate) -> tuple:
    career = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}
    penalty = 0.0
    reasons = []
    if len(career) > 3:
        durations = [j.get("duration_months", 24) or 24 for j in career]
        avg_duration = sum(durations) / len(durations)
        if avg_duration < 18:
            penalty += 0.35
            reasons.append(f"avg tenure {avg_duration:.0f}mo (title-chaser)")
    all_skill_names = " ".join(s.get("name", "").lower() for s in skills)
    all_career_text = " ".join(
        (j.get("description", "") or "") + " " + (j.get("title", "") or "")
        for j in career
    ).lower()
    combined = all_skill_names + " " + all_career_text
    has_cv_speech = any(kw in combined for kw in CV_SPEECH_KEYWORDS)
    has_nlp_ir = any(kw in combined for kw in NLP_IR_KEYWORDS)
    if has_cv_speech and not has_nlp_ir:
        penalty += 0.40
        reasons.append("CV/speech-primary without NLP/IR")
    if career:
        all_consulting = all(
            any(f in (j.get("company", "") or "").lower() for f in CONSULTING_FIRMS)
            for j in career
        )
        if all_consulting:
            penalty += 0.35
            reasons.append("consulting-only career")
    last_active_str = signals.get("last_active_date", "") or ""
    open_to_work = signals.get("open_to_work_flag", False)
    rr_val = float(signals.get("recruiter_response_rate") or 0.5)
    if last_active_str:
        try:
            la = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_inactive = (REFERENCE_DATE - la).days
            is_ghost = (rr_val < 0.20 and days_inactive > 90)
            if is_ghost:
                penalty += 0.55
                reasons.append(f"GHOST: rr={rr_val:.2f} inactive={days_inactive}d")
            elif days_inactive > 180 and not open_to_work:
                penalty += 0.20
                reasons.append(f"stale: inactive {days_inactive}d")
            elif days_inactive > 120 and rr_val < 0.30 and not open_to_work:
                penalty += 0.15
                reasons.append(f"borderline stale: inactive {days_inactive}d")
        except Exception:
            pass
    return (min(penalty, 0.85), reasons)

def extract_best_career_sentence(career, matched_concepts: list) -> str:
    for job in sorted(career, key=lambda j: 0 if j.get("is_current") else 1):
        desc = job.get("description", "") or ""
        company = job.get("company", "") or ""
        title = job.get("title", "") or ""
        sentences = [s.strip() for s in re.split(r'[.!?]', desc) if len(s.strip()) > 30]
        for sent in sentences:
            sl = sent.lower()
            has_scale = bool(re.search(r'\d+[kmb]?\s*(users|queries|req)', sl))
            has_deploy = any(w in sl for w in ["deployed", "shipped", "launched", "serving"])
            has_concept = any(
                kw in sl
                for concept in matched_concepts
                for kw in CONCEPT_MAP.get(concept, [])
            )
            if has_scale or (has_deploy and has_concept):
                return f'{title} at {company}: "{sent[:120]}"'
        if sentences:
            return f'{title} at {company}: "{sentences[0][:100]}"'
    return ""

def build_reasoning(candidate, matched_concepts, neg_score, neg_reasons,
                    core_fit, hiring_chain, final_score,
                    cap_breakdown=None) -> str:
    profile = candidate.get("profile", {}) or {}
    signals = candidate.get("redrob_signals", {}) or {}
    career = candidate.get("career_history", []) or []
    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0) or 0
    location = profile.get("location", "") or ""
    company = profile.get("current_company", "") or ""
    rr = signals.get("recruiter_response_rate")
    notice = int(signals.get("notice_period_days") or 60)
    last = signals.get("last_active_date", "") or ""
    parts = []

    best_sentence = extract_best_career_sentence(career, matched_concepts)
    if best_sentence:
        parts.append(best_sentence)
    else:
        company_str = f" at {company}" if company else ""
        parts.append(f"{yoe:.1f}-yr {title}{company_str}")

    if matched_concepts:
        parts.append(f"Covers: {', '.join(matched_concepts[:5])}")
    else:
        parts.append("No core JD concepts matched in career text")

    critical_concepts = {"vector_db", "retrieval_and_search", "ranking_systems", "evaluation"}
    missing_critical = [c for c in critical_concepts if c not in matched_concepts]
    if missing_critical:
        parts.append(f"Gaps: {', '.join(missing_critical)}")

    parts.append(f"Technical fit: {core_fit:.2f}")

    parts.append(
        f"Chain: vis={hiring_chain[0]:.2f} resp={hiring_chain[1]:.2f} "
        f"int={hiring_chain[2]:.2f} acc={hiring_chain[3]:.2f}"
    )

    if cap_breakdown:
        sorted_caps = sorted(cap_breakdown.items(), key=lambda x: -x[1])
        strong = [k for k,v in sorted_caps[:2] if v > 0.55]
        gaps   = [k for k,v in sorted_caps if v < 0.30][:2]
        if strong:
            parts.append(f"Strong: {'; '.join(strong)}")
        if gaps:
            parts.append(f"Capability gaps: {'; '.join(gaps)}")

    if rr is not None and rr != -1:
        rr_pct = int(float(rr) * 100)
        if rr_pct >= 70:
            parts.append(f"High recruiter response ({rr_pct}%)")
        elif rr_pct >= 40:
            parts.append(f"Moderate response ({rr_pct}%)")
        else:
            parts.append(f"Low response ({rr_pct}%) -- outreach risk")
    if notice <= 30:
        parts.append(f"available in {notice}d")
    elif notice > 90:
        parts.append(f"{notice}d notice -- friction")
    if location:
        parts.append(f"based in {location}")
    if last:
        try:
            la = datetime.strptime(last, "%Y-%m-%d").date()
            days = (REFERENCE_DATE - la).days
            if days <= 7:
                parts.append("active this week")
            elif days <= 30:
                parts.append(f"active {days}d ago")
            elif days > 120:
                parts.append(f"last active {days}d ago -- stale")
        except Exception:
            pass

    if neg_score >= 0.35 and neg_reasons:
        parts.append(f"FLAG: {neg_reasons[0]}")

    parts.append(f"Score: {final_score:.4f}")

    result = ". ".join(parts) + "."
    if len(result) > 500:
        result = result[:497] + "..."
    return result


# ── STRUCTURAL SCORING (OPTIMISED) ──────────────────────────────────────

def structural_score_optimised(candidate) -> dict:
    cid = candidate.get("candidate_id", "UNKNOWN")
    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}
    assessments = signals.get("skill_assessment_scores", {}) or {}

    # 1. Honeypot (fast)
    honeypot, hp_reason = is_honeypot(candidate)
    if honeypot:
        return {
            "candidate_id": cid,
            "score": 0.0001,
            "reasoning": f"Honeypot: {hp_reason}.",
            "_dbg": None,
            "candidate_data": candidate,
        }

    # 2. Title (fast)
    t_score = get_title_score(profile.get("current_title", ""))
    if t_score == 0.0:
        return {
            "candidate_id": cid,
            "score": 0.005,
            "reasoning": f"Disqualified title: {profile.get('current_title', '')}",
            "_dbg": None,
            "candidate_data": candidate,
        }

    # 3. CV/Speech gate (fast)
    cv_speech_primary = is_cv_speech_primary(career, skills)

    # 4. Hard gates (fast, using sets)
    has_prod = check_production_deployment_fast(career, skills)
    has_retrieval = check_retrieval_experience_fast(career, skills, cv_speech_primary)
    has_tech_depth = check_technical_depth_fast(career)
    gates_passed = sum([has_prod, has_retrieval, has_tech_depth])

    if gates_passed == 0:
        gate_mult = 0.15
    elif gates_passed == 1:
        gate_mult = 0.50
    elif gates_passed == 2:
        gate_mult = 0.80
    else:
        gate_mult = 1.00

    # 5. Core fit — only compute heavy concept scoring if gates_passed > 0
    if gates_passed > 0:
        all_career_text = " ".join(
            (j.get("description", "") or "") + " " + (j.get("title", "") or "")
            for j in career
        )[:2000]  # reduced length
        concept_score, matched_concepts = contextualized_concept_score_optimized(all_career_text, max_len=2000)
        # v7: skill trust now decays per-skill based on career-traced recency
        skill_score = get_skill_trust_score(skills, assessments, career, REFERENCE_DATE)
        momentum = career_momentum_score_fast(career)  # also uses limited regex
    else:
        # If no gates passed, assign low scores without heavy computation
        concept_score, matched_concepts = 0.0, []
        skill_score = 0.0
        momentum = 0.0

    positive_fit = (
        0.50 * concept_score +
        0.22 * skill_score  +
        0.20 * momentum
    )
    core_fit = positive_fit * (0.40 + 0.60 * t_score)

    # ── Causal hiring chain ──
    p_visibility = recruiter_visibility_score(signals)
    p_respond = float(signals.get("recruiter_response_rate") or 0.5)
    p_interview = float(signals.get("interview_completion_rate") or 0.5)
    p_accept = float(signals.get("offer_acceptance_rate") or 0.5)
    p_respond = max(0.01, min(0.99, p_respond))
    p_interview = max(0.01, min(0.99, p_interview))
    p_accept = max(0.01, min(0.99, p_accept))
    hiring_prob = p_visibility * p_respond * p_interview * p_accept
    hiring_mult = 0.40 + (hiring_prob / 0.30) * 0.80
    # FIX (v7): cap restored to 1.00 — behavioral signals can no longer
    # inflate a candidate's score above what their technical fit earned.
    hiring_mult = max(0.35, min(1.00, hiring_mult))

    notice_score = get_notice_score(signals)
    availability_mult = 0.92 * hiring_mult + 0.08 * notice_score

    loc_mult = get_location_multiplier(profile, signals)
    neg_score, neg_reasons = get_negative_signal_score(candidate)
    neg_mult = 1.0 - neg_score

    # ── Final score ──────────────────────────────────────────
    raw = (core_fit
           * gate_mult
           * availability_mult
           * loc_mult
           * neg_mult)

    if raw > 0.5:
        raw = 0.5 + (raw - 0.5) * 1.3
    # FIX (v7): exponent restored to 0.65 — 0.75 over-compressed the top end
    # and reduced score spread across the leaderboard.
    raw = math.pow(max(raw, 0.001), 0.65)

    yoe = float(profile.get("years_of_experience", 0) or 0)
    yoe_gate = get_yoe_gate(yoe)
    notice_days = int(signals.get("notice_period_days") or 60)
    notice_gate = get_notice_gate(notice_days)
    cv_gate = get_cv_speech_mismatch_gate(cv_speech_primary)
    foreign_gate = get_foreign_gate(profile, signals)
    raw = raw * yoe_gate * notice_gate * cv_gate * foreign_gate

    prestige_bonus = get_company_prestige_bonus(career)
    final = max(0.0001, min(0.9999, raw * (1.0 + prestige_bonus)))

    # ── Reasoning ─────────────────────────────────────────────
    reasoning = build_reasoning(
        candidate, matched_concepts, neg_score, neg_reasons,
        core_fit, (p_visibility, p_respond, p_interview, p_accept),
        final,
        cap_breakdown=None
    )

    return {
        "candidate_id": cid,
        "score": final,
        "reasoning": reasoning,
        "_dbg": {
            "title": profile.get("current_title", ""),
            "t_score": t_score,
            "concept": concept_score,
            "skill": skill_score,
            "momentum": momentum,
            "yoe_gate": yoe_gate,
            "notice_gate": notice_gate,
            "cv_gate": cv_gate,
            "foreign_gate": foreign_gate,
            "positive_fit": positive_fit,
            "core_fit": core_fit,
            "gate_mult": gate_mult,
            "hiring_mult": hiring_mult,
            "notice_score": notice_score,
            "neg": neg_score,
            "loc": loc_mult,
            "final": final,
            "cv_speech_primary": cv_speech_primary,
        },
        "candidate_data": candidate,
    }


# ── PARETO FRONTIER TAGGING (NEW in v7) ─────────────────────────────────

def tag_pareto_frontier(rows) -> set:
    """
    Identify candidates on the Pareto frontier of (technical fit, availability)
    within the final top-100. A candidate is on the frontier if no other
    top-100 candidate beats them on BOTH dimensions simultaneously.

    This is informational only — it does NOT change scores or ordering.
    It surfaces, e.g., "best fit but slow notice period" vs "great fit AND
    available now" so a recruiter can see the actual trade-off space instead
    of a single collapsed number. Mutates `result["reasoning"]` in place.
    """
    points = []
    for s, cid, result in rows:
        dbg = result.get("_dbg") or {}
        fit = dbg.get("core_fit", 0.0)
        avail = dbg.get("notice_score", 0.0)
        points.append((fit, avail, cid))

    frontier_ids = set()
    for fit_a, avail_a, cid_a in points:
        dominated = False
        for fit_b, avail_b, cid_b in points:
            if cid_b == cid_a:
                continue
            if fit_b >= fit_a and avail_b >= avail_a and (fit_b > fit_a or avail_b > avail_a):
                dominated = True
                break
        if not dominated:
            frontier_ids.add(cid_a)

    note = " Pareto-optimal: no other top candidate beats both fit and availability."
    for s, cid, result in rows:
        if cid in frontier_ids:
            base = result["reasoning"]
            combined = base.rstrip(".") + "." + note
            if len(combined) > 500:
                combined = base[: 500 - len(note) - 1].rstrip(".") + "." + note
            result["reasoning"] = combined

    return frontier_ids


# ─────────────────────────────────────────────
# MAIN TWO-PASS ORCHESTRATOR (with tie-break fix)
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob Ranker v12 — Production-Ready")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--sample", type=int, default=0,
                        help="Process only first N candidates (0=all)")
    args = parser.parse_args()

    path = Path(args.candidates)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    start_time = datetime.now()
    print(f"Redrob Ranker v12 starting on {path}")
    print(f"Output: {args.out}")
    if args.sample > 0:
        print(f"SAMPLE MODE: first {args.sample} candidates only")

    heap = []
    HEAP_SIZE = 500
    processed = 0
    honeypot_cnt = 0
    disq_cnt = 0
    error_cnt = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                error_cnt += 1
                continue

            result = structural_score_optimised(candidate)
            processed += 1

            sc = result["score"]
            if sc <= 0.0002:
                honeypot_cnt += 1
            elif sc <= 0.006:
                disq_cnt += 1

            if DEBUG and processed <= 5:
                dbg = result.get("_dbg")
                if dbg:
                    print(
                        f"DEBUG {result['candidate_id']}: "
                        f"title={dbg['title'][:28]} "
                        f"t={dbg['t_score']:.2f} "
                        f"concept={dbg['concept']:.3f} "
                        f"skill={dbg['skill']:.3f} "
                        f"mom={dbg['momentum']:.3f} "
                        f"core={dbg['core_fit']:.3f} "
                        f"gate={dbg['gate_mult']:.2f} "
                        f"neg={dbg['neg']:.3f} "
                        f"loc={dbg['loc']:.2f} "
                        f"yoe_gate={dbg['yoe_gate']:.3f} "
                        f"FINAL={result['score']:.4f}"
                    )

            entry = (result["score"], result["candidate_id"], result)
            if len(heap) < HEAP_SIZE:
                heapq.heappush(heap, entry)
            elif result["score"] > heap[0][0]:
                heapq.heapreplace(heap, entry)

            if args.sample > 0 and processed >= args.sample:
                break

            if processed % 10000 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"  Pass 1: Processed {processed:,} candidates ({elapsed:.0f}s)...")

    print(f"\nPass 1 done. Processed={processed:,} Honeypots={honeypot_cnt} Disqualified={disq_cnt} Errors={error_cnt}")

    # ── Pass 2: Semantic re-ranking (same as before) ────────────────────────
    top500_entries = sorted(heap, key=lambda x: -x[0])[:500]
    print(f"Pass 2: Semantic re-ranking on {len(top500_entries)} top candidates...")

    needed_ids = {entry[1] for entry in top500_entries}
    candidate_data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            if c["candidate_id"] in needed_ids:
                candidate_data[c["candidate_id"]] = c
                if len(candidate_data) == len(needed_ids):
                    break

    cap_breakdowns = {}
    if SEMANTIC_AVAILABLE and len(top500_entries) > 0:
        print("Loading semantic model...")
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            cap_embeddings = model.encode(CAPABILITIES, normalize_embeddings=True, show_progress_bar=False)
            print("Encoding top candidates...")
            texts = []
            cids = []
            for score, cid, result in top500_entries:
                cand = candidate_data.get(cid, {})
                career = cand.get("career_history", []) or []
                text = " ".join(j.get("description", "") for j in career)[:1500]
                if not text.strip():
                    profile = cand.get("profile", {})
                    text = (profile.get("summary", "") or "") + " " + (profile.get("headline", "") or "")
                    text = text[:1500]
                texts.append(text)
                cids.append(cid)

            embeddings = model.encode(
                texts,
                batch_size=512,
                normalize_embeddings=True,
                show_progress_bar=True
            )

            for idx, (score, cid, result) in enumerate(top500_entries):
                cand_emb = embeddings[idx]
                cap_scores = {}
                total = 0.0
                for i, cap in enumerate(CAPABILITIES):
                    sim = float(np.dot(cand_emb, cap_embeddings[i]))
                    sim = max(0.0, sim)
                    cap_scores[cap[:35]] = round(sim, 3)
                    total += sim
                cap_score = total / len(CAPABILITIES)
                cap_breakdowns[cid] = (cap_score, cap_scores)

            for i, (score, cid, result) in enumerate(top500_entries):
                if cid in cap_breakdowns:
                    cap_score, cap_scores = cap_breakdowns[cid]
                    new_score = 0.65 * score + 0.35 * cap_score
                    result["score"] = max(0.0001, min(0.9999, new_score))
                    # Update reasoning with capability info
                    existing = result["reasoning"]
                    sorted_caps = sorted(cap_scores.items(), key=lambda x: -x[1])
                    strong = [k for k,v in sorted_caps[:2] if v > 0.55]
                    gaps   = [k for k,v in sorted_caps if v < 0.30][:2]
                    extra = []
                    if strong:
                        extra.append(f"Strong: {'; '.join(strong)}")
                    if gaps:
                        extra.append(f"Capability gaps: {'; '.join(gaps)}")
                    if extra:
                        parts = existing.split(". ")
                        new_parts = []
                        inserted = False
                        for p in parts:
                            new_parts.append(p)
                            if not inserted and ("Chain:" in p or "Score:" in p):
                                new_parts.extend(extra)
                                inserted = True
                        if not inserted:
                            new_parts.extend(extra)
                        result["reasoning"] = ". ".join(new_parts) + "."

            print("Semantic re-ranking complete.")
        except Exception as e:
            print(f"WARNING: Semantic re-ranking failed: {e}")
    else:
        if not SEMANTIC_AVAILABLE:
            print("Semantic layer not available. Skipping re-ranking.")

    # ── FINAL OUTPUT WITH TIE-BREAK FIX ────────────────────────────────────
    # Build rows with rounded scores and sort by (-rounded_score, candidate_id)
    rows = []
    for score, cid, result in top500_entries:
        rounded = round(score, 4)
        rows.append((rounded, cid, result))
    # Sort: descending rounded score, ascending candidate_id for ties
    rows.sort(key=lambda x: (-x[0], x[1]))
    top100 = rows[:100]

    # v7: tag Pareto-optimal candidates (informational only, doesn't reorder)
    frontier_ids = tag_pareto_frontier(top100)

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        seen_reasoning = set()
        for rank_idx, (s, cid, result) in enumerate(top100):
            # Since rows are already sorted by (-s, cid), we guarantee non-increasing s
            # and ascending cid for ties. No need to adjust scores further.
            reasoning = result["reasoning"]
            if reasoning in seen_reasoning:
                reasoning = reasoning.rstrip(".") + f" [uid:{cid}]."
            seen_reasoning.add(reasoning)
            writer.writerow([cid, rank_idx+1, f"{s:.4f}", reasoning])

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nSubmission written: {out_path}")
    print(f"Score range: {top100[-1][0]:.4f} to {top100[0][0]:.4f}  "
          f"(spread={top100[0][0]-top100[-1][0]:.4f})")
    print(f"Pareto-optimal candidates in top 100: {len(frontier_ids)}")
    print(f"Runtime: {elapsed:.1f} seconds")
    print(f"\nTop 10 candidates:")
    for i, (s, cid, _) in enumerate(top100[:10], 1):
        print(f"  #{i:2d} {cid} score={s:.4f}")


if __name__ == "__main__":
    main()
