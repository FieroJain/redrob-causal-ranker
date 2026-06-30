#!/usr/bin/env python3
"""
Redrob Intelligent Candidate Ranker — Live Demo
═══════════════════════════════════════════════
Run: streamlit run app.py

Demonstrates the architecture, not a generic dashboard:
- Live gate firing (watch a candidate get filtered in real time)
- Causal hiring chain breakdown per candidate
- Before/after: the keyword-stuffer trap vs. this system
- Career momentum visualization
- Iteration history (the audit-driven engineering story)

Requires:
    pip install -r requirements.txt

Files expected in the same folder:
    redrob_ranker_v12.py     (the ranker — exposes structural_score_optimised)
    submissionv12.csv        (or .xlsx — final ranked output)
    sample_submission.csv    (optional — naive baseline, for the "trap" page)
    candidates.jsonl         (optional — only needed for Live Scoring Demo)
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

try:
    import redrob_ranker_v12 as ranker
    RANKER_AVAILABLE = True
except ImportError:
    RANKER_AVAILABLE = False

st.set_page_config(
    page_title="Redrob — Causal Hiring Chain Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# STYLING — light, elegant, professional
# ─────────────────────────────────────────────────────────────
NAVY = "#1E2761"
AMBER = "#F2A93B"
SLATE = "#5B6B8C"
ICE = "#EEF1FA"
GREEN = "#1E8E5A"
RED = "#B23A48"

st.markdown(f"""
<style>
    .stApp {{
        background-color: #FAFBFD;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {NAVY};
    }}
    section[data-testid="stSidebar"] * {{
        color: #EEF1FA !important;
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        color: #EEF1FA !important;
    }}
    h1, h2, h3 {{
        color: {NAVY} !important;
        font-family: "Georgia", serif;
    }}
    p, span, div, label {{
        color: #2A2F3A;
    }}
    .metric-card {{
        background: #FFFFFF;
        border: 1px solid #E3E7F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(30, 39, 97, 0.06);
    }}
    .big-number {{
        font-size: 2.1rem;
        font-weight: 700;
        color: {NAVY};
    }}
    .pill-pass {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        background: #E4F3EA;
        color: {GREEN};
        font-weight: 600;
        font-size: 0.85rem;
    }}
    .pill-fail {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        background: #FBEAEA;
        color: {RED};
        font-weight: 600;
        font-size: 0.85rem;
    }}
    .pill-warn {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        background: #FDF2E0;
        color: {AMBER};
        font-weight: 600;
        font-size: 0.85rem;
    }}
    .card {{
        background: #FFFFFF;
        border: 1px solid #E3E7F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(30, 39, 97, 0.05);
    }}
    .card-bad {{
        background: #FFF8F8;
        border: 1px solid #F3D6D9;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
    }}
    .card-good {{
        background: #F6FBF8;
        border: 1px solid #CDEBDA;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
    }}
    .stButton > button {{
        background-color: {NAVY};
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }}
    .stButton > button:hover {{
        background-color: #2A3680;
        color: white;
    }}
    .stProgress > div > div > div {{
        background-color: {AMBER};
    }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING (cached) — with fallback
# ─────────────────────────────────────────────────────────────

@st.cache_data
def load_candidates_jsonl(path: str, limit: int = 100_000):
    """Load candidates from JSONL — returns empty list if file missing or too large."""
    if not Path(path).exists():
        return []
    try:
        file_size = Path(path).stat().st_size
        if file_size > 50_000_000:  # >50 MB — skip for demo
            st.warning(f"⚠️ {path} is {file_size/1e6:.1f} MB — too large for demo. Using sample data only.")
            return []
        candidates = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
        return candidates
    except Exception as e:
        st.warning(f"⚠️ Could not load {path}: {e}")
        return []


@st.cache_data
def load_submission(path: str):
    if path.endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)


@st.cache_data
def build_candidate_lookup(_candidates):
    return {c["candidate_id"]: c for c in _candidates}


# ─────────────────────────────────────────────────────────────
# FALLBACK: Create minimal profiles from submission CSV
# ─────────────────────────────────────────────────────────────
def create_minimal_lookup_from_submission(df: pd.DataFrame) -> dict:
    """Create a minimal candidate lookup from submission CSV for Deep Dive."""
    lookup = {}
    for _, row in df.iterrows():
        cid = row['candidate_id']
        # Parse reasoning to extract title and location if possible
        reasoning = row.get('reasoning', '')
        parts = reasoning.split(' — ')
        if len(parts) > 1:
            title_parts = parts[0].split(',')
            title = title_parts[0].strip() if title_parts else 'Unknown'
        else:
            title = reasoning[:60] if reasoning else 'Unknown'
        
        # Guess location from reasoning
        location = 'India'
        if 'based in' in reasoning.lower():
            loc_start = reasoning.lower().find('based in')
            loc_end = reasoning.find('.', loc_start)
            if loc_end > loc_start:
                location = reasoning[loc_start+8:loc_end].strip()
        
        lookup[cid] = {
            "candidate_id": cid,
            "profile": {
                "current_title": title,
                "location": location,
                "years_of_experience": 5,  # fallback
            },
            "redrob_signals": {},
            "career_history": [],
            "skills": [],
        }
    return lookup


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

st.sidebar.title("🎯 Redrob Ranker")
st.sidebar.markdown("**Causal Hiring Chain Architecture**")
st.sidebar.markdown("---")

candidates_path = st.sidebar.text_input("candidates.jsonl path", "candidates.jsonl")
submission_path = st.sidebar.text_input("Final submission path", "submissionv12.csv")
sample_sub_path = st.sidebar.text_input("Sample (baseline) submission path", "sample_submission.csv")

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "⚡ Live Scoring Demo",
        "🔍 Candidate Deep Dive",
        "⚠️ The Keyword Stuffer Trap",
        "📈 Career Momentum Visualizer",
        "🧩 Architecture & Iteration History",
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Redrob Hackathon Submission")
st.sidebar.caption("AI tools used: Claude (architecture review, code audit)")

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

data_loaded = False
candidates, lookup, submission_df, sample_df = [], {}, None, None
try:
    if submission_path and Path(submission_path).exists():
        submission_df = load_submission(submission_path)
    if sample_sub_path and Path(sample_sub_path).exists():
        sample_df = load_submission(sample_sub_path)
    
    # Try to load candidates.jsonl — but don't fail if missing
    if Path(candidates_path).exists():
        file_size = Path(candidates_path).stat().st_size
        if file_size > 50_000_000:
            st.sidebar.warning("candidates.jsonl is large. Using submission data only.")
        else:
            candidates = load_candidates_jsonl(candidates_path)
            lookup = build_candidate_lookup(candidates)
            data_loaded = True
    else:
        st.sidebar.info("ℹ️ candidates.jsonl not found. Using submission data only.")
    
    # If lookup is empty but we have submission_df, build minimal lookup from it
    if not lookup and submission_df is not None:
        lookup = create_minimal_lookup_from_submission(submission_df)
        data_loaded = True
        
except Exception as e:
    st.error(f"Error loading data: {e}")

# ─────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────

if page == "🏠 Overview":
    st.title("Redrob Intelligent Candidate Discovery Engine")
    st.markdown(
        "##### *“We don't ask: does this candidate match the JD? "
        "We ask: if we contact them today, what is the probability "
        "they become a hire?”*"
    )

    if submission_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Candidates Scored", f"{len(submission_df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Top Score", f"{submission_df['score'].max():.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            spread = submission_df['score'].max() - submission_df['score'].min()
            st.metric("Score Spread", f"{spread:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Runtime (full pass)", "~103 sec")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("####")
        st.subheader("Top 10 Candidates")

        top10 = submission_df.head(10).copy()
        for _, row in top10.iterrows():
            cand = lookup.get(row['candidate_id'], {})
            profile = cand.get('profile', {})
            title = profile.get('current_title', 'Unknown')
            loc = profile.get('location', '')
            
            st.markdown(
                f'<div class="card">'
                f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                f'<div><span style="color:{SLATE}; font-weight:700; font-size:1.05rem;">#{int(row["rank"])}</span>'
                f'&nbsp;&nbsp;<b>{title}</b><br>'
                f'<span style="color:{SLATE}; font-size:0.85rem;">{row["candidate_id"]} · {loc}</span></div>'
                f'<div class="big-number">{row["score"]:.4f}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("Point the sidebar paths to your data files to load the demo.")

    st.markdown("### Three Things This System Does Differently")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="card"><b>Hierarchical Gating</b><br><br>'
            'Hard requirements gate core fit — not averaged with it. '
            'A non-technical title with AI keywords stuffed into skills '
            'still scores near zero.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(
            '<div class="card"><b>Career Momentum</b><br><br>'
            'We score trajectory over time, not a snapshot. A rising AI '
            'trajectory outranks a flat consulting tenure with the same title.</div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            '<div class="card"><b>Temporal Skill Decay</b><br><br>'
            'Skills have half-lives. 2021 GPT-3 knowledge is discounted '
            'relative to 2025 QLoRA knowledge — modeled explicitly, not assumed.</div>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: LIVE SCORING DEMO — now works without full JSONL
# ─────────────────────────────────────────────────────────────

elif page == "⚡ Live Scoring Demo":
    st.title("Live Scoring — Watch The Gates Fire")
    st.markdown("Pick any candidate from the dataset and watch the full scoring pipeline execute step by step.")

    if submission_df is None:
        st.warning("Load your submission file first (sidebar).")
    elif not RANKER_AVAILABLE:
        st.error("Could not import redrob_ranker_v12.py. Make sure it's in the same folder as app.py.")
    else:
        # Use the candidate IDs from submission
        cid_options = submission_df['candidate_id'].tolist()[:50]  # first 50 for demo
        cid_input = st.selectbox("Select a candidate_id", cid_options, index=0)

        if st.button("▶  Run Scoring Pipeline", type="primary"):
            # Try to get full candidate data from lookup; fallback to minimal
            cand = lookup.get(cid_input)
            if not cand:
                # Create minimal candidate from submission row
                row = submission_df[submission_df['candidate_id'] == cid_input].iloc[0]
                cand = {
                    "candidate_id": cid_input,
                    "profile": {
                        "current_title": "ML Engineer",
                        "location": "India",
                        "years_of_experience": 5,
                    },
                    "career_history": [],
                    "skills": [],
                    "redrob_signals": {}
                }
                st.info("ℹ️ Using minimal candidate data (candidates.jsonl not loaded).")
            
            profile = cand.get("profile", {})
            career = cand.get("career_history", []) or []
            skills = cand.get("skills", []) or []
            signals = cand.get("redrob_signals", {})

            st.markdown(f"## {profile.get('current_title','Unknown')}")
            st.caption(
                f"{cid_input} · {profile.get('location','Unknown')} · "
                f"{profile.get('years_of_experience',0)} years experience"
            )

            progress = st.progress(0)
            status = st.empty()

            # Step 1: Honeypot
            status.markdown("**Step 1/6 — Honeypot detection…**")
            time.sleep(0.3)
            hp, hp_reason = ranker.is_honeypot(cand)
            progress.progress(15)
            if hp:
                st.markdown(f'<span class="pill-fail">HONEYPOT DETECTED</span>  {hp_reason}',
                            unsafe_allow_html=True)
                st.stop()
            else:
                st.markdown('<span class="pill-pass">PASSED</span>  Profile timeline is consistent',
                            unsafe_allow_html=True)

            # Step 2: Title gate
            status.markdown("**Step 2/6 — Title gate…**")
            time.sleep(0.3)
            t_score = ranker.get_title_score(profile.get("current_title", ""))
            progress.progress(30)
            if t_score == 0.0:
                st.markdown(
                    f'<span class="pill-fail">TITLE DISQUALIFIED</span>  '
                    f'"{profile.get("current_title")}" — non-technical role',
                    unsafe_allow_html=True)
                st.markdown("**Score capped at 0.005 regardless of any AI skill keywords listed.**")
                st.stop()
            else:
                st.markdown(
                    f'<span class="pill-pass">PASSED</span>  Title score: '
                    f'<b>{t_score:.2f}</b> — {profile.get("current_title")}',
                    unsafe_allow_html=True)

            # Step 3: CV/Speech gate
            status.markdown("**Step 3/6 — CV/Speech structural gate…**")
            time.sleep(0.3)
            cv_speech = ranker.is_cv_speech_primary(career, skills)
            progress.progress(45)
            if cv_speech:
                st.markdown(
                    '<span class="pill-warn">FLAGGED</span>  CV/Speech-primary without NLP/IR '
                    'evidence — retrieval gate requires stricter evidence (3 keywords instead of 2)',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<span class="pill-pass">PASSED</span>  Not CV/Speech-primary — '
                    'standard retrieval gate threshold applies',
                    unsafe_allow_html=True)

            # Step 4: Hard gates
            status.markdown("**Step 4/6 — Hard requirement gates…**")
            time.sleep(0.3)
            has_prod = ranker.check_production_deployment_fast(career, skills)
            has_retrieval = ranker.check_retrieval_experience_fast(career, skills, cv_speech)
            has_depth = ranker.check_technical_depth_fast(career)
            progress.progress(60)

            gc1, gc2, gc3 = st.columns(3)
            for col, label, passed in [
                (gc1, "Production Deployment", has_prod),
                (gc2, "Retrieval Experience", has_retrieval),
                (gc3, "Technical Depth", has_depth),
            ]:
                with col:
                    pill = '<span class="pill-pass">PASS</span>' if passed else '<span class="pill-fail">FAIL</span>'
                    st.markdown(f'<div class="card">{pill}<br><b>{label}</b></div>', unsafe_allow_html=True)

            gates_passed = sum([has_prod, has_retrieval, has_depth])
            gate_mult = {0: 0.15, 1: 0.50, 2: 0.80, 3: 1.00}[gates_passed]
            st.markdown(f"**Gates passed: {gates_passed}/3 → multiplier = {gate_mult}**")

            # Step 5: Full scoring
            status.markdown("**Step 5/6 — Computing full score…**")
            time.sleep(0.3)
            result = ranker.structural_score_optimised(cand)
            progress.progress(85)

            dbg = result.get("_dbg", {})
            if dbg:
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.metric("Concept Score", f"{dbg.get('concept',0):.3f}")
                    st.metric("Skill Score (decayed)", f"{dbg.get('skill',0):.3f}")
                with sc2:
                    st.metric("Career Momentum", f"{dbg.get('momentum',0):.3f}")
                    st.metric("Core Fit", f"{dbg.get('core_fit',0):.3f}")
                with sc3:
                    st.metric("YOE Gate", f"{dbg.get('yoe_gate',0):.2f}")
                    st.metric("Hiring Mult (capped 1.0)", f"{dbg.get('hiring_mult',0):.2f}")

            # Step 6: Final
            status.markdown("**Step 6/6 — Final score**")
            progress.progress(100)
            st.markdown("---")
            st.markdown(f'<div class="big-number">Final Score: {result["score"]:.4f}</div>',
                        unsafe_allow_html=True)
            st.markdown("**Reasoning:**")
            st.markdown(f'<div class="card">{result["reasoning"]}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: CANDIDATE DEEP DIVE — now works without full JSONL
# ─────────────────────────────────────────────────────────────

elif page == "🔍 Candidate Deep Dive":
    st.title("Candidate Deep Dive")

    if submission_df is not None:
        rank_choice = st.selectbox(
            "Select rank to inspect",
            options=submission_df['rank'].tolist(),
            format_func=lambda r: f"#{r} — {submission_df[submission_df['rank']==r]['candidate_id'].values[0]}"
        )
        row = submission_df[submission_df['rank'] == rank_choice].iloc[0]
        cand = lookup.get(row['candidate_id'], {})
        profile = cand.get('profile', {})
        signals = cand.get('redrob_signals', {})
        career = cand.get('career_history', []) or []

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"## {profile.get('current_title', 'Unknown')}")
            st.caption(f"{row['candidate_id']} · Rank #{int(row['rank'])} · Score {row['score']:.4f}")
            st.markdown("#### Why this candidate")
            st.markdown(f'<div class="card">{row["reasoning"]}</div>', unsafe_allow_html=True)

            if career:
                st.markdown("#### Career History")
                for job in career:
                    current_tag = ' <span class="pill-pass">CURRENT</span>' if job.get('is_current') else ""
                    st.markdown(
                        f'<div class="card"><b>{job.get("title","")}</b> at '
                        f'{job.get("company","")}{current_tag}<br>'
                        f'<span style="color:{SLATE}; font-size:0.85rem;">'
                        f'{job.get("start_date","")} → {job.get("end_date") or "present"} · '
                        f'{job.get("duration_months",0)} months</span><br><br>'
                        f'<i>{job.get("description","")}</i></div>',
                        unsafe_allow_html=True,
                    )

        with col2:
            st.markdown("#### Behavioral Signals")
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Response Rate", f"{signals.get('recruiter_response_rate',0)*100:.0f}%")
            st.metric("Notice Period", f"{signals.get('notice_period_days',0)}d")
            st.metric("Last Active", signals.get('last_active_date','—'))
            st.metric("Open to Work", "Yes" if signals.get('open_to_work_flag') else "No")
            st.metric("Years Experience", profile.get('years_of_experience',0))
            st.metric("Location", profile.get('location',''))
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Load your submission file in the sidebar first.")


# ─────────────────────────────────────────────────────────────
# PAGE: THE KEYWORD STUFFER TRAP
# ─────────────────────────────────────────────────────────────

elif page == "⚠️ The Keyword Stuffer Trap":
    st.title("The Trap A Naive System Falls Into")
    st.markdown(
        "This dataset includes candidates with non-technical titles and AI "
        "keywords stuffed into their skills section. A naive keyword-matcher "
        "ranks them highly. This system doesn't."
    )

    if sample_df is not None and submission_df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<h4 style="color:{RED};">✕ Naive keyword-matching baseline</h4>',
                        unsafe_allow_html=True)
            st.caption("sample_submission.csv — provided format reference")
            bad_top5 = sample_df.head(5)[['rank', 'candidate_id', 'reasoning']]
            for _, r in bad_top5.iterrows():
                st.markdown(
                    f'<div class="card-bad"><b>#{r["rank"]}</b> {r["candidate_id"]}<br>'
                    f'<span style="font-size:0.85rem;">{str(r["reasoning"])[:140]}…</span></div>',
                    unsafe_allow_html=True)

        with col2:
            st.markdown(f'<h4 style="color:{GREEN};">✓ This system</h4>', unsafe_allow_html=True)
            st.caption("submissionv12.csv — causal hiring chain architecture")
            good_top5 = submission_df.head(5)
            for _, r in good_top5.iterrows():
                cand = lookup.get(r['candidate_id'], {})
                title = cand.get('profile', {}).get('current_title', '')
                st.markdown(
                    f'<div class="card-good"><b>#{int(r["rank"])}</b> {r["candidate_id"]} — {title}<br>'
                    f'<span style="font-size:0.85rem;">{str(r["reasoning"])[:140]}…</span></div>',
                    unsafe_allow_html=True)
    else:
        st.warning("Load both sample_submission.csv and your final submission to see this comparison.")


# ─────────────────────────────────────────────────────────────
# PAGE: CAREER MOMENTUM VISUALIZER
# ─────────────────────────────────────────────────────────────

elif page == "📈 Career Momentum Visualizer":
    st.title("Career Momentum — Trajectory Over Snapshot")
    st.markdown("We don't score what a candidate IS. We score the direction they're moving.")

    if data_loaded and RANKER_AVAILABLE and candidates:
        cid_input = st.text_input("Candidate ID", "CAND_0018499", key="momentum_input")
        cand = lookup.get(cid_input)

        if cand:
            career = cand.get("career_history", []) or []
            if career:
                sorted_career = sorted(career, key=lambda j: j.get("start_date", "") or "")
                job_scores, labels = [], []
                for job in sorted_career:
                    text = (job.get("description", "") or "") + " " + (job.get("title", "") or "")
                    cs, _ = ranker.contextualized_concept_score_optimized(text, max_len=1500)
                    job_scores.append(cs)
                    labels.append(f"{job.get('title','')[:25]}<br>{job.get('start_date','')[:4]}")

                if job_scores:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=labels, y=job_scores,
                        mode='lines+markers',
                        line=dict(color=NAVY, width=3),
                        marker=dict(size=11, color=AMBER, line=dict(width=2, color=NAVY)),
                        fill='tozeroy',
                        fillcolor='rgba(30,39,97,0.08)',
                    ))
                    fig.update_layout(
                        title="AI/ML Concept Relevance Over Career Timeline",
                        xaxis_title="Job (chronological)",
                        yaxis_title="Concept Score",
                        template="plotly_white",
                        height=420,
                        font=dict(color="#2A2F3A"),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    momentum = ranker.career_momentum_score_fast(career)
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Computed Momentum Score", f"{momentum:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No career history found for this candidate.")
            else:
                st.info("This candidate has no career history in the dataset.")
        else:
            st.info("Candidate not found — try one from the Overview top-10 list.")
    else:
        st.warning("Load candidates.jsonl and ensure the ranker module is importable.")


# ─────────────────────────────────────────────────────────────
# PAGE: ARCHITECTURE & ITERATION HISTORY
# ─────────────────────────────────────────────────────────────

elif page == "🧩 Architecture & Iteration History":
    st.title("System Architecture")

    st.markdown("""
#### Two-Pass Hybrid Pipeline

**Pass 1 — Structural Scoring** (all candidates, ~80–85 seconds for 100,000)
1. Honeypot detection — impossible-timeline checks
2. Title gate — non-technical titles capped near zero, before any skill is read
3. CV/Speech structural gate — applied before the retrieval check runs
4. Three hard requirement gates — production, retrieval, technical depth
5. Core fit — concept matching + skill trust (with temporal decay) + career momentum
6. Causal hiring chain — capped at 1.00; behavioral signals can only penalize, never inflate

**Pass 2 — Semantic Re-ranking** (top 500 only, ~12 seconds)
- 8 capability statements embedded via sentence-transformers
- Blended with the structural score

#### The Formula
""")
    st.code(
        "P(hire) = P(recruiter finds them)\n"
        "        × P(candidate responds)\n"
        "        × P(interview completes)\n"
        "        × P(offer accepted)\n\n"
        "final_score = core_fit\n"
        "            × gate_multiplier\n"
        "            × hiring_chain_multiplier   (capped at 1.00)\n"
        "            × location_multiplier\n"
        "            × negative_signal_multiplier\n"
        "            × yoe_gate\n"
        "            × notice_gate\n"
        "            × cv_speech_gate\n"
        "            × foreign_gate",
        language="text",
    )

    st.markdown("""
#### Why No Component Can Inflate Above Fit

Every multiplier in this system is bounded at 1.00 or below, except the
company prestige bonus (max +8%). This is deliberate: a mediocre-fit
candidate with perfect behavioral signals cannot outrank a strong-fit
candidate with average signals. Fit sets the ceiling.
""")

    st.markdown("### Iteration History — Audited at Every Step")
    iteration_data = pd.DataFrame({
        "Version": ["submission.csv (early best)", "v7", "v8", "v9", "v10-experimental", "v12 (final)"],
        "NOTICE:120d": [19, 19, 1, 0, 1, 0],
        "CV-primary": [4, 2, 0, 0, 0, 0],
        "FOREIGN": [2, 2, 2, 2, 2, 0],
        "Spread": [0.6568, 0.5918, 0.6296, 0.6223, 0.6450, 0.6242],
    })
    st.dataframe(iteration_data, use_container_width=True, hide_index=True)
    st.caption(
        "Every version above was independently audited with a custom verifier "
        "(not the dashboard you're looking at) before being trusted. v10-experimental "
        "looked promising in isolation but regressed NOTICE:120d when audited — held back "
        "rather than shipped. v12 isolates only the one additional change (a foreign-candidate "
        "gate) that audited clean against v9."
    )