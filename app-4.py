#############################################################
# AI-BIM / Digital Construction Research Gap Checker
# Full QUB-Branded Streamlit App – December 2025 Edition
# Author: Dr Faris Elghaish (QUB)
#############################################################

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from openai import OpenAI

# ==========================================================
# PAGE CONFIGURATION + BRANDING
# ==========================================================
st.set_page_config(
    page_title="AI-BIM Research Gap Checker",
    layout="wide",
    page_icon="📘"
)

# QUB COLOUR PALETTE
QUB_RED = "#CC0033"
QUB_DARK = "#002147"
QUB_LIGHT = "#F5F5F5"
QUB_GREY = "#ECECEC"

# ==========================================================
# GLOBAL STYLE (CSS)
# ==========================================================
st.markdown(f"""
<style>

body {{
    background-color: {QUB_LIGHT};
}}

.block-container {{
    padding-top: 1rem;
    padding-bottom: 1rem;
    background-color: white;
    border-radius: 10px;
    border: 1px solid #ddd;
}}

.header-container {{
    background-color: {QUB_DARK};
    padding: 25px 40px;
    border-radius: 8px;
    margin-bottom: 25px;
    color: white;
}}

.header-title {{
    font-size: 32px;
    font-weight: 700;
}}

.header-subtitle {{
    font-size: 18px;
    opacity: 0.8;
}}

.metric-card {{
    background-color: white;
    border-left: 6px solid {QUB_RED};
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #ddd;
    text-align: center;
}}

.metric-title {{
    font-size: 18px;
    color: {QUB_DARK};
    margin-bottom: 10px;
    font-weight: 600;
}}

.metric-value {{
    font-size: 34px;
    font-weight: 700;
    color: {QUB_RED};
}}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER BANNER
# ==========================================================
st.markdown(f"""
<div class="header-container">
    <div class="header-title">AI-BIM / Digital Construction Research Gap Checker</div>
    <div class="header-subtitle">
        Intelligent validation of dissertation topics using semantic similarity,
        citation coverage, keyword alignment, and GPT-4.1 academic evaluation.
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR INPUTS
# ==========================================================
st.sidebar.header("Upload Your Data")

DATA_PATH_1 = st.sidebar.file_uploader(
    "Upload dt_construction_filtered_topics.csv", type=["csv"]
)

DATA_PATH_2 = st.sidebar.file_uploader(
    "Upload dt_topic_summary_reconstructed.csv", type=["csv"]
)

EMB_PATH = st.sidebar.file_uploader(
    "Upload embeddings.npy", type=["npy"]
)

api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
client = OpenAI(api_key=api_key) if api_key else None

# ==========================================================
# LOAD DATA
# ==========================================================
if DATA_PATH_1 and DATA_PATH_2 and EMB_PATH:
    df1 = pd.read_csv(DATA_PATH_1)
    df2 = pd.read_csv(DATA_PATH_2)
    embeddings = np.load(EMB_PATH)

    df1 = df1.fillna("")
    df2 = df2.fillna("")

    st.success("✓ Data loaded successfully.")
else:
    st.warning("Please upload all required files.")
    st.stop()

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def compute_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9)


def extract_keywords(text, n=10):
    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    if not tokens:
        return pd.Series([])
    freq = pd.Series(tokens).value_counts()
    return freq.head(n)


# ==========================================================
# GPT EVALUATION (JSON OUTPUT)
# ==========================================================
def gpt_evaluate_json(title, gap, refs, top10_text):
    prompt = f"""
You are an academic supervisor evaluating a dissertation research gap.

Return ONLY valid JSON. No explanations.

JSON FORMAT:
{{
  "title_comment": "",
  "clarity_comment": "",
  "future_comment": "",
  "originality_comment": "",
  "weaknesses": [],
  "suggestions": [],
  "rewritten_gap": ""
}}

Student Input:
Title: {title}
Research Gap: {gap}
References: {refs}

Top Matched Literature:
{top10_text}

Write precise, critical, academic feedback.
    """

    resp = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )

    raw = resp.choices[0].message.content

    try:
        parsed = json.loads(raw)
    except:
        parsed = {
            "title_comment": "Error reading JSON.",
            "clarity_comment": "Error.",
            "future_comment": "Error.",
            "originality_comment": "Error.",
            "weaknesses": [],
            "suggestions": [],
            "rewritten_gap": gap,
        }

    return parsed


# ==========================================================
# MAIN UI
# ==========================================================
st.title("Research Gap Evaluation")

title_input = st.text_input("Enter Dissertation Title")
gap_input = st.text_area("Paste Research Gap", height=180)
refs_input = st.text_area("Paste APA References", height=150)

if st.button("Evaluate Research Gap"):

    # ======================================================
    # EMBEDDINGS
    # ======================================================
    full_text = title_input + " " + gap_input + " " + refs_input
    embed = client.embeddings.create(
        model="text-embedding-3-small",
        input=[full_text]
    )
    query_vec = np.array(embed.data[0].embedding)

    # Compute similarity with df1 embeddings
    corpus_vecs = embeddings[: len(df1)]
    df1["similarity"] = [compute_similarity(query_vec, v) for v in corpus_vecs]

    top10 = df1.sort_values("similarity", ascending=False).head(10)
    avg_sim = top10["similarity"].mean()

    # ======================================================
    # OBJECTIVE CITATION COVERAGE
    # ======================================================
    ref_titles = [r.lower() for r in refs_input.split("\n") if r.strip()]
    match_count = 0

    for r in ref_titles:
        for t in top10["Title"]:
            if t.lower()[:25] in r:
                match_count += 1
                break

    total_refs = len(ref_titles)
    coverage_ratio = match_count / max(total_refs, 1)
    citation_score = int(coverage_ratio * 40)

    # ======================================================
    # KEYWORD SCORE
    # ======================================================
    gap_kw = extract_keywords(gap_input)
    lit_kw = extract_keywords(" ".join(df1["Abstract"].tolist()))

    overlap = set(gap_kw.index).intersection(lit_kw.index)
    keyword_ratio = len(overlap) / max(len(gap_kw.index), 1)
    keyword_score = int(keyword_ratio * 20)

    # ======================================================
    # GPT EVALUATION
    # ======================================================
    top10_text = "\n".join(top10["Title"].tolist())
    gpt_eval = gpt_evaluate_json(
        title_input, gap_input, refs_input, top10_text
    )

    # ======================================================
    # FINAL SCORING
    # ======================================================
    clarity_score = 15   # Fixed moderate score; can be replaced later
    future_score = 15    # Replace with GPT numeric if needed
    originality_score = 15

    total_score = clarity_score + citation_score + keyword_score + future_score + originality_score

    if total_score >= 70:
        verdict = "VALID"
        colour = "🟢"
    elif total_score >= 50:
        verdict = "BORDERLINE"
        colour = "🟡"
    else:
        verdict = "NOT VALID"
        colour = "🔴"

    # ======================================================
    # DASHBOARD METRICS
    # ======================================================
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Avg Similarity</div>
        <div class="metric-value">{avg_sim:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Citation Coverage</div>
        <div class="metric-value">{match_count}/{total_refs}</div>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Keyword Score</div>
        <div class="metric-value">{keyword_score}/20</div>
    </div>
    """, unsafe_allow_html=True)

    col4.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Validity Verdict</div>
        <div class="metric-value">{colour} {verdict}</div>
    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # TABS
    # ======================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "Top Literature",
        "GPT Evaluation",
        "Weaknesses",
        "Rewritten Research Gap"
    ])

    with tab1:
        st.subheader("Top 10 Relevant Papers")
        st.dataframe(top10[["Title", "Year", "DOI", "similarity"]])

    with tab2:
        st.subheader("GPT Evaluation")
        st.write("### Title")
        st.write(gpt_eval["title_comment"])
        st.write("### Clarity")
        st.write(gpt_eval["clarity_comment"])
        st.write("### Future Contribution")
        st.write(gpt_eval["future_comment"])
        st.write("### Originality")
        st.write(gpt_eval["originality_comment"])

    with tab3:
        st.subheader("Critical Weaknesses")
        for w in gpt_eval["weaknesses"]:
            st.write(f"- {w}")

        st.subheader("Suggestions")
        for s in gpt_eval["suggestions"]:
            st.write(f"- {s}")

    with tab4:
        st.subheader("Rewritten Research Gap (GPT-Improved)")
        st.write(gpt_eval["rewritten_gap"])
