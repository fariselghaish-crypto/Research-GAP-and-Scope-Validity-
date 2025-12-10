# FULL UPDATED & FIXED AI-BIM / Research Gap Checker App
# (Corrected variable scope + 200-word rule + reference ranges + strict scoring)

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from io import BytesIO
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chardet

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(page_title="AI-BIM Research Gap Checker", layout="wide")
QUB_RED = "#CC0033"
QUB_DARK = "#002147"
QUB_LIGHT = "#F5F5F5"

# ------------------------------------------------------------------
# STYLING
# ------------------------------------------------------------------
st.markdown(f"""
<style>
.metric-card {{
    background-color: white;
    border-left: 6px solid {QUB_RED};
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #ddd;
    text-align: center;
}}
.metric-title {{
    font-size: 17px;
    color: {QUB_DARK};
    margin-bottom: 8px;
    font-weight: 600;
}}
.metric-value {{
    font-size: 28px;
    font-weight: 700;
    color: {QUB_RED};
}}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.header("Upload Files (Two Mandatory)")
PARQUET = st.sidebar.file_uploader("Upload Scopus CSV or PARQUET", type=["csv", "parquet"])
EMB_PATH = st.sidebar.file_uploader("Upload Embeddings (.npy)", type=["npy"])
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
client = OpenAI(api_key=api_key) if api_key else None
sbert = SentenceTransformer("all-mpnet-base-v2")

if not (PARQUET and EMB_PATH):
    st.warning("Please upload both files to proceed.")
    st.stop()

# ------------------------------------------------------------------
# LOAD DOCUMENTS
# ------------------------------------------------------------------
def load_csv_with_encoding(uploaded):
    raw = uploaded.read()
    enc = chardet.detect(raw)["encoding"]
    return pd.read_csv(BytesIO(raw), encoding=enc)

if PARQUET.name.endswith('.csv'):
    df1 = load_csv_with_encoding(PARQUET)
else:
    df1 = pd.read_parquet(PARQUET)

df1 = df1.fillna("")
embeddings = np.load(EMB_PATH)
vector_dim = embeddings.shape[1]

# ------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------
def compute_similarity(query_vec, emb_matrix):
    qn = np.linalg.norm(query_vec)
    dn = np.linalg.norm(emb_matrix, axis=1)
    return (emb_matrix @ query_vec) / (dn * qn + 1e-9)

def align_vector(vec, dim):
    if len(vec) == dim:
        return vec
    if len(vec) < dim:
        return np.concatenate([vec, np.zeros(dim - len(vec))])
    return vec[:dim]

# ------------------------------------------------------------------
# GPT REVIEW FUNCTION
# ------------------------------------------------------------------
def gpt_review(title, gap, refs, top10_titles):

    top10_text = "; ".join(top10_titles)

    prompt = f"""
You are a strict reviewer for Automation in Construction, ECAM, and ITcon.
Be critical, direct, and academically rigorous.

============================ SCORING RULES ============================
Novelty (0–10): How new relative to top literature.
Significance (0–10): Academic + industry impact.
Clarity (0–10): Structure, focus, precision.
Citation Quality (0–10): relevance + quantity (≥7 ideal).
======================================================================

OUTPUT MUST BE JSON ONLY:
{{
"novelty_score": 0,
"significance_score": 0,
"clarity_score": 0,
"citation_score": 0,
"novelty_rationale": "",
"significance_rationale": "",
"clarity_rationale": "",
"citation_rationale": "",
"good_points": [],
"improvements": [],
"novelty_comment": "",
"significance_comment": "",
"citation_comment": "",
"rewritten_gap": ""
}}

======================== REWRITE REQUIREMENT ========================
YOU MUST rewrite the research gap using AT LEAST **200 words**.
Aim for 220–260 words. If shorter than 200 words = fail.
Write in journal style.
====================================================================

Title: {title}
Gap: {gap}
References: {refs}
Top 10 Papers: {top10_text}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.0,
        max_tokens=2400,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content

    # Attempt JSON parse
    try:
        return json.loads(raw)
    except:
        try:
            cleaned = raw[raw.find("{"):raw.rfind("}")+1]
            return json.loads(cleaned)
        except:
            return {
                "novelty_score": 0,
                "significance_score": 0,
                "clarity_score": 0,
                "citation_score": 0,
                "novelty_rationale": "",
                "significance_rationale": "",
                "clarity_rationale": "",
                "citation_rationale": "",
                "good_points": [],
                "improvements": [],
                "novelty_comment": "",
                "significance_comment": "",
                "citation_comment": "",
                "rewritten_gap": gap
            }

# ------------------------------------------------------------------
# MAIN UI
# ------------------------------------------------------------------
st.title("AI-BIM / Research Gap Evaluation App (Updated)")

title = st.text_input("Enter Dissertation Title")
gap = st.text_area("Paste Research Gap", height=180)
refs = st.text_area("Paste References (1 per line)", height=150)

if st.button("Evaluate Research Gap"):
    with st.spinner("Evaluating..."):

        # Embed query
        full_text = f"{title} {gap} {refs}"
        raw_vec = sbert.encode(full_text)
        q_vec = align_vector(raw_vec, vector_dim)

        # Similarity
        sims = compute_similarity(q_vec, embeddings)
        df1["similarity"] = sims
        top10 = df1.sort_values("similarity", ascending=False).head(10)
        top10_titles = top10["Title"].tolist()
        avg_sim = top10["similarity"].mean()

        # GPT REVIEW
        gpt = gpt_review(title, gap, refs, top10_titles)

        rewritten = gpt.get("rewritten_gap", "")
        word_count = len(rewritten.split())

        # ---------------- LENGTH RULE -----------------
        if word_count >= 200:
            length_penalty = 0
            length_flag = "valid"
        elif 150 <= word_count < 200:
            length_penalty = 5
            length_flag = "borderline"
        else:
            length_penalty = 15
            length_flag = "invalid"

        # ---------------- REFERENCE RULE -----------------
        ref_list = [r for r in refs.split("\n")) if r.strip()]) if r.strip()]
        ref_count = len(ref_list)

        if ref_count >= 7:
            ref_penalty = 0
            ref_flag = "valid"
        elif 5 <= ref_count <= 6:
            ref_penalty = 5
            ref_flag = "borderline"
        else:
            ref_penalty = 15
            ref_flag = "invalid"

        # ---------------- TOTAL SCORE -----------------
        total_raw = (
            gpt["novelty_score"] +
            gpt["significance_score"] +
            gpt["clarity_score"] +
            gpt["citation_score"] -
            length_penalty -
            ref_penalty
        )

        total_score = max(0, min(40, total_raw))

        # ---------------- VERDICT -----------------
        if length_flag == "invalid" or ref_flag == "invalid":
            verdict = "❌ NOT VALID"
        elif total_score >= 30:
            verdict = "🟢 VALID"
        elif total_score >= 20:
            verdict = "🟡 BORDERLINE"
        else:
            verdict = "❌ NOT VALID"

        # ---------------- DISPLAY -----------------
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"<div class='metric-card'><div class='metric-title'>Avg Similarity</div><div class='metric-value'>{avg_sim:.3f}</div></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-card'><div class='metric-title'>Novelty</div><div class='metric-value'>{gpt['novelty_score']}/10</div></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-card'><div class='metric-title'>Significance</
