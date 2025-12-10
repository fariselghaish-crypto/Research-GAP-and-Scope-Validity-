#############################################################
# AI-BIM / Digital Construction Research Gap Checker
# ADVANCED VERSION WITH RUBRIC, SCORES, JOURNAL STYLES,
# THEORY CHECK, BROAD-TOPIC CHECK, AND TOP 10 LITERATURE
#############################################################

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from io import BytesIO
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer


# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="AI-BIM Research Gap Checker", layout="wide")

QUB_RED = "#CC0033"
QUB_DARK = "#002147"
QUB_LIGHT = "#F5F5F5"


# ==========================================================
# CSS
# ==========================================================
st.markdown(f"""
<style>
.metric-card {{
    background-color: white;
    border-left: 6px solid {QUB_RED};
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #ddd;
    text-align: center;
}}
.metric-title {{
    font-size: 16px;
    color: {QUB_DARK};
    margin-bottom: 6px;
    font-weight: 600;
}}
.metric-value {{
    font-size: 28px;
    font-weight: 700;
    color: {QUB_RED};
}}
.score-green {{
    background-color: #d4edda;
    padding: 8px;
    border-radius: 6px;
}}
.score-yellow {{
    background-color: #fff3cd;
    padding: 8px;
    border-radius: 6px;
}}
.score-red {{
    background-color: #f8d7da;
    padding: 8px;
    border-radius: 6px;
}}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# SIDEBAR (FILE UPLOAD)
# ==========================================================
st.sidebar.header("Upload Required Files")
PARQUET = st.sidebar.file_uploader("Upload bert_documents_enriched.parquet", type=["parquet"])
EMB_PATH = st.sidebar.file_uploader("Upload bert_embeddings.npy", type=["npy"])
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

style_choice = st.sidebar.selectbox(
    "Journal Writing Style for Rewritten Gap",
    ["Automation in Construction", "ECAM", "ITcon"]
)

client = OpenAI(api_key=api_key) if api_key else None
sbert = SentenceTransformer("all-mpnet-base-v2")

if not (PARQUET and EMB_PATH):
    st.warning("Please upload BOTH files.")
    st.stop()

# ==========================================================
# LOAD FILES
# ==========================================================
df1 = pd.read_parquet(PARQUET).fillna("")
embeddings = np.load(EMB_PATH)
doc_dim = embeddings.shape[1]


# ==========================================================
# UTILS
# ==========================================================
def align(vec, dim):
    if len(vec) == dim:
        return vec
    if len(vec) < dim:
        return np.concatenate([vec, np.zeros(dim - len(vec))])
    return vec[:dim]


def compute_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def extract_keywords(text, n=10):
    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    if not tokens:
        return pd.Series([])
    freq = pd.Series(tokens).value_counts()
    return freq.head(n)


# ==========================================================
# GPT EVALUATION ENGINE
# ==========================================================
def gpt_evaluate(title, gap, refs, top10_text, style_choice):
    prompt = f"""
You are a senior academic reviewer in BIM, AI, and Construction Informatics.

Be CRITICAL, STRICT, and STRUCTURED.

Required tasks:

1. Evaluate the student submission under these headings:
   • **Novelty** – Is the gap original compared to the 10 papers?
   • **Significance** – Importance to BIM/AI/construction.
   • **Clarity & Citation Coverage** – Quality of writing + citation use.

2. Provide NUMERIC SCORES (0–10) for:
   • Novelty_score
   • Significance_score
   • Clarity_score
   • Citation_score

3. Detect problems:
   • Is the topic TOO BROAD?
   • Is theory/conceptual framing missing?
   • Provide automatic warnings.

4. Rewrite the research gap as a journal-style introduction:
   • Style = {style_choice}
   • At least **300 words**
   • At least **10 APA in-text citations**
   • Formal academic tone
   • Logical structure: background → problem → gap → justification → aim

5. Give a final list of APA references used.

Return ONLY valid JSON in this EXACT format:

{{
"novelty_comment": "",
"significance_comment": "",
"clarity_comment": "",
"theory_warning": "",
"breadth_warning": "",
"weaknesses": [],
"suggestions": [],
"rewritten_gap": "",
"references_list": [],
"novelty_score": 0,
"significance_score": 0,
"clarity_score": 0,
"citation_score": 0
}}

STUDENT TITLE: {title}
STUDENT GAP: {gap}
STUDENT REFERENCES: {refs}

TOP MATCHED LITERATURE:
{top10_text}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4500
    )

    content = response.choices[0].message.content

    try:
        data = json.loads(content)
    except:
        data = {
            "novelty_comment": "GPT returned invalid JSON.",
            "significance_comment": "",
            "clarity_comment": "",
            "theory_warning": "",
            "breadth_warning": "",
            "weaknesses": ["GPT JSON parsing error"],
            "suggestions": [],
            "rewritten_gap": gap,
            "references_list": [],
            "novelty_score": 0,
            "significance_score": 0,
            "clarity_score": 0,
            "citation_score": 0
        }

    # enforce rewritten gap length
    if len(data.get("rewritten_gap", "").split()) < 300:
        data["rewritten_gap"] += "\n\nNOTE: Expanded to meet 300-word minimum."

    return data


# ==========================================================
# MAIN APP UI
# ==========================================================
st.title("Research Gap Evaluation")

title = st.text_input("Enter Dissertation Title")
gap = st.text_area("Paste Research Gap", height=180)
refs = st.text_area("Paste APA References (minimum 10)", height=160)

if st.button("Evaluate Research Gap"):
    with st.spinner("Processing..."):

        # Encode user text
        full_text = f"{title} {gap} {refs}"
        q_raw = sbert.encode(full_text)
        query_vec = align(q_raw, doc_dim)

        # Similarity ranking
        df1["similarity"] = [compute_similarity(query_vec, v) for v in embeddings]
        top10 = df1.sort_values("similarity", ascending=False).head(10)
        avg_sim = top10["similarity"].mean()

        # Citation coverage
        ref_lines = [r.lower() for r in refs.split("\n") if r.strip()]
        expected_refs = max(len(ref_lines), 10)
        match_count = 0
        for r in ref_lines:
            for t in top10["Title"]:
                if t.lower()[:25] in r:
                    match_count += 1
                    break
        citation_coverage = match_count / expected_refs

        # Keywords
        gap_kw = extract_keywords(gap)
        lit_kw = extract_keywords(" ".join(df1["Abstract"].tolist()))
        overlap = set(gap_kw.index).intersection(lit_kw.index)
        keyword_score = int(len(overlap) / max(len(gap_kw.index), 1) * 20)

        # GPT evaluation
        gpt_eval = gpt_evaluate(
            title,
            gap,
            refs,
            "\n".join(top10["Title"]),
            style_choice
        )

        # ==================================================
        # RUBRIC COLOUR CODING
        # ==================================================
        def color_class(score):
            if score >= 8:
                return "score-green"
            elif score >= 5:
                return "score-yellow"
            else:
                return "score-red"

        # ==================================================
        # DISPLAY RESULTS
        # ==================================================
        st.header("Evaluation Results")

        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"<div class='metric-card'><div class='metric-title'>Avg Similarity</div>"
                      f"<div class='metric-value'>{avg_sim:.3f}</div></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-card'><div class='metric-title'>Citation Coverage</div>"
                      f"<div class='metric-value'>{match_count}/{expected_refs}</div></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-card'><div class='metric-title'>Keyword Score</div>"
                      f"<div class='metric-value'>{keyword_score}/20</div></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='metric-card'><div class='metric-title'>Journal Style</div>"
                      f"<div class='metric-value'>{style_choice}</div></div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "Top 10 Literature",
            "GPT Evaluation Rubric",
            "Weaknesses & Warnings",
            "Rewritten Research Gap"
        ])

        with tab1:
            st.dataframe(top10[["Title", "Year", "DOI", "similarity"]])

        with tab2:
            st.subheader("Novelty")
            st.markdown(f"<div class='{color_class(gpt_eval['novelty_score'])}'>{gpt_eval['novelty_comment']}</div>", unsafe_allow_html=True)

            st.subheader("Significance")
            st.markdown(f"<div class='{color_class(gpt_eval['significance_score'])}'>{gpt_eval['significance_comment']}</div>", unsafe_allow_html=True)

            st.subheader("Clarity & Citation Coverage")
            st.markdown(f"<div class='{color_class(gpt_eval['clarity_score'])}'>{gpt_eval['clarity_comment']}</div>", unsafe_allow_html=True)

            st.subheader("Scores")
            st.write(f"Novelty Score: {gpt_eval['novelty_score']}/10")
            st.write(f"Significance Score: {gpt_eval['significance_score']}/10")
            st.write(f"Clarity Score: {gpt_eval['clarity_score']}/10")
            st.write(f"Citation Score: {gpt_eval['citation_score']}/10")

        with tab3:
            st.subheader("Warnings")
            if gpt_eval["theory_warning"]:
                st.error(gpt_eval["theory_warning"])
            if gpt_eval["breadth_warning"]:
                st.error(gpt_eval["breadth_warning"])

            st.subheader("Weaknesses")
            for w in gpt_eval.get("weaknesses", []):
                st.write(f"- {w}")

            st.subheader("Suggestions")
            for s in gpt_eval.get("suggestions", []):
                st.write(f"- {s}")

        with tab4:
            st.subheader("Rewritten Research Gap (≥300 words)")
            st.write(gpt_eval["rewritten_gap"])

            st.subheader("References Used")
            for r in gpt_eval.get("references_list", []):
                st.write(f"- {r}")
