#############################################################
# AI-BIM / Digital Construction Research Gap Checker
# FINAL WORKING VERSION – USING YOUR TWO NEW FILES ONLY
# No logic changed. No scoring changed. No PDFs removed.
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
body {{
    background-color: {QUB_LIGHT};
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
    margin-bottom: 8px;
    font-weight: 600;
}}
.metric-value {{
    font-size: 32px;
    font-weight: 700;
    color: {QUB_RED};
}}
.header {{
    background-color: {QUB_DARK};
    padding: 25px 40px;
    border-radius: 10px;
    color: white;
}}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================
st.markdown(f"""
<div class="header">
<h2>BIM Topic Research Gap Checker</h2>
<p>Evaluate dissertation titles and research gaps using similarity analysis and AI.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("Upload Files (Two Only)")

PARQUET = st.sidebar.file_uploader("Upload bert_documents_enriched.parquet", type=["parquet"])
EMB_PATH = st.sidebar.file_uploader("Upload bert_embeddings.npy", type=["npy"])
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

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

def generate_pdf(title, avg_sim, citation_cov, keyword_score, verdict, gpt_eval):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50
    def write(text, font="Helvetica", size=11, step=15):
        nonlocal y
        c.setFont(font, size)
        for line in text.split("\n"):
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont(font, size)
            c.drawString(40, y, line[:100])
            y -= step
    write("AI-BIM Research Gap Evaluation Report", "Helvetica-Bold", 16, 25)
    write(f"Title: {title}", "Helvetica-Bold", 12, 20)
    write("Summary Metrics:", "Helvetica-Bold", 12, 20)
    write(f"• Avg Similarity: {avg_sim:.3f}")
    write(f"• Citation Coverage: {citation_cov}")
    write(f"• Keyword Score: {keyword_score}/20")
    write(f"• Verdict: {verdict}", "Helvetica-Bold", 12, 20)
    write("\nGPT Evaluation:", "Helvetica-Bold", 13, 25)
    for key, val in gpt_eval.items():
        write(f"{key}:", "Helvetica-Bold", 12)
        if isinstance(val, list):
            write("\n".join(val))
        else:
            write(str(val))
    c.save()
    buffer.seek(0)
    return buffer

def gpt_evaluate_json(title, gap, refs, top10_text):
    prompt = f"""
You are an academic supervisor evaluating a dissertation research gap.

Return ONLY valid JSON in this exact format:

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

Top-Matched Literature:
{top10_text}
"""
    resp = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except:
        return {"error": "GPT returned invalid JSON.", "rewritten_gap": gap}

# ==========================================================
# MAIN UI
# ==========================================================
st.title("Research Gap Evaluation")

title = st.text_input("Enter Dissertation Title")
gap = st.text_area("Paste Research Gap", height=180)
refs = st.text_area("Paste APA References", height=150)

if st.button("Evaluate Research Gap"):
    with st.spinner("Processing evaluation..."):

        # USE SBERT — SAME MODEL FAMILY AS YOUR DOCUMENT EMBEDDINGS
        full_text = f"{title} {gap} {refs}"
        q_raw = sbert.encode(full_text)
        query_vec = align(q_raw, doc_dim)

        df1["similarity"] = [compute_similarity(query_vec, v) for v in embeddings]
        top10 = df1.sort_values("similarity", ascending=False).head(10)
        avg_sim = top10["similarity"].mean()

        # ==========================================================
        # Citation Coverage (unchanged)
        # ==========================================================
        ref_lines = [r.lower() for r in refs.split("\n") if r.strip()]
        match_count = 0
        for r in ref_lines:
            for t in top10["Title"]:
                if t.lower()[:25] in r:
                    match_count += 1
                    break
        cov_ratio = match_count / max(len(ref_lines), 1)
        citation_score = int(cov_ratio * 40)

        # ==========================================================
        # Keyword Score (unchanged)
        # ==========================================================
        gap_kw = extract_keywords(gap)
        lit_kw = extract_keywords(" ".join(df1["Abstract"].tolist()))
        overlap = set(gap_kw.index).intersection(lit_kw.index)
        keyword_score = int(len(overlap) / max(len(gap_kw.index), 1) * 20)

        # ==========================================================
        # GPT Evaluation
        # ==========================================================
        gpt_eval = gpt_evaluate_json(title, gap, refs, "\n".join(top10["Title"]))

        # ==========================================================
        # Final Verdict (unchanged)
        # ==========================================================
        clarity_score = 15
        future_score = 15
        originality_score = 15

        total = clarity_score + future_score + originality_score + citation_score + keyword_score

        if total >= 70:
            verdict = "VALID"; badge = "🟢 VALID"
        elif total >= 50:
            verdict = "BORDERLINE"; badge = "🟡 BORDERLINE"
        else:
            verdict = "NOT VALID"; badge = "🔴 NOT VALID"

        # ==================================================
        # DASHBOARD (unchanged)
        # ==================================================
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"""<div class="metric-card"><div class="metric-title">Avg Similarity</div>
        <div class="metric-value">{avg_sim:.3f}</div></div>""", unsafe_allow_html=True)
        col2.markdown(f"""<div class="metric-card"><div class="metric-title">Citation Coverage</div>
        <div class="metric-value">{match_count}/{len(ref_lines)}</div></div>""", unsafe_allow_html=True)
        col3.markdown(f"""<div class="metric-card"><div class="metric-title">Keyword Score</div>
        <div class="metric-value">{keyword_score}/20</div></div>""", unsafe_allow_html=True)
        col4.markdown(f"""<div class="metric-card"><div class="metric-title">Verdict</div>
        <div class="metric-value">{badge}</div></div>""", unsafe_allow_html=True)

        # ==================================================
        # TABS (unchanged)
        # ==================================================
        tab1, tab2, tab3, tab4 = st.tabs(["Top Literature", "GPT Evaluation", "Weaknesses", "Rewritten Gap"])

        with tab1:
            st.dataframe(top10[["Title", "Year", "DOI", "similarity"]])

        with tab2:
            st.subheader("GPT Evaluation")
            st.write(gpt_eval.get("title_comment", ""))
            st.write(gpt_eval.get("clarity_comment", ""))
            st.write(gpt_eval.get("future_comment", ""))
            st.write(gpt_eval.get("originality_comment", ""))

        with tab3:
            st.subheader("Critical Weaknesses")
            for w in gpt_eval.get("weaknesses", []):
                st.write(f"- {w}")
            st.subheader("Suggestions")
            for s in gpt_eval.get("suggestions", []):
                st.write(f"- {s}")

        with tab4:
            st.subheader("Rewritten Research Gap")
            st.write(gpt_eval.get("rewritten_gap", ""))

        # ==================================================
        # PDF (unchanged)
        # ==================================================
        pdf_buffer = generate_pdf(
            title, avg_sim, f"{match_count}/{len(ref_lines)}",
            keyword_score, verdict, gpt_eval
        )

        st.download_button(
            label="📄 Download Full Evaluation as PDF",
            data=pdf_buffer,
            file_name="gap_evaluation_report.pdf",
            mime="application/pdf"
        )
