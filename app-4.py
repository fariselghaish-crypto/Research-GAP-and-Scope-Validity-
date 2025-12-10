#############################################################
# AI-BIM / Digital Construction Research Gap Checker
# UPDATED VERSION – 25 PAPERS, CRITICAL RUBRIC, JOURNAL-STYLE GAP
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

# ==========================================================
# PDF GENERATOR
# ==========================================================
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

    write("\nGPT Evaluation Rubric:", "Helvetica-Bold", 13, 25)
    write("Novelty:", "Helvetica-Bold", 12)
    write(str(gpt_eval.get("novelty_comment", "")))
    write("Significance:", "Helvetica-Bold", 12)
    write(str(gpt_eval.get("significance_comment", "")))
    write("Clarity & Citation Coverage:", "Helvetica-Bold", 12)
    write(str(gpt_eval.get("clarity_citation_comment", "")))

    write("\nWeaknesses:", "Helvetica-Bold", 12)
    for w in gpt_eval.get("weaknesses", []):
        write(f"- {w}")

    write("\nSuggestions:", "Helvetica-Bold", 12)
    for s in gpt_eval.get("suggestions", []):
        write(f"- {s}")

    write("\nRewritten Research Gap:", "Helvetica-Bold", 12)
    write(gpt_eval.get("rewritten_gap", ""))

    write("\nReferences:", "Helvetica-Bold", 12)
    for r in gpt_eval.get("references_list", []):
        write(f"- {r}")

    c.save()
    buffer.seek(0)
    return buffer

# ==========================================================
# GPT JSON EVALUATOR (MORE CRITICAL, JOURNAL STYLE)
# ==========================================================
def gpt_evaluate_json(title, gap, refs, top_lit_text):
    prompt = f"""
You are an experienced academic supervisor and journal reviewer in construction informatics, BIM and AI.

Be CRITICAL and STRICT in your evaluation.

Your tasks:

1. Evaluate the student's research gap under THREE headings:
   - Novelty (originality of the gap compared to the literature)
   - Significance (importance for theory, practice and the BIM/AI community)
   - Clarity & Citation Coverage (clarity of problem framing, logical flow, and how well the gap is grounded in citations)

2. Rewrite the research gap:
   - At least 300 WORDS.
   - Written in the style of a JOURNAL PAPER introduction/gap section (formal, structured, analytical, academic tone).
   - Use at least 10 APA-style in-text citations (author, year).
   - The citations should be plausible and aligned with BIM / AI / digital construction topics.

3. Add a list of references:
   - Provide a list of all references used in the rewritten gap as APA-style reference strings.

Return ONLY valid JSON in this exact format:

{{
  "novelty_comment": "",
  "significance_comment": "",
  "clarity_citation_comment": "",
  "weaknesses": [],
  "suggestions": [],
  "rewritten_gap": "",
  "references_list": []
}}

Student Input:
Title: {title}
Research Gap: {gap}
References Provided by Student: {refs}

Top-Matched Literature (most relevant 25 papers):
{top_lit_text}
"""
    resp = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000
    )
    content = resp.choices[0].message.content

    try:
        parsed = json.loads(content)
    except:
        # Fallback in case of invalid JSON
        parsed = {
            "novelty_comment": "",
            "significance_comment": "",
            "clarity_citation_comment": "",
            "weaknesses": ["GPT returned invalid JSON."],
            "suggestions": [],
            "rewritten_gap": gap,
            "references_list": []
        }

    return parsed

# ==========================================================
# MAIN UI
# ==========================================================
st.title("Research Gap Evaluation")

title = st.text_input("Enter Dissertation Title")
gap = st.text_area("Paste Research Gap", height=180)
refs = st.text_area("Paste APA References", height=150)

if st.button("Evaluate Research Gap"):
    with st.spinner("Processing evaluation..."):

        full_text = f"{title} {gap} {refs}"
        q_raw = sbert.encode(full_text)
        query_vec = align(q_raw, doc_dim)

        # Use TOP 25 papers instead of 10
        df1["similarity"] = [compute_similarity(query_vec, v) for v in embeddings]
        top_n = 25
        top_lit = df1.sort_values("similarity", ascending=False).head(top_n)
        avg_sim = top_lit["similarity"].mean()

        # ==========================================================
        # Citation Coverage (minimum expected = 10 references)
        # ==========================================================
        ref_lines = [r.lower() for r in refs.split("\n") if r.strip()]
        expected_refs = max(len(ref_lines), 10)

        match_count = 0
        for r in ref_lines:
            for t in top_lit["Title"]:
                if t.lower()[:25] in r:
                    match_count += 1
                    break

        cov_ratio = match_count / expected_refs
        citation_score = int(cov_ratio * 40)

        # ==========================================================
        # Keyword Score (unchanged)
        # ==========================================================
        gap_kw = extract_keywords(gap)
        lit_kw = extract_keywords(" ".join(df1["Abstract"].tolist()))
        overlap = set(gap_kw.index).intersection(lit_kw.index)
        keyword_score = int(len(overlap) / max(len(gap_kw.index), 1) * 20)

        # ==========================================================
        # GPT Evaluation (CRITICAL RUBRIC + JOURNAL STYLE)
        # ==========================================================
        gpt_eval = gpt_evaluate_json(
            title,
            gap,
            refs,
            "\n".join(top_lit["Title"])
        )

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
        # DASHBOARD
        # ==================================================
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"""<div class="metric-card"><div class="metric-title">Avg Similarity</div>
        <div class="metric-value">{avg_sim:.3f}</div></div>""", unsafe_allow_html=True)
        col2.markdown(f"""<div class="metric-card"><div class="metric-title">Citation Coverage</div>
        <div class="metric-value">{match_count}/{expected_refs}</div></div>""", unsafe_allow_html=True)
        col3.markdown(f"""<div class="metric-card"><div class="metric-title">Keyword Score</div>
        <div class="metric-value">{keyword_score}/20</div></div>""", unsafe_allow_html=True)
        col4.markdown(f"""<div class="metric-card"><div class="metric-title">Verdict</div>
        <div class="metric-value">{badge}</div></div>""", unsafe_allow_html=True)

        # ==================================================
        # TABS
        # ==================================================
        tab1, tab2, tab3, tab4 = st.tabs([
            "Top Literature (Top 25)",
            "GPT Evaluation (Rubric)",
            "Weaknesses",
            "Rewritten Gap"
        ])

        with tab1:
            st.subheader("Top 25 Most Relevant Papers")
            st.dataframe(top_lit[["Title", "Year", "DOI", "similarity"]])

        with tab2:
            st.subheader("Rubric Evaluation")
            st.markdown("### Novelty")
            st.write(gpt_eval.get("novelty_comment", ""))
            st.markdown("### Significance")
            st.write(gpt_eval.get("significance_comment", ""))
            st.markdown("### Clarity & Citation Coverage")
            st.write(gpt_eval.get("clarity_citation_comment", ""))

        with tab3:
            st.subheader("Critical Weaknesses")
            for w in gpt_eval.get("weaknesses", []):
                st.write(f"- {w}")
            st.subheader("Suggestions")
            for s in gpt_eval.get("suggestions", []):
                st.write(f"- {s}")

        with tab4:
            st.subheader("Rewritten Research Gap (Journal-Style, ≥300 words, ≥10 citations)")
            st.write(gpt_eval.get("rewritten_gap", ""))
            st.subheader("References Used")
            for r in gpt_eval.get("references_list", []):
                st.write(f"- {r}")

        # ==================================================
        # PDF EXPORT
        # ==================================================
        pdf_buffer = generate_pdf(
            title,
            avg_sim,
            f"{match_count}/{expected_refs}",
            keyword_score,
            verdict,
            gpt_eval
        )

        st.download_button(
            label="📄 Download Full Evaluation as PDF",
            data=pdf_buffer,
            file_name="gap_evaluation_report.pdf",
            mime="application/pdf"
        )
