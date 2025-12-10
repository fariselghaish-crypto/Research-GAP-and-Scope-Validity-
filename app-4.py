#############################################################
# BIM / Digital Construction Research Gap Checker (BERT Version)
# FINAL STREAMLIT APP – DEC 2025
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
# PAGE CONFIG + BRANDING
# ==========================================================
st.set_page_config(
    page_title="BIM Topic Research Gap Checker",
    layout="wide",
    page_icon="📘"
)

QUB_RED = "#CC0033"
QUB_DARK = "#002147"
QUB_LIGHT = "#F5F5F5"

# ==========================================================
# GLOBAL CSS STYLE
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
<p>Evaluate dissertation titles and research gaps using BERT similarity and academic analysis.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("Upload Dataset")

CSV_PATH = st.sidebar.file_uploader(
    "Upload bert_documents_enriched.parquet",
    type=["parquet"]
)

api_key = st.sidebar.text_input("OpenAI API Key (optional for GPT comments)", type="password")
client = OpenAI(api_key=api_key) if api_key else None

if CSV_PATH is None:
    st.warning("Please upload the Parquet dataset.")
    st.stop()

df = pd.read_parquet(CSV_PATH).fillna("")

# Extract embeddings (already stored as arrays)
embeddings = np.vstack(df["embedding"].values)

# Load SAME BERT model used to create these embeddings
bert = SentenceTransformer("all-mpnet-base-v2")

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def compute_similarity(vec1, vec2):
    """Cosine similarity"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9)

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

    def write_line(text, font="Helvetica", size=11, step=15):
        nonlocal y
        c.setFont(font, size)
        for line in text.split("\n"):
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont(font, size)
            c.drawString(40, y, line[:100])
            y -= step

    write_line("BIM Topic Gap Evaluation Report", "Helvetica-Bold", 16, 25)
    write_line(f"Title: {title}", "Helvetica-Bold", 12, 20)
    write_line("Summary Metrics:", "Helvetica-Bold", 12, 20)
    write_line(f"• Avg Similarity: {avg_sim:.3f}")
    write_line(f"• Citation Coverage: {citation_cov}")
    write_line(f"• Keyword Score: {keyword_score}/20")
    write_line(f"• Verdict: {verdict}", "Helvetica-Bold", 12, 20)

    if gpt_eval:
        write_line("\nGPT Evaluation:", "Helvetica-Bold", 13, 25)
        write_line("Title Evaluation:", "Helvetica-Bold", 12)
        write_line(gpt_eval.get("title_comment", ""))
        write_line("\nClarity:", "Helvetica-Bold", 12)
        write_line(gpt_eval.get("clarity_comment", ""))
        write_line("\nFuture Contribution:", "Helvetica-Bold", 12)
        write_line(gpt_eval.get("future_comment", ""))
        write_line("\nOriginality:", "Helvetica-Bold", 12)
        write_line(gpt_eval.get("originality_comment", ""))
        write_line("\nWeaknesses:", "Helvetica-Bold", 12)
        write_line("\n".join(gpt_eval.get("weaknesses", [])))
        write_line("\nSuggestions:", "Helvetica-Bold", 12)
        write_line("\n".join(gpt_eval.get("suggestions", [])))
        write_line("\nRewritten Gap:", "Helvetica-Bold", 12)
        write_line(gpt_eval.get("rewritten_gap", ""))

    c.save()
    buffer.seek(0)
    return buffer

def gpt_evaluate_json(title, gap, refs, top10_text):
    if not client:
        return None

    prompt = f"""
You are an academic supervisor evaluating a dissertation research gap.

Return ONLY valid JSON:

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
        return {
            "title_comment": "Error parsing GPT response.",
            "clarity_comment": "Error.",
            "future_comment": "Error.",
            "originality_comment": "Error.",
            "weaknesses": [],
            "suggestions": [],
            "rewritten_gap": gap
        }

# ==========================================================
# MAIN UI
# ==========================================================
st.title("Research Gap Evaluation")

title_input = st.text_input("Enter Dissertation Title")
gap_input = st.text_area("Paste Research Gap", height=180)
refs_input = st.text_area("Paste APA References", height=150)

if st.button("Evaluate Research Gap"):

    with st.spinner("Embedding and evaluating..."):

        full_text = title_input + " " + gap_input + " " + refs_input

        # Embed query using SAME BERT model
        query_vec = bert.encode(full_text, convert_to_numpy=True)

        # Compute similarities
        df["similarity"] = [compute_similarity(query_vec, v) for v in embeddings]

        top10 = df.sort_values("similarity", ascending=False).head(10)
        avg_sim = top10["similarity"].mean()

        # ===== Citation Coverage =====
        ref_lines = [r.lower() for r in refs_input.split("\n") if r.strip()]
        match_count = 0
        for r in ref_lines:
            for t in top10["Title"]:
                if t.lower()[:25] in r:
                    match_count += 1
                    break
        citation_score = int((match_count / max(1, len(ref_lines))) * 40)

        # ===== Keyword Score =====
        gap_kw = extract_keywords(gap_input)
        lit_kw = extract_keywords(" ".join(df["Abstract"].tolist()))
        overlap = set(gap_kw.index).intersection(lit_kw.index)
        keyword_score = int(len(overlap) / max(1, len(gap_kw.index)) * 20)

        # ===== GPT Evaluation =====
        gpt_eval = gpt_evaluate_json(
            title_input, gap_input, refs_input,
            "\n".join(top10["Title"])
        )

        # ===== Final Score =====
        clarity_score = 15
        future_score = 15
        originality_score = 15

        total = clarity_score + future_score + originality_score + citation_score + keyword_score

        if total >= 70:
            verdict = "VALID"
            badge = "🟢 VALID"
        elif total >= 50:
            verdict = "BORDERLINE"
            badge = "🟡 BORDERLINE"
        else:
            verdict = "NOT VALID"
            badge = "🔴 NOT VALID"

        # ==================================================
        # DASHBOARD
        # ==================================================
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"""<div class="metric-card">
            <div class="metric-title">Avg Similarity</div>
            <div class="metric-value">{avg_sim:.3f}</div></div>""", unsafe_allow_html=True)
        col2.markdown(f"""<div class="metric-card">
            <div class="metric-title">Citation Coverage</div>
            <div class="metric-value">{match_count}/{len(ref_lines)}</div></div>""", unsafe_allow_html=True)
        col3.markdown(f"""<div class="metric-card">
            <div class="metric-title">Keyword Score</div>
            <div class="metric-value">{keyword_score}/20</div></div>""", unsafe_allow_html=True)
        col4.markdown(f"""<div class="metric-card">
            <div class="metric-title">Verdict</div>
            <div class="metric-value">{badge}</div></div>""", unsafe_allow_html=True)

        # ==================================================
        # TABS
        # ==================================================
        tab1, tab2, tab3, tab4 = st.tabs([
            "Top Literature",
            "GPT Evaluation",
            "Weaknesses",
            "Rewritten Gap"
        ])

        with tab1:
            st.dataframe(top10[["Title", "Year", "DOI", "similarity"]])

        with tab2:
            if gpt_eval:
                st.write("### Title")
                st.write(gpt_eval["title_comment"])
                st.write("### Clarity")
                st.write(gpt_eval["clarity_comment"])
                st.write("### Future Contribution")
                st.write(gpt_eval["future_comment"])
                st.write("### Originality")
                st.write(gpt_eval["originality_comment"])
            else:
                st.info("Add OpenAI key to enable GPT evaluation.")

        with tab3:
            if gpt_eval:
                st.subheader("Critical Weaknesses")
                for w in gpt_eval["weaknesses"]:
                    st.write(f"- {w}")
                st.subheader("Suggestions")
                for s in gpt_eval["suggestions"]:
                    st.write(f"- {s}")
            else:
                st.info("GPT evaluation disabled.")

        with tab4:
            st.subheader("Rewritten Research Gap")
            st.write(gpt_eval["rewritten_gap"] if gpt_eval else "GPT disabled.")

        # ==================================================
        # PDF DOWNLOAD
        # ==================================================
        pdf_buffer = generate_pdf(
            title_input, avg_sim,
            f"{match_count}/{len(ref_lines)}",
            keyword_score, verdict, gpt_eval
        )

        st.download_button(
            label="📄 Download Full Evaluation as PDF",
            data=pdf_buffer,
            file_name="gap_evaluation_report.pdf",
            mime="application/pdf"
        )
