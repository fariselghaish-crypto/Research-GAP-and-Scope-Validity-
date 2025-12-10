#############################################################
# BIM Topic Research Gap Checker
# Single-File Version (bert_documents_enriched.csv only)
# QUB Branding | Dec 2025
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

#############################
# PAGE CONFIG
#############################
st.set_page_config(
    page_title="BIM Topic Research Gap Checker",
    layout="wide",
    page_icon="📘"
)

QUB_RED = "#CC0033"
QUB_DARK = "#002147"
QUB_LIGHT = "#F5F5F5"

#############################
# GLOBAL CSS STYLE
#############################
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

#############################
# HEADER
#############################
st.markdown(f"""
<div class="header">
<h2>BIM Topic Research Gap Checker</h2>
<p>Evaluate dissertation topics using similarity search, GPT evaluation, and academic scoring.</p>
</div>
""", unsafe_allow_html=True)

#############################
# SIDEBAR
#############################
st.sidebar.header("Upload Dataset")

CSV_PATH = st.sidebar.file_uploader("Upload bert_documents_enriched.parquet", type=["parquet"])
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

client = OpenAI(api_key=api_key) if api_key else None

if CSV_PATH is None:
    st.warning("Please upload bert_documents_enriched.csv")
    st.stop()

df = pd.read_csv(CSV_PATH).fillna("")
embeddings = np.vstack(df["embedding"].apply(lambda x: np.array(eval(x))).values)

#############################
# HELPER FUNCTIONS
#############################
def compute_similarity(vec1, vec2):
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-9
    return float(np.dot(vec1, vec2) / denom)

def extract_keywords(text, n=10):
    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    if not tokens:
        return pd.Series([])
    return pd.Series(tokens).value_counts().head(n)

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

    write_line("BIM Research Gap Evaluation Report", "Helvetica-Bold", 16, 25)
    write_line(f"Title: {title}", "Helvetica-Bold", 12, 20)
    write_line("Summary Metrics:", "Helvetica-Bold", 12, 20)
    write_line(f"• Avg Similarity: {avg_sim:.3f}")
    write_line(f"• Citation Coverage: {citation_cov}")
    write_line(f"• Keyword Score: {keyword_score}/20")
    write_line(f"• Verdict: {verdict}", "Helvetica-Bold", 12, 20)

    write_line("\nGPT Evaluation:", "Helvetica-Bold", 13, 25)
    write_line("Title Evaluation:", "Helvetica-Bold", 12)
    write_line(gpt_eval["title_comment"])
    write_line("\nClarity:", "Helvetica-Bold", 12)
    write_line(gpt_eval["clarity_comment"])
    write_line("\nFuture Contribution:", "Helvetica-Bold", 12)
    write_line(gpt_eval["future_comment"])
    write_line("\nOriginality:", "Helvetica-Bold", 12)
    write_line(gpt_eval["originality_comment"])
    write_line("\nWeaknesses:", "Helvetica-Bold", 12)
    write_line("\n".join(gpt_eval["weaknesses"]))
    write_line("\nSuggestions:", "Helvetica-Bold", 12)
    write_line("\n".join(gpt_eval["suggestions"]))
    write_line("\nRewritten Gap:", "Helvetica-Bold", 12)
    write_line(gpt_eval["rewritten_gap"])

    c.save()
    buffer.seek(0)
    return buffer

def gpt_evaluate_json(title, gap, refs, top10_text):
    prompt = f"""
You are an expert academic supervisor. Return ONLY valid JSON:

{{
"title_comment": "",
"clarity_comment": "",
"future_comment": "",
"originality_comment": "",
"weaknesses": [],
"suggestions": [],
"rewritten_gap": ""
}}

Title: {title}
Research Gap: {gap}
References: {refs}

Top-Matched Literature:
{top10_text}
"""

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1500
    )

    txt = resp.choices[0].message.content
    try:
        return json.loads(txt)
    except:
        return {
            "title_comment": "JSON formatting error.",
            "clarity_comment": "Error.",
            "future_comment": "Error.",
            "originality_comment": "Error.",
            "weaknesses": [],
            "suggestions": [],
            "rewritten_gap": gap
        }

#############################
# MAIN UI
#############################
st.title("Research Gap Evaluation")

title_input = st.text_input("Enter Dissertation Title")
gap_input = st.text_area("Paste Research Gap", height=180)
refs_input = st.text_area("Paste APA References", height=150)

if st.button("Evaluate Research Gap"):
    with st.spinner("Evaluating..."):

        full_text = title_input + " " + gap_input + " " + refs_input
        q_embed = client.embeddings.create(
            model="text-embedding-3-small",
            input=[full_text]
        ).data[0].embedding

        q_vec = np.array(q_embed)

        df["similarity"] = [
            compute_similarity(q_vec, emb) for emb in embeddings
        ]

        top25 = df.sort_values("similarity", ascending=False).head(25)
        top10 = top25.head(10)

        avg_sim = top25["similarity"].mean()

        ######## Citation coverage ########
        ref_lines = [x.lower() for x in refs_input.split("\n") if x.strip()]
        match_count = 0
        for r in ref_lines:
            for t in top25["Title"]:
                if t.lower()[:25] in r:
                    match_count += 1
                    break
        citation_cov = f"{match_count}/{len(ref_lines)}"

        ######## Keyword score ########
        gap_kw = extract_keywords(gap_input)
        corpus_kw = extract_keywords(" ".join(df["Abstract"].astype(str).tolist()))
        overlap = set(gap_kw.index).intersection(corpus_kw.index)
        keyword_score = int(len(overlap) / max(len(gap_kw.index), 1) * 20)

        ######## GPT Evaluation ########
        gpt_eval = gpt_evaluate_json(
            title_input,
            gap_input,
            refs_input,
            "\n".join(top10["Title"])
        )

        ######## FINAL SCORE ########
        clarity_score = 15
        future_score = 15
        originality_score = 15

        citation_score = int((match_count / max(len(ref_lines), 1)) * 40)

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

        #############################
        # DASHBOARD
        #############################
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"""<div class="metric-card">
            <div class="metric-title">Avg Similarity</div>
            <div class="metric-value">{avg_sim:.3f}</div></div>""", unsafe_allow_html=True)
        col2.markdown(f"""<div class="metric-card">
            <div class="metric-title">Citation Coverage</div>
            <div class="metric-value">{citation_cov}</div></div>""", unsafe_allow_html=True)
        col3.markdown(f"""<div class="metric-card">
            <div class="metric-title">Keyword Score</div>
            <div class="metric-value">{keyword_score}/20</div></div>""", unsafe_allow_html=True)
        col4.markdown(f"""<div class="metric-card">
            <div class="metric-title">Verdict</div>
            <div class="metric-value">{badge}</div></div>""", unsafe_allow_html=True)

        #############################
        # TABS
        #############################
        tab1, tab2, tab3, tab4 = st.tabs(["Top Papers", "GPT Evaluation", "Weaknesses", "Rewritten Gap"])

        with tab1:
            st.dataframe(top25[["Title", "Year", "DOI", "similarity"]])

        with tab2:
            st.write("### Title")
            st.write(gpt_eval["title_comment"])
            st.write("### Clarity")
            st.write(gpt_eval["clarity_comment"])
            st.write("### Future Contribution")
            st.write(gpt_eval["future_comment"])
            st.write("### Originality")
            st.write(gpt_eval["originality_comment"])

        with tab3:
            st.subheader("Weaknesses")
            for w in gpt_eval["weaknesses"]:
                st.write(f"- {w}")

            st.subheader("Suggestions")
            for s in gpt_eval["suggestions"]:
                st.write(f"- {s}")

        with tab4:
            st.subheader("Rewritten Research Gap")
            st.write(gpt_eval["rewritten_gap"])

        #############################
        # PDF EXPORT
        #############################
        pdf_buffer = generate_pdf(
            title_input,
            avg_sim,
            citation_cov,
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
