#############################################################
# AI-BIM / Digital Construction Research Gap Checker
# VERSION: FULL – STRICT SCORING – APA REFERENCES – UPDATED RULES
#############################################################

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from io import BytesIO
from openai import OpenAI
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

#############################################################
# PAGE CONFIG – QUB BRANDING
#############################################################
st.set_page_config(page_title="AI-BIM Research Gap Checker", layout="wide")

QUB_RED = "#CC0033"
QUB_DARK = "#002147"
QUB_LIGHT = "#F5F5F5"

#############################################################
# CSS
#############################################################
st.markdown(f"""
<style>
body {{
    background-color: {QUB_LIGHT};
    font-family: Arial, sans-serif;
}}
.header {{
    background-color: {QUB_DARK};
    padding: 25px 40px;
    border-radius: 10px;
    color: white;
}}
.metric-card {{
    background-color: white;
    border-left: 6px solid {QUB_RED};
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #ddd;
    text-align: center;
}}
.metric-title {{
    font-size: 16px;
    font-weight: 600;
    color: {QUB_DARK};
}}
.metric-value {{
    font-size: 28px;
    font-weight: 700;
    color: {QUB_RED};
}}
</style>
""", unsafe_allow_html=True)

#############################################################
# HEADER
#############################################################
st.markdown(f"""
<div class="header">
<h2>🎓 AI-BIM / Digital Construction Research Gap Checker</h2>
<p>Evaluate research gaps using similarity, Scopus metadata, and GPT analysis.</p>
</div>
""", unsafe_allow_html=True)

#############################################################
# SIDEBAR – Uploads
#############################################################
st.sidebar.header("Upload Required Files")

PARQUET = st.sidebar.file_uploader("Upload bert_documents_enriched.parquet", type=["parquet"])
EMB_PATH = st.sidebar.file_uploader("Upload bert_embeddings.npy", type=["npy"])
SCOPUS = st.sidebar.file_uploader("Upload Scopus.csv", type=["csv"])
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

style_choice = st.sidebar.selectbox(
    "Journal Style for Rewriting",
    ["Automation in Construction", "ECAM", "ITcon"]
)

if not (PARQUET and EMB_PATH and SCOPUS and api_key):
    st.warning("Please upload all 3 files and enter an API key.")
    st.stop()

client = OpenAI(api_key=api_key)

#############################################################
# LOADERS
#############################################################
@st.cache_resource
def load_docs(parquet_file, emb_file):
    df = pd.read_parquet(parquet_file).fillna("")
    emb = np.load(emb_file)
    return df, emb

@st.cache_resource
def load_scopus(csv_file):
    raw = csv_file.read()
    for enc in ["utf-8", "iso-8859-1", "utf-16"]:
        try:
            df = pd.read_csv(BytesIO(raw), encoding=enc, low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            return df
        except:
            pass
    df = pd.read_csv(BytesIO(raw), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

df_docs, embeddings = load_docs(PARQUET, EMB_PATH)
df_scopus = load_scopus(SCOPUS)

#############################################################
# EMBEDDINGS
#############################################################
def embed_query(text):
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(resp.data[0].embedding)

#############################################################
# APA Builder
#############################################################
def build_apa(row):
    authors = row.get("Authors", "")
    year = str(row.get("Year", "n.d."))
    title = row.get("Title", "")
    journal = row.get("Source title", "")
    volume = row.get("Volume", "")
    issue = row.get("Issue", "")
    p1 = str(row.get("Page start", ""))
    p2 = str(row.get("Page end", ""))
    art = str(row.get("Art. No.", ""))
    doi = str(row.get("DOI", ""))

    pages = ""
    if p1 and p2 and p1 != "nan" and p2 != "nan":
        pages = f"{p1}-{p2}"
    elif art and art != "nan":
        pages = f"Article {art}"

    apa = f"{authors} ({year}). {title}. {journal}"
    if volume and volume != "nan":
        apa += f", {volume}"
    if issue and issue != "nan":
        apa += f"({issue})"
    if pages:
        apa += f", {pages}"
    if doi and doi != "nan":
        apa += f". https://doi.org/{doi}"

    return apa

#############################################################
# SIMILARITY
#############################################################
def vector_similarity(query_vec, emb_matrix):
    qn = np.linalg.norm(query_vec)
    dn = np.linalg.norm(emb_matrix, axis=1)
    return emb_matrix @ query_vec / (dn * qn + 1e-9)

#############################################################
# GPT REVIEW – STRICT, STRUCTURED, JSON-PROOF
#############################################################
def gpt_review(title, gap, refs, top10_titles, style_choice):

    top10_text = "; ".join(top10_titles)

    prompt = f"""
You are a senior academic reviewer for journals such as Automation in Construction, ECAM, and ITcon.

TASK:
Provide a structured, detailed, critical evaluation and rewrite of the research gap.

Journal style required: {style_choice}

TOP 10 PAPER TITLES:
{top10_text}

RETURN JSON ONLY:
{{
"novelty_score": 0,
"significance_score": 0,
"clarity_score": 0,
"citation_score": 0,
"good_points": [],
"improvements": [],
"novelty_comment": "",
"significance_comment": "",
"citation_comment": "",
"rewritten_gap": ""
}}

RULES:
- Rewritten gap MUST be 250–300 words.
- Use academic tone, critical, structured.

TEXT:
Title: {title}
Gap: {gap}
References: {refs}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.0,
        max_tokens=2400,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content

    # JSON REPAIR
    try:
        return json.loads(raw)
    except:
        try:
            cleaned = raw[raw.find("{"): raw.rfind("}")+1]
            return json.loads(cleaned)
        except:
            return {
                "novelty_score": 0,
                "significance_score": 0,
                "clarity_score": 0,
                "citation_score": 0,
                "good_points": [],
                "improvements": [],
                "novelty_comment": "",
                "significance_comment": "",
                "citation_comment": "",
                "rewritten_gap": gap
            }

#############################################################
# UI INPUT
#############################################################
st.title("📄 Research Gap Evaluation")

title = st.text_input("Enter Dissertation Title")
gap = st.text_area("Paste Research Gap", height=200)
refs = st.text_area("Paste References (APA)", height=200)

#############################################################
# RUN EVALUATION
#############################################################
if st.button("Run Evaluation"):
    with st.spinner("Processing..."):

        full_text = f"{title} {gap} {refs}"
        q_vec = embed_query(full_text)

        sims = vector_similarity(q_vec, embeddings)
        df_docs["similarity"] = sims

        top10 = df_docs.sort_values("similarity", ascending=False).head(10)
        top10_titles = top10["Title"].tolist()

        # Build APA references for top 10
        apa_list = []
        for t in top10_titles:
            row = df_scopus[df_scopus["Title"] == t]
            if len(row) > 0:
                apa_list.append(build_apa(row.iloc[0]))
            else:
                apa_list.append(f"{t} (metadata not found)")

        # GPT REVIEW
        gpt_out = gpt_review(title, gap, refs, top10_titles, style_choice)

        #############################################################
        # HARD VALIDITY RULES (UPDATED)
        #############################################################

        # Extract rewritten gap
        rewritten_gap = gpt_out["rewritten_gap"]
        gap_word_count = len(rewritten_gap.split())

        # --- Word count rule ---
        if gap_word_count >= 200:
            length_flag = "valid"
            length_penalty = 0
        elif 150 <= gap_word_count < 200:
            length_flag = "borderline"
            length_penalty = 5
        else:
            length_flag = "invalid"
            length_penalty = 15

        # --- Reference count rule ---
        ref_list = [r for r in refs.split("\n") if r.strip()]
        ref_count = len(ref_list)

        if ref_count >= 7:
            ref_flag = "valid"
            ref_penalty = 0
        elif 5 <= ref_count <= 6:
            ref_flag = "borderline"
            ref_penalty = 5
        else:
            ref_flag = "invalid"
            ref_penalty = 15

        # --- FINAL SCORE CALCULATION ---
        total_raw = (
            gpt_out["novelty_score"]
            + gpt_out["significance_score"]
            + gpt_out["clarity_score"]
            + gpt_out["citation_score"]
            - length_penalty
            - ref_penalty
        )

        # Ensure score stays between 0 and 40
        total_score = max(0, min(40, total_raw))

        # --- FINAL VERDICT ---
        if length_flag == "invalid" or ref_flag == "invalid":
            verdict = "❌ NOT VALID"
        elif total_score >= 30:
            verdict = "🟢 VALID"
        elif total_score >= 20:
            verdict = "🟡 BORDERLINE"
        else:
            verdict = "❌ NOT VALID"

        #############################################################
        # METRICS UI
        #############################################################
        col1, col2, col3, col4 = st.columns(4)

        col1.markdown(
            f"<div class='metric-card'><div class='metric-title'>Novelty</div>"
            f"<div class='metric-value'>{gpt_out['novelty_score']}/10</div></div>",
            unsafe_allow_html=True
        )

        col2.markdown(
            f"<div class='metric-card'><div class='metric-title'>Significance</div>"
            f"<div class='metric-value'>{gpt_out['significance_score']}/10</div></div>",
            unsafe_allow_html=True
        )

        col3.markdown(
            f"<div class='metric-card'><div class='metric-title'>Clarity</div>"
            f"<div class='metric-value'>{gpt_out['clarity_score']}/10</div></div>",
            unsafe_allow_html=True
        )

        col4.markdown(
            f"<div class='metric-card'><div class='metric-title'>Citation Quality</div>"
            f"<div class='metric-value'>{gpt_out['citation_score']}/10</div></div>",
            unsafe_allow_html=True
        )

        st.subheader(f"Overall Verdict: {verdict}")

        #############################################################
        # TABS
        #############################################################
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📚 Top 10 Literature",
            "⭐ Good Points",
            "🚧 Improvements",
            "🔎 Novelty & Significance",
            "📝 Rewritten Gap",
            "📑 APA References"
        ])

        with tab1:
            st.write(top10[["Title","Year","DOI","similarity"]])

        with tab2:
            for p in gpt_out["good_points"]:
                st.write("•", p)

        with tab3:
            for p in gpt_out["improvements"]:
                st.write("•", p)

        with tab4:
            st.write("### Novelty Comment")
            st.write(gpt_out["novelty_comment"])
            st.write("### Significance Comment")
            st.write(gpt_out["significance_comment"])
            st.write("### Citation Comment")
            st.write(gpt_out["citation_comment"])

        with tab5:
            st.write(gpt_out["rewritten_gap"])

        with tab6:
            for ref in apa_list:
                st.write("•", ref)
