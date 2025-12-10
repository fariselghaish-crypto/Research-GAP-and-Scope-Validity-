#############################################################
# AI-BIM / Digital Construction Research Gap Checker
# FINAL FULL VERSION – QUB Branding + Similarity + APA
# Uses:
# 1) bert_documents_enriched.parquet
# 2) bert_embeddings.npy
# 3) Scopus.csv
#############################################################

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import chardet
from io import BytesIO
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer

#############################################################
# PAGE CONFIG – QUB BRANDING
#############################################################
st.set_page_config(page_title="AI-BIM Research Gap Checker", layout="wide")

QUB_RED = "#CC0033"
QUB_DARK = "#002147"
QUB_LIGHT = "#F5F5F5"

#############################################################
# CSS – UI STYLING
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
.section-title {{
    font-size: 20px;
    font-weight: bold;
    color: {QUB_DARK};
    margin-top: 25px;
    margin-bottom: 10px;
}}
</style>
""", unsafe_allow_html=True)

#############################################################
# HEADER
#############################################################
st.markdown(f"""
<div class="header">
<h2>🎓 AI-BIM / Digital Construction Research Gap Checker</h2>
<p>Evaluate dissertation research gaps using similarity, Scopus metadata, and GPT-4.1-Premium rewriting.</p>
</div>
""", unsafe_allow_html=True)

#############################################################
# SIDEBAR – FILE UPLOADS
#############################################################
st.sidebar.header("Upload Required Files")

PARQUET = st.sidebar.file_uploader("Upload bert_documents_enriched.parquet", type=["parquet"])
EMB_PATH = st.sidebar.file_uploader("Upload bert_embeddings.npy", type=["npy"])
SCOPUS = st.sidebar.file_uploader("Upload Scopus.csv", type=["csv"])
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

style_choice = st.sidebar.selectbox(
    "Journal Style for Rewrite",
    ["Automation in Construction", "ECAM", "ITcon"]
)

if not (PARQUET and EMB_PATH and SCOPUS and api_key):
    st.warning("Please upload all 3 files and enter API key.")
    st.stop()

client = OpenAI(api_key=api_key)

#############################################################
# LOADERS
#############################################################
@st.cache_resource
def load_sbert():
    return SentenceTransformer("all-mpnet-base-v2")

@st.cache_resource
def load_data(parquet_file, emb_file):
    df = pd.read_parquet(parquet_file).fillna("")
    emb = np.load(emb_file)
    return df, emb

@st.cache_resource
def load_scopus(csv_file):
    raw = csv_file.read()
    enc = chardet.detect(raw)['encoding']
    df = pd.read_csv(BytesIO(raw), encoding=enc, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

sbert = load_sbert()
df_docs, embeddings = load_data(PARQUET, EMB_PATH)
df_scopus = load_scopus(SCOPUS)

#############################################################
# HELPERS – APA BUILDER
#############################################################
def build_apa(row):
    authors = row.get("Authors", "")
    year = str(row.get("Year", "n.d."))
    title = row.get("Title", "")
    journal = row.get("Source title", row.get("Source titl", ""))
    volume = row.get("Volume", "")
    issue = row.get("Issue", "")
    p1 = str(row.get("Page start", ""))
    p2 = str(row.get("Page end", ""))
    art = str(row.get("Art. No.", ""))
    doi = str(row.get("DOI", ""))

    pages = ""
    if p1 != "" and p2 != "" and p1 != "nan" and p2 != "nan":
        pages = f"{p1}-{p2}"
    elif art not in ["", "nan"]:
        pages = f"Article {art}"

    apa = f"{authors} ({year}). {title}. {journal}"
    if volume not in ["", "nan"]:
        apa += f", {volume}"
    if issue not in ["", "nan"]:
        apa += f"({issue})"
    if pages != "":
        apa += f", {pages}"
    if doi not in ["", "nan"]:
        apa += f". https://doi.org/{doi}"

    return apa

#############################################################
# SIMILARITY – FAST VECTORIZED
#############################################################
def vector_similarity(query_vec, emb_matrix):
    qn = np.linalg.norm(query_vec)
    dn = np.linalg.norm(emb_matrix, axis=1)
    sim = emb_matrix @ query_vec / (dn * qn + 1e-9)
    return sim

#############################################################
# GPT ENGINE
#############################################################
def gpt_review(title, gap, refs, top10_titles, style_choice):
    prompt = f"""
You are a senior academic reviewer. Analyse and rewrite the student's research gap.

Journal style required: {style_choice}.

TOP 10 MOST RELEVANT PAPERS:
{json.dumps(top10_titles, indent=2)}

TASKS:
1. Evaluate novelty, significance, clarity, citation strength, missing theory.
2. Produce scores (0-10) for:
   - Novelty
   - Significance
   - Clarity
   - Citation coverage
3. Provide critical comments.
4. Rewrite the gap in 300+ words in academic journal style, grounded in the top 10 papers.

RETURN JSON ONLY:
{{
"novelty_score": 0,
"significance_score": 0,
"clarity_score": 0,
"citation_score": 0,
"comments": [],
"rewritten_gap": ""
}}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-preview",
        temperature=0.0,
        max_tokens=1800,
        messages=[{"role":"user","content":prompt}]
    )

    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {
            "novelty_score": 0,
            "significance_score": 0,
            "clarity_score": 0,
            "citation_score": 0,
            "comments": ["GPT returned invalid JSON"],
            "rewritten_gap": gap
        }

#############################################################
# UI – MAIN INPUTS
#############################################################
st.title("📄 Research Gap Evaluation")

title = st.text_input("Enter Dissertation Title")
gap = st.text_area("Paste Research Gap", height=200)
refs = st.text_area("Paste References (APA)", height=150)

if st.button("Run Evaluation"):
    with st.spinner("Processing..."):

        ##################################################
        # SIMILARITY
        ##################################################
        full_text = f"{title} {gap} {refs}"
        q_vec = sbert.encode(full_text)
        sims = vector_similarity(q_vec, embeddings)

        df_docs["similarity"] = sims
        top10 = df_docs.sort_values("similarity", ascending=False).head(10)

        top10_titles = top10["Title"].tolist()

        ##################################################
        # BUILD APA REFERENCES FOR TOP 10
        ##################################################
        apa_list = []
        for t in top10_titles:
            row = df_scopus[df_scopus["Title"] == t]
            if len(row) > 0:
                apa_list.append(build_apa(row.iloc[0]))
            else:
                apa_list.append(f"{t} (metadata not found)")

        ##################################################
        # GPT REVIEW
        ##################################################
        gpt_out = gpt_review(title, gap, refs, top10_titles, style_choice)

        ##################################################
        # METRICS DISPLAY
        ##################################################
        col1, col2, col3, col4 = st.columns(4)

        col1.markdown(f"""<div class="metric-card">
        <div class="metric-title">🧠 Novelty</div>
        <div class="metric-value">{gpt_out['novelty_score']}/10</div></div>""", unsafe_allow_html=True)

        col2.markdown(f"""<div class="metric-card">
        <div class="metric-title">🔬 Significance</div>
        <div class="metric-value">{gpt_out['significance_score']}/10</div></div>""", unsafe_allow_html=True)

        col3.markdown(f"""<div class="metric-card">
        <div class="metric-title">📝 Clarity</div>
        <div class="metric-value">{gpt_out['clarity_score']}/10</div></div>""", unsafe_allow_html=True)

        col4.markdown(f"""<div class="metric-card">
        <div class="metric-title">🔗 Citation Coverage</div>
        <div class="metric-value">{gpt_out['citation_score']}/10</div></div>""", unsafe_allow_html=True)

        ##################################################
        # TABS
        ##################################################
        tab1, tab2, tab3, tab4 = st.tabs(["Top 10 Literature", "GPT Comments", "Rewritten Gap", "APA References"])

        with tab1:
            st.write(top10[["Title","Year","DOI","similarity"]])

        with tab2:
            for c in gpt_out["comments"]:
                st.write("•", c)

        with tab3:
            st.write(gpt_out["rewritten_gap"])

        with tab4:
            for a in apa_list:
                st.write(a)

