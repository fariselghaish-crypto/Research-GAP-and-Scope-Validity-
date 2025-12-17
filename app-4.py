#############################################################
# AI-BIM / Digital Construction Research Gap Checker
# FULL VERSION – WITH FULL ABSTRACTS – NO REWRITING
# DOI + FUZZY MATCH OVERRIDE, THRESHOLD 5, GPT>=20
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
<p>Evaluate research gaps using Top-10 literature, abstracts, Scopus metadata, and GPT reviewer analysis.</p>
</div>
""", unsafe_allow_html=True)

#############################################################
# SIDEBAR
#############################################################
st.sidebar.header("Upload Required Files")

PARQUET = st.sidebar.file_uploader("Upload bert_documents_enriched.parquet", type=["parquet"])
EMB_PATH = st.sidebar.file_uploader("Upload bert_embeddings.npy", type=["npy"])
SCOPUS = st.sidebar.file_uploader("Upload Scopus.csv", type=["csv"])
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

style_choice = st.sidebar.selectbox(
    "Journal Style for Review",
    ["Automation in Construction", "ECAM", "ITcon"]
)

if not (PARQUET and EMB_PATH and SCOPUS and api_key):
    st.warning("Please upload all 3 files and enter your API key.")
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
# ALIGN ROW COUNTS
#############################################################
num_docs = len(df_docs)
num_embs = embeddings.shape[0]

if num_docs != num_embs:
    min_len = min(num_docs, num_embs)
    st.warning(f"Document count ({num_docs}) ≠ Embeddings count ({num_embs}). Using first {min_len}.")
    df_docs = df_docs.iloc[:min_len].reset_index(drop=True)
    embeddings = embeddings[:min_len, :]

#############################################################
# EMBEDDING CALL
#############################################################
def embed_query(text):
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(resp.data[0].embedding)

#############################################################
# APA BUILDER
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
# GPT REVIEW WITH FULL ABSTRACTS
#############################################################
def gpt_review(title, gap, refs, top10_titles, top10_abstracts, style_choice):

    combined_abstracts = "\n\n".join(
        [
            f"PAPER {i+1}:\nTITLE: {t}\nABSTRACT:\n{a}"
            for i, (t, a) in enumerate(zip(top10_titles, top10_abstracts))
        ]
    )

    prompt = f"""
You are a senior academic reviewer for Automation in Construction, ECAM, and ITcon.

Evaluate the student's research gap using the ORIGINAL text and the Top-10 most relevant abstracts.

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
"citation_comment": ""
}}

SCORING:
Novelty: 0–3 low, 4–6 moderate, 7–8 strong, 9–10 outstanding
Significance: 0–3 low, 4–6 moderate, 7–8 high, 9–10 transformative
Clarity: 0–3 unclear, 4–6 acceptable, 7–8 clear, 9–10 excellent
Citation: 0–3 weak, 4–6 acceptable, 7–8 strong, 9–10 excellent

DO NOT rewrite the gap.

STUDENT GAP:
{gap}

REFERENCES:
{refs}

TOP-10 ABSTRACTS:
{combined_abstracts}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.0,
        max_tokens=5000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content

    try:
        return json.loads(raw)
    except:
        cleaned = raw[raw.find("{"): raw.rfind("}")+1]
        try:
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
                "citation_comment": ""
            }
