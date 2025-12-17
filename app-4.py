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

You will evaluate the research gap using the ORIGINAL student gap and the Top-10 most relevant abstracts from the literature.

You are a senior academic reviewer for Automation in Construction, ECAM, and ITcon.

You will evaluate the research gap using the ORIGINAL student gap and the Top-10 most relevant abstracts from the literature.

RETURN JSON ONLY in this format:
{
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

SCORING RUBRIC:
Novelty: 0–3 low, 4–6 moderate, 7–8 strong, 9–10 outstanding
Significance: 0–3 low, 4–6 moderate, 7–8 high, 9–10 transformative
Clarity: 0–3 unclear, 4–6 acceptable, 7–8 clear, 9–10 excellent
Citation: 0–3 weak, 4–6 acceptable, 7–8 strong, 9–10 excellent


DO NOT rewrite the gap. Only evaluate it.

STUDENT GAP:
{gap}

REFERENCES PROVIDED:
{refs}

TOP-10 MOST RELEVANT PAPERS (WITH FULL ABSTRACTS):
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


#############################################################
# MAIN UI
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

        q_vec = embed_query(f"{title} {gap} {refs}")

        sims = vector_similarity(q_vec, embeddings)
        df_docs["similarity"] = sims

        top10 = df_docs.sort_values("similarity", ascending=False).head(10)
        top10_titles = top10["Title"].tolist()
        top10_abstracts = top10["Abstract"].fillna("").tolist()

        apa_list = []
        for t in top10_titles:
            row = df_scopus[df_scopus["Title"] == t]
            apa_list.append(build_apa(row.iloc[0]) if len(row) else f"{t} (metadata not found)")

        gpt_out = gpt_review(title, gap, refs, top10_titles, top10_abstracts, style_choice)

        #############################################################
        # HARD VALIDITY RULES (ORIGINAL GAP ONLY)
        #############################################################

        gap_word_count = len(gap.split())

        if gap_word_count >= 200:
            length_flag = "valid"
            length_penalty = 0
        elif 150 <= gap_word_count < 200:
            length_flag = "borderline"
            length_penalty = 5
        else:
            length_flag = "invalid"
            length_penalty = 15

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

        total_raw = (
            gpt_out["novelty_score"]
            + gpt_out["significance_score"]
            + gpt_out["clarity_score"]
            + gpt_out["citation_score"]
            - length_penalty
            - ref_penalty
        )
        total_score = max(0, min(40, total_raw))

        #############################################################
        # OVERRIDE RULE (IMPROVEMENTS + DOI/ TITLE + GPT >= 20)
        #############################################################

        original_text = gap.lower()
        improvements = gpt_out.get("improvements", [])

        improve_hits = sum(
            1 for imp in improvements
            if imp and imp.lower().split()[0] in original_text
        )
        improvement_ratio = improve_hits / len(improvements) if len(improvements) > 0 else 0

        ref_text = refs.lower()

        def normalize_doi(text):
            if not isinstance(text, str):
                return ""
            text = text.lower().strip()
            text = text.replace("https://doi.org/", "").replace("http://doi.org/", "")
            text = text.replace("doi:", "").replace(" ", "")
            return text

        top10_dois = []
        for title in top10_titles:
            row = df_scopus[df_scopus["Title"] == title]
            doi = normalize_doi(str(row.iloc[0].get("DOI", ""))) if len(row) else ""
            top10_dois.append((title, doi))

        def fuzzy_contains(a, b):
            a_clean = re.sub(r"[^a-z0-9 ]", "", a)
            b_clean = re.sub(r"[^a-z0-9 ]", "", b)
            return b_clean[:12] in a_clean

        lit_hits = 0
        for title, doi in top10_dois:
            if doi and doi in ref_text.replace(" ", ""):
                lit_hits += 1
                continue
            if fuzzy_contains(ref_text, title.lower()):
                lit_hits += 1

        improvements_ok = improvement_ratio >= 0.7
        literature_ok = lit_hits >= 5
        gpt_ok = total_score >= 20

        forced_valid = improvements_ok and literature_ok and gpt_ok

        #############################################################
        # VERDICT
        #############################################################
        if forced_valid:
            verdict = "🟢 VALID (Override Triggered)"
        elif length_flag == "invalid" or ref_flag == "invalid":
            verdict = "❌ NOT VALID"
        elif total_score >= 30:
            verdict = "🟢 VALID"
        elif total_score >= 20:
            verdict = "🟡 BORDERLINE"
        else:
            verdict = "❌ NOT VALID"

        #############################################################
        # METRICS DISPLAY
        #############################################################
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"<div class='metric-card'><div class='metric-title'>Novelty</div><div class='metric-value'>{gpt_out['novelty_score']}/10</div></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-card'><div class='metric-title'>Significance</div><div class='metric-value'>{gpt_out['significance_score']}/10</div></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-card'><div class='metric-title'>Clarity</div><div class='metric-value'>{gpt_out['clarity_score']}/10</div></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='metric-card'><div class='metric-title'>Citation Quality</div><div class='metric-value'>{gpt_out['citation_score']}/10</div></div>", unsafe_allow_html=True)

        st.subheader(f"Overall Verdict: {verdict}")

        #############################################################
        # TABS
        #############################################################
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📚 Top 10 Literature",
            "⭐ Good Points",
            "🚧 Improvements",
            "🔎 Reviewer Comments",
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
            for ref in apa_list:
                st.write("•", ref)
