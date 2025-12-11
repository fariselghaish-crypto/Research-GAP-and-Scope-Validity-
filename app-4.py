#############################################################
# AI-BIM / Digital Construction Research Gap Checker
# FULL STABLE VERSION WITH:
# - Weakness Map
# - Missing Concepts Detector (Top-10 + APA)
# - Score Breakdown
# - 200-word validity rule
#############################################################

import streamlit as st
import pandas as pd
import numpy as np
import json
from io import BytesIO
from openai import OpenAI

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
.score-box {{
    background-color: white;
    border: 2px solid {QUB_DARK};
    padding: 15px;
    border-radius: 10px;
    margin: 20px 0;
}}
</style>
""", unsafe_allow_html=True)

#############################################################
# HEADER
#############################################################
st.markdown(f"""
<div class="header">
<h2>🎓 AI-BIM / Digital Construction Research Gap Checker</h2>
<p>Evaluate research gaps using similarity, Scopus metadata, GPT review, weakness detection, and missing concept analysis.</p>
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
    "Journal Style for Rewriting",
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
# SAFE ROW ALIGNMENT (NON-DESTRUCTIVE)
#############################################################
num_docs = len(df_docs)
num_embs = embeddings.shape[0]

if num_docs != num_embs:
    min_len = min(num_docs, num_embs)
    st.warning(f"Docs ({num_docs}) ≠ Embeddings ({num_embs}). Using first {min_len}.")
    df_docs = df_docs.iloc[:min_len].reset_index(drop=True)
    embeddings = embeddings[:min_len, :]

#############################################################
# EMBEDDING FOR QUERY
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
# GPT REVIEW (WITH WEAKNESS MAP)
#############################################################
def gpt_review(title, gap, refs, top10_titles, style_choice):

    top10_text = "; ".join(top10_titles)

    prompt = f"""
You are a senior academic reviewer.

TASKS:
1. Evaluate the research gap.
2. Provide comments.
3. Rewrite the gap (250–300 words).
4. Provide sentence-level feedback (strong/weak + reason).

RETURN JSON ONLY:
{{
"novelty_score":0,
"significance_score":0,
"clarity_score":0,
"citation_score":0,
"good_points":[],
"improvements":[],
"novelty_comment":"",
"significance_comment":"",
"citation_comment":"",
"rewritten_gap":"",
"sentence_feedback":[]
}}

STYLE: {style_choice}

TEXT:
Title: {title}
Gap: {gap}
References: {refs}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.0,
        max_tokens=2600,
        messages=[{"role":"user","content":prompt}]
    )

    raw = response.choices[0].message.content

    try:
        data = json.loads(raw)
    except:
        data = json.loads(raw[raw.find("{"): raw.rfind("}")+1])

    if "sentence_feedback" not in data:
        data["sentence_feedback"] = []

    return data

#############################################################
# GPT – MISSING CONCEPTS DETECTOR (TOP-10 ONLY)
#############################################################
def gpt_missing_concepts(gap, top10_df, apa_list):

    combined = ""
    for i, row in top10_df.iterrows():
        combined += f"""
TITLE: {row['Title']}
ABSTRACT: {row['Abstract']}
APA: {apa_list[i]}
"""

    prompt = f"""
You must use only the Top-10 papers.

TOP-10:
{combined}

GAP:
{gap}

TASK:
Identify all important concepts in the Top-10 that do NOT appear in the student's gap.

RETURN JSON ONLY:
{{
"missing_concepts":[
  {{
    "concept":"",
    "importance":"",
    "supported_by":[""],
    "suggested_sentence":""
  }}
]
}}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.0,
        max_tokens=2600,
        messages=[{"role":"user","content":prompt}]
    )

    raw = response.choices[0].message.content

    try:
        return json.loads(raw)
    except:
        return json.loads(raw[raw.find("{"): raw.rfind("}")+1])

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

        # APA LIST
        apa_list = []
        for t in top10_titles:
            row = df_scopus[df_scopus["Title"] == t]
            if len(row):
                apa_list.append(build_apa(row.iloc[0]))
            else:
                apa_list.append(f"{t} (metadata not found)")

        # GPT REVIEW
        gpt_out = gpt_review(title, gap, refs, top10_titles, style_choice)

        # MISSING CONCEPTS
        missing_out = gpt_missing_concepts(gap, top10, apa_list)

        #############################################################
        # VALIDITY RULES (200-word rule)
        #############################################################
        rewritten_gap = gpt_out["rewritten_gap"]
        word_count = len(rewritten_gap.split())

        if word_count >= 200:
            length_penalty = 0
        elif 150 <= word_count < 200:
            length_penalty = 5
        else:
            length_penalty = 15

        ref_count = len([r for r in refs.split("\n") if r.strip()])
        if ref_count >= 7:
            ref_penalty = 0
        elif 5 <= ref_count <= 6:
            ref_penalty = 5
        else:
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

        if length_penalty == 15 or ref_penalty == 15:
            verdict = "❌ NOT VALID"
        elif total_score >= 30:
            verdict = "🟢 VALID"
        elif total_score >= 20:
            verdict = "🟡 BORDERLINE"
        else:
            verdict = "❌ NOT VALID"

        #############################################################
        # SCORE BREAKDOWN
        #############################################################
        st.markdown(f"""
<div class='score-box'>
<h4>🔍 Score Breakdown</h4>
<p>Novelty: {gpt_out['novelty_score']} / 10</p>
<p>Significance: {gpt_out['significance_score']} / 10</p>
<p>Clarity: {gpt_out['clarity_score']} / 10</p>
<p>Citation Quality: {gpt_out['citation_score']} / 10</p>

<hr>
<p>Length Penalty: {length_penalty}</p>
<p>Reference Penalty: {ref_penalty}</p>

<hr>
<h4>Total Score: {total_score} / 40</h4>
<h3>Verdict: {verdict}</h3>
</div>
""", unsafe_allow_html=True)

        #############################################################
        # TABS (8)
        #############################################################
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📚 Top 10 Literature",
            "⭐ Good Points",
            "🚧 Improvements",
            "🔎 Novelty & Significance",
            "📝 Rewritten Gap",
            "📑 APA References",
            "🩻 Weakness Map",
            "🧭 Missing Concepts"
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
            st.write(rewritten_gap)

        with tab6:
            for ref in apa_list:
                st.write("•", ref)

        #############################################################
        # TAB 7 — WEAKNESS MAP
        #############################################################
        with tab7:

            fb = gpt_out.get("sentence_feedback", [])

            if not fb:
                st.info("No sentence-level feedback returned.")
            else:
                st.write("### Highlighted Gap")
                highlighted = []
                for s in fb:
                    color = "#ffe6e6" if s["label"].lower()=="weak" else "#e6ffe6"
                    highlighted.append(
                        f"<span style='background-color:{color}; padding:4px; border-radius:4px;'>{s['sentence']}</span>"
                    )
                st.markdown("<div style='line-height:1.8;'>" + " ".join(highlighted) + "</div>", unsafe_allow_html=True)

                st.write("### Detailed Feedback")
                for s in fb:
                    st.write(f"**Sentence:** {s['sentence']}")
                    st.write(f"- Label: {s['label']}")
                    st.write(f"- Comment: {s['comment']}")
                    st.write("---")

        #############################################################
        # TAB 8 — MISSING CONCEPTS
        #############################################################
        with tab8:

            missing = missing_out.get("missing_concepts", [])

            if not missing:
                st.info("No missing concepts detected.")
            else:
                st.write("### Missing Concepts (Based on Top 10 Papers)")
                for m in missing:
                    st.write(f"#### 🧩 {m['concept']}")
                    st.write(f"- Importance: {m['importance']}")
                    st.write("- Supported by:")
                    for ref in m["supported_by"]:
                        st.write(f"  • {ref}")
                    st.write(f"- Suggested sentence: {m['suggested_sentence']}")
                    st.write("---")
