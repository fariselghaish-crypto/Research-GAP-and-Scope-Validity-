#############################################################
# AI-BIM / Digital Construction Research Gap Checker
# FULL VERSION WITH:
# - Balanced scoring
# - Weakness Map
# - Missing Concepts Detector (Top 10 ONLY)
# - Score Breakdown Box
# - APA grounding
# - Updated validity rules
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
    padding: 12px;
    border-radius: 10px;
    margin-top: 15px;
}}
</style>
""", unsafe_allow_html=True)

#############################################################
# HEADER
#############################################################
st.markdown(f"""
<div class="header">
<h2>🎓 AI-BIM / Digital Construction Research Gap Checker</h2>
<p>Evaluate research gaps using similarity, GPT analysis, weakness detection, and missing-concept identification grounded in your Top-10 relevant papers.</p>
</div>
""", unsafe_allow_html=True)

#############################################################
# SIDEBAR UPLOADS
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
    st.warning("Please upload all files and enter API key.")
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
# ALIGN LENGTHS (CRITICAL FIX)
#############################################################
num_docs = len(df_docs)
num_embs = embeddings.shape[0]

if num_docs != num_embs:
    min_len = min(num_docs, num_embs)
    st.warning(f"Docs ({num_docs}) ≠ Embeddings ({num_embs}). Using first {min_len}.")
    df_docs = df_docs.iloc[:min_len].reset_index(drop=True)
    embeddings = embeddings[:min_len, :]

#############################################################
# EMBEDDINGS FUNCTION
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
# GPT REVIEW (WEAKNESS MAP + BALANCED SCORING)
#############################################################
def gpt_review(title, gap, refs, top10_titles, style_choice):

    top10_texts = "; ".join(top10_titles)

    prompt = f"""
You are a senior academic reviewer.

TASK:
Provide a structured evaluation AND rewrite of the research gap.

SCORING RULES:
- Score 6–8 for strong but not perfect gaps.
- Score 9–10 only for exceptional novelty/significance/clarity.
- Score 5–6 for acceptable but weak gaps.
- Score <5 for serious issues.

SENTENCE-LEVEL WEAKNESS:
Analyse EVERY sentence in the student's gap and label as strong/weak.

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

Rules:
- Rewritten gap MUST be 250–300 words.
- Use journal style: {style_choice}

TEXT:
Title: {title}
Gap: {gap}
References: {refs}
"""

    resp = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.0,
        max_tokens=2600,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = resp.choices[0].message.content

    try:
        data = json.loads(raw)
    except:
        cleaned = raw[raw.find("{"):raw.rfind("}")+1]
        data = json.loads(cleaned)

    if "sentence_feedback" not in data:
        data["sentence_feedback"] = []

    return data


#############################################################
# GPT — MISSING CONCEPTS DETECTOR (TOP-10 ONLY)
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
You MUST only use the Top-10 papers provided.
No external knowledge. No hallucination.

TOP-10 PAPERS:
{combined}

STUDENT GAP:
{gap}

TASK:
1. Extract key concepts from Top-10 papers.
2. Extract concepts from student gap.
3. Identify concepts that appear in Top-10 but NOT in the student gap.
4. For each missing concept, return:
   - concept
   - importance
   - supported_by (APA list)
   - suggested_sentence

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

    resp = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.0,
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = resp.choices[0].message.content

    try:
        data = json.loads(raw)
    except:
        cleaned = raw[raw.find("{"):raw.rfind("}")+1]
        data = json.loads(cleaned)

    return data


#############################################################
# UI INPUTS
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

        apa_list = []
        for t in top10_titles:
            row = df_scopus[df_scopus["Title"] == t]
            if len(row) > 0:
                apa_list.append(build_apa(row.iloc[0]))
            else:
                apa_list.append(f"{t} (metadata not found)")

        # GPT review
        gpt_out = gpt_review(title, gap, refs, top10_titles, style_choice)

        # Missing concepts
        missing_out = gpt_missing_concepts(gap, top10, apa_list)

        #############################################################
        # VALIDITY RULES
        #############################################################
        rewritten_gap = gpt_out["rewritten_gap"]
        gap_word_count = len(rewritten_gap.split())

        # Word length penalty
        if gap_word_count >= 200:
            length_penalty = 0
            length_flag = "valid"
        elif 150 <= gap_word_count < 200:
            length_penalty = 5
            length_flag = "borderline"
        else:
            length_penalty = 15
            length_flag = "invalid"

        # Reference penalty
        ref_count = len([r for r in refs.split("\n") if r.strip()])
        if ref_count >= 7:
            ref_penalty = 0
            ref_flag = "valid"
        elif 5 <= ref_count <= 6:
            ref_penalty = 5
            ref_flag = "borderline"
        else:
            ref_penalty = 15
            ref_flag = "invalid"

        # Final score
        raw_score = (
            gpt_out["novelty_score"]
            + gpt_out["significance_score"]
            + gpt_out["clarity_score"]
            + gpt_out["citation_score"]
            - length_penalty
            - ref_penalty
        )

        total_score = max(0, min(40, raw_score))

        # Verdict
        if length_flag == "invalid" or ref_flag == "invalid":
            verdict = "❌ NOT VALID"
        elif total_score >= 30:
            verdict = "🟢 VALID"
        elif total_score >= 20:
            verdict = "🟡 BORDERLINE"
        else:
            verdict = "❌ NOT VALID"

        #############################################################
        # SHOW SCORE BREAKDOWN BOX
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
        # TABS
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
            st.write(top10[["Title", "Year", "DOI", "similarity"]])

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

        with tab7:
            st.write("### Sentence-Level Weakness Map")
            fb = gpt_out.get("sentence_feedback", [])

            if not fb:
                st.info("No sentence-level feedback returned.")
            else:
                # Highlighted version
                marked = []
                for s in fb:
                    sentence = s["sentence"]
                    label = s["label"].lower()
                    color = "#ffe6e6" if label == "weak" else "#e6ffe6"
                    marked.append(
                        f"<span style='background-color:{color}; padding:3px 5px; margin:2px; display:inline-block;'>{sentence}</span>"
                    )
                html_all = "<div style='line-height:1.8;'>" + " ".join(marked) + "</div>"
                st.markdown(html_all, unsafe_allow_html=True)

                st.write("### Detailed Feedback")
                for s in fb:
                    st.write(f"**Sentence:** {s['sentence']}")
                    st.write(f"- Label: {s['label']}")
                    st.write(f"- Comment: {s['comment']}")
                    st.write("---")

        with tab8:
            st.write("### Missing Concepts (Top-10 Grounded)")
            missing = missing_out.get("missing_concepts", [])

            if not missing:
                st.info("No missing concepts detected.")
            else:
                for m in missing:
                    st.write(f"#### 🧩 {m['concept']}")
                    st.write(f"- Importance: {m['importance']}")
                    st.write("- Supported by:")
                    for ref in m["supported_by"]:
                        st.write(f"  • {ref}")
                    st.write(f"- Suggested sentence: {m['suggested_sentence']}")
                    st.write("---")
