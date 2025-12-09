#############################################################
# AI-BIM / Digital Construction Research Gap Checker
# Full Streamlit App – Updated Version (Dec 2025)
#############################################################

import streamlit as st
import pandas as pd
import numpy as np
import re
from openai import OpenAI

# ----------------------------------------------------------
# BASIC PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="AI-BIM Research Gap Checker",
    layout="wide"
)

st.markdown("""
<style>
.metric-card {
    background-color: #f7f9fc;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    border: 1px solid #dde3ec;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
st.sidebar.header("Upload Your Data")

DATA_PATH_1 = st.sidebar.file_uploader(
    "Upload dt_construction_filtered_topics.csv", type=["csv"]
)

DATA_PATH_2 = st.sidebar.file_uploader(
    "Upload dt_topic_summary_reconstructed.csv", type=["csv"]
)

EMB_PATH = st.sidebar.file_uploader(
    "Upload embeddings.npy", type=["npy"]
)

api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
client = OpenAI(api_key=api_key) if api_key else None

if DATA_PATH_1 and DATA_PATH_2 and EMB_PATH:
    df1 = pd.read_csv(DATA_PATH_1)
    df2 = pd.read_csv(DATA_PATH_2)
    embeddings = np.load(EMB_PATH)

    df1 = df1.fillna("")
    df2 = df2.fillna("")

    st.success("Data loaded successfully.")
else:
    st.warning("Please upload all required files to continue.")
    st.stop()

# ----------------------------------------------------------
# SIMILARITY FUNCTION
# ----------------------------------------------------------
def compute_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9)

# ----------------------------------------------------------
# GPT EVALUATION WITH RUBRIC
# ----------------------------------------------------------
def gpt_evaluate(title, gap, refs, top10):

    study_context = "\n\n".join([
        f"- Title: {row['Title']}\n  Abstract: {row['Abstract']}"
        for _, row in top10.iterrows()
    ])

    prompt = f"""
You are an academic supervisor evaluating a dissertation research gap using a strict weighted rubric.

### Student Title:
{title}

### Student Research Gap:
{gap}

### Student Reference List (APA):
{refs}

### Top 10 Relevant Papers:
{study_context}

============================================================
🔵 EVALUATION RUBRIC (100%)
============================================================

1. **CLARITY OF RESEARCH GAP (20%)**
Score 0–20.

2. **CITATION RELEVANCE (40%)**
- Detect APA-style citations from the student's reference list.
- Match citations to the titles in the Top-10 papers.
- Score based on how many citations match.
- Minimum 5–7 valid matches required for high score.

3. **FUTURE DIRECTION & CONTRIBUTION (20%)**
Score 0–20.

4. **ORIGINALITY & SIGNIFICANCE (20%)**
Score 0–20.

FINAL DECISION:
- VALID ≥ 70  
- BORDERLINE 50–69  
- NOT VALID < 50

============================================================
🔵 REQUIRED OUTPUT FORMAT
============================================================

### 🔹 TITLE QUALITY EVALUATION
Score: X/10  
Feedback:

### 🔹 CLARITY SCORE (20%)
Score: X/20  
Justification:

### 🔹 CITATION SCORE (40%)
Score: X/40  
Justification:

### 🔹 FUTURE CONTRIBUTION (20%)
Score: X/20  
Justification:

### 🔹 ORIGINALITY (20%)
Score: X/20  
Justification:

### 🔹 TOTAL SCORE (0–100)

### 🔹 FINAL VERDICT

### 🔹 CRITICAL WEAKNESSES (5–10 bullets)

### 🔹 IMPROVEMENT SUGGESTIONS (5–10 bullets)

### 🔹 REWRITTEN RESEARCH GAP (150–220 words)
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )

    return resp.choices[0].message.content


# ----------------------------------------------------------
# KEYWORD EXTRACTION
# ----------------------------------------------------------
def extract_keywords(text, n=10):
    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    if not tokens:
        return pd.Series([])
    freq = pd.Series(tokens).value_counts()
    return freq.head(n)


# ----------------------------------------------------------
# UI INPUTS
# ----------------------------------------------------------
st.title("AI-BIM / Digital Construction Research Gap Checker")

title_input = st.text_input("Enter Title")
gap_input = st.text_area("Paste Research Gap", height=200)
refs_input = st.text_area("Paste APA References", height=150)

if st.button("Evaluate Research Gap"):

    # ------------------------------------------------------
    # EMBED TITLE + GAP + REFERENCES
    # ------------------------------------------------------
    combined_text = title_input + " " + gap_input + " " + refs_input

    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[combined_text]
    )
    query_vec = np.array(resp.data[0].embedding)

    # ------------------------------------------------------
    # COMPUTE SIMILARITIES AGAINST df1 ONLY
    # ------------------------------------------------------
    corpus_vecs = embeddings[: len(df1)]
    sims = [compute_similarity(query_vec, v) for v in corpus_vecs]
    df1["similarity"] = sims

    top10 = df1.sort_values("similarity", ascending=False).head(10)
    top25 = df1.sort_values("similarity", ascending=False).head(25)

    avg_sim = top10["similarity"].mean()

    # ------------------------------------------------------
    # KPI CARDS
    # ------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"""
    <div class="metric-card">
    <h3>Matched Papers</h3><h2>10</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="metric-card">
    <h3>Avg Similarity</h3><h2>{avg_sim:.3f}</h2>
    </div>
    """, unsafe_allow_html=True)

    # Literature Match Strength
    if avg_sim >= 0.55:
        match_strength = "High Match"
    elif avg_sim >= 0.40:
        match_strength = "Moderate Match"
    else:
        match_strength = "Low Match"

    col3.markdown(f"""
    <div class="metric-card">
    <h3>Literature Match Strength</h3><h2>{match_strength}</h2>
    </div>
    """, unsafe_allow_html=True)

    # GPT Evaluation
    evaluation_output = gpt_evaluate(title_input, gap_input, refs_input, top10)

    # Extract Validity Verdict
    validity = "—"
    m = re.search(r"FINAL VERDICT\s*\n*([A-Z]+)", evaluation_output)
    if m:
        validity = m.group(1)

    validity_color = {
        "VALID": "🟩 VALID",
        "BORDERLINE": "🟨 BORDERLINE",
        "NOT VALID": "🟥 NOT VALID"
    }.get(validity, validity)

    col4.markdown(f"""
    <div class="metric-card">
    <h3>Validity Verdict</h3><h2>{validity_color}</h2>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------
    # TABS FOR RESULTS
    # ------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Top 10 Papers",
        "Must-Consider Papers",
        "Title Evaluation",
        "Gap Evaluation",
        "Keyword Analysis"
    ])

    with tab1:
        st.subheader("Top 10 Relevant Papers")
        st.dataframe(top10[["Title", "Year", "DOI", "similarity"]])

    with tab2:
        st.subheader("Top 5 MUST-Consider Papers")
        st.dataframe(top10.head(5)[["Title", "Year", "DOI", "similarity"]])

    with tab3:
        st.subheader("Title Evaluation")
        st.write(evaluation_output.split("### 🔹 TITLE QUALITY EVALUATION")[1].split("### 🔹 CLARITY")[0])

    with tab4:
        st.subheader("Research Gap Evaluation")
        st.write(evaluation_output)

    with tab5:
        st.subheader("Keyword Analysis")

        title_kw = extract_keywords(title_input)
        gap_kw = extract_keywords(gap_input)
        lit_kw = extract_keywords(" ".join(df1["Abstract"].tolist()))

        st.write("### Title Keywords")
        st.write(title_kw)

        st.write("### Research Gap Keywords")
        st.write(gap_kw)

        st.write("### Literature Keywords")
        st.write(lit_kw)

        overlap = set(gap_kw.index).intersection(set(lit_kw.index))
        overlap_score = len(overlap) / max(len(gap_kw.index), 1)

        if overlap_score >= 0.6:
            level = "High Keyword Alignment"
        elif overlap_score >= 0.35:
            level = "Moderate Keyword Alignment"
        else:
            level = "Low Keyword Alignment"

        st.info(f"Keyword Overlap Score: {overlap_score:.2f} → {level}")

