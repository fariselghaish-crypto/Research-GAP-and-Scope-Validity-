import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI

# ------------------------------------------------------------
# Page Setup
# ------------------------------------------------------------
st.set_page_config(page_title="AI-BIM Research Gap Checker", layout="wide")
st.title("AI-BIM / Digital Construction Research Gap Checker")
st.write(
    "Step 1: Check title or scope relevance. Step 2: Evaluate the research gap using AI."
)

# ------------------------------------------------------------
# Sidebar Upload Section
# ------------------------------------------------------------
st.sidebar.header("Upload Required Files")

uploaded_df1 = st.sidebar.file_uploader(
    "Upload dt_construction_filtered_topics (1).csv", type=["csv"]
)

uploaded_df2 = st.sidebar.file_uploader(
    "Upload dt_topic_summary_reconstructed.csv", type=["csv"]
)

uploaded_emb = st.sidebar.file_uploader(
    "Upload embeddings.npy", type=["npy"]
)

api_key = st.sidebar.text_input(
    "Enter your OpenAI API key:", type="password"
)

# ------------------------------------------------------------
# Session state to enforce workflow
# ------------------------------------------------------------
if "title_checked" not in st.session_state:
    st.session_state["title_checked"] = False

# ------------------------------------------------------------
# Check Files
# ------------------------------------------------------------
if uploaded_df1 is None or uploaded_df1.name != "dt_construction_filtered_topics (1).csv":
    st.warning("Please upload dt_construction_filtered_topics (1).csv")
    st.stop()

if uploaded_df2 is None or uploaded_df2.name != "dt_topic_summary_reconstructed.csv":
    st.warning("Please upload dt_topic_summary_reconstructed.csv")
    st.stop()

if uploaded_emb is None or uploaded_emb.name != "embeddings.npy":
    st.warning("Please upload embeddings.npy")
    st.stop()

if not api_key.strip():
    st.warning("Please enter your OpenAI API key")
    st.stop()

# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------
df1 = pd.read_csv(uploaded_df1)
df2 = pd.read_csv(uploaded_df2)
corpus_df = pd.concat([df1, df2], ignore_index=True)

embeddings = np.load(uploaded_emb)

# ------------------------------------------------------------
# OpenAI client
# ------------------------------------------------------------
client = OpenAI(api_key=api_key)

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def compute_embedding(text: str):
    if not text.strip():
        return None
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(vec, mat):
    vec = vec / np.linalg.norm(vec)
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    return np.dot(mat, vec)

def get_top_k(similarities, k=25):
    idx = np.argsort(similarities)[::-1][:k]
    results = corpus_df.iloc[idx].copy()
    results["similarity"] = similarities[idx]
    return results

def gpt_evaluate_gap(gap_text, top10_df):
    study_context = "\n\n".join(
        [
            f"Title: {row.get('Title', '')}\nAbstract: {row.get('Abstract', '')}"
            for _, row in top10_df.iterrows()
        ]
    )

    prompt = f"""
You are an experienced academic supervisor in construction management and digital construction.
Evaluate whether the text below represents a VALID research gap.

Be strict.

### Research Gap:
{gap_text}

### Relevant Studies:
{study_context}

Provide the following:

1. Validity Decision (choose ONE): VALID or NOT VALID.
2. Reasons for the Decision.
3. Critical Weaknesses (min 5 bullet points).
4. Novelty Check (1 paragraph).
5. Improvement Points (5 to 10 bullets).
6. Rewrite an improved version of the research gap (150 to 200 words).

Important:
- Many student gaps must be classified as NOT VALID.
- Be objective and critical.
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )
    return completion.choices[0].message.content

# ------------------------------------------------------------
# STAGE 1: Title / Scope Relevance
# ------------------------------------------------------------
st.header("1. Title / Scope Relevance Check")

title_input = st.text_area(
    "Enter your dissertation TITLE or SCOPE (1 to 3 sentences):",
    height=110
)

if st.button("Check Title Relevance (Top 25 Papers)"):
    if not title_input.strip():
        st.error("Please enter a title or scope")
    else:
        title_emb = compute_embedding(title_input)
        sims = cosine_similarity(title_emb, embeddings)
        top25 = get_top_k(sims, k=25)

        st.subheader("Top 25 Most Relevant Studies")
        cols = [c for c in ["Title", "Year", "Source title", "DOI", "similarity"] if c in top25.columns]
        st.dataframe(top25[cols])

        csv = top25.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "top25_relevance.csv")

        st.session_state["title_checked"] = True
        st.success("Title relevance checked. Step 2 unlocked.")

# ------------------------------------------------------------
# STAGE 2: Research Gap Evaluation
# ------------------------------------------------------------
st.header("2. Research Gap Evaluation")

gap_input = st.text_area(
    "Enter your RESEARCH GAP paragraph (150 to 300 words):",
    height=160
)

if st.button("Evaluate Research Gap"):
    if not st.session_state["title_checked"]:
        st.error("Please complete Step 1 before Step 2.")
        st.stop()

    if not gap_input.strip():
        st.error("Please enter a research gap paragraph.")
        st.stop()

    gap_emb = compute_embedding(gap_input)
    sims = cosine_similarity(gap_emb, embeddings)
    top10 = get_top_k(sims, k=10)

    st.subheader("Top 10 Most Relevant Studies for This Gap")
    cols = [c for c in ["Title", "Year", "Source title", "DOI", "similarity"] if c in top10.columns]
    st.dataframe(top10[cols])

    st.subheader("Studies That MUST Be Considered")
    must_cols = [c for c in ["Title", "Year", "Source title", "DOI"] if c in top10.columns]
    st.table(top10[must_cols].head(5))

    st.subheader("AI Evaluation Output")
    with st.spinner("Evaluating research gap..."):
        result = gpt_evaluate_gap(gap_input, top10)
    st.write(result)
