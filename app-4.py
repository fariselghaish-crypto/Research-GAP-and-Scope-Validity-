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
    "Step 1: Check if your dissertation title/scope is aligned with existing research.\n"
    "Step 2: Evaluate the research gap with AI-based feedback."
)

# ------------------------------------------------------------
# Sidebar: file uploads + API key
# ------------------------------------------------------------
st.sidebar.header("Upload Required Files")

uploaded_df1 = st.sidebar.file_uploader(
    "Upload dt_construction_filtered_topics (1).csv",
    type=["csv"]
)

uploaded_df2 = st.sidebar.file_uploader(
    "Upload dt_topic_summary_reconstructed.csv",
    type=["csv"]
)

uploaded_emb = st.sidebar.file_uploader(
    "Upload embeddings.npy",
    type=["npy"]
)

api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")

# init session flag for workflow
if "title_checked" not in st.session_state:
    st.session_state["title_checked"] = False

# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------
if uploaded_df1 is None or uploaded_df1.name != "dt_construction_filtered_topics (1).csv":
    st.warning("Please upload **dt_construction_filtered_topics (1).csv** in the sidebar.")
    st.stop()

if uploaded_df2 is None or uploaded_df2.name != "dt_topic_summary_reconstructed.csv":
    st.warning("Please upload **dt_topic_summary_reconstructed.csv** in the sidebar.")
    st.stop()

if uploaded_emb is None or uploaded_emb.name != "embeddings.npy":
    st.warning("Please upload **embeddings.npy** in the sidebar.")
    st.stop()

if not api_key.strip():
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

# ------------------------------------------------------------
# Load data
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
# Utility functions
# ------------------------------------------------------------
def compute_embedding(text: str):
    """Embed text using OpenAI."""
    if not text.strip():
        return None
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return np.array(response.data[0].embedding)


def cosine_similarity(vec, mat):
    """Compute cosine similarity between a vector and matrix of vectors."""
    vec = vec / np.linalg.norm(vec)
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    return np.dot(mat, vec)


def get_top_k(similarities, k=25):
    """Return top-k most similar rows from corpus_df."""
    idx = np.argsort(similarities)[::-1][:k]
    results = corpus_df.iloc[idx].copy()
    results["similarity"] = similarities[idx]
    return results


def gpt_evaluate_gap(gap_text, top10_df):
    """Generate structured academic feedback using GPT."""
    study_context = "\n\n".join(
        [
            f"Title: {row.get('Title', '')}\nAbstract: {row.get('Abstract', '')}"
            for _, row in top10_df.iterrows()
        ]
    )

    prompt = f"""
You are an academic reviewer. Evaluate the following MSc research gap in AI/BIM/Digital Construction.

### Research Gap:
{gap_text}

### Most Relevant Studies:
{study_context}

Provide the following:
1. Is this a valid research gap? Explain why or why not.
2. How original/novel is this gap compared with the listed studies?
3. What is missing from the gap (e.g. context, variables, methods, datasets, case studies)?
4. List 5–10 specific improvement points as bullet points.
5. Rewrite an improved version of the research gap (150–200 words).
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )

    return completion.choices[0].message.content

# ------------------------------------------------------------
# STAGE 1 – Title / Scope relevance (Top 25 papers)
# ------------------------------------------------------------
st.header("1. Title / Scope Relevance Check")

title_input = st.text_area(
    "Enter your dissertation TITLE or SCOPE (1–3 sentences):",
    height=100,
    key="title_input",
)

if st.button("Check Title Relevance (Top 25 Papers)"):
    if not title_input.strip():
        st.error("Please enter a title or scope first.")
    else:
        title_emb = compute_embedding(title_input)
        if title_emb is None:
            st.error("Could not compute embedding. Please try again.")
        else:
            sims_title = cosine_similarity(title_emb, embeddings)
            top25 = get_top_k(sims_title, k=25)

            st.subheader("Top 25 Most Relevant Studies for This Title / Scope")

            # Try to use standard columns, but fall back gracefully if some do not exist
            cols_to_show = [c for c in ["Title", "Year", "Journal", "DOI", "similarity"] if c in top25.columns]
            if not cols_to_show:
                cols_to_show = list(top25.columns)

            st.dataframe(top25[cols_to_show])

            csv = top25.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Top 25 as CSV",
                csv,
                "top25_relevant_title_scope.csv",
                "text/csv"
            )

            # Mark that Stage 1 has been completed
            st.session_state["title_checked"] = True
            st.success("Title relevance checked. You can now move to Step 2 (Research Gap Evaluation).")

# ------------------------------------------------------------
# STAGE 2 – Research gap evaluation
# ------------------------------------------------------------
st.header("2. Research Gap Evaluation")

gap_input = st.text_area(
    "Enter your RESEARCH GAP paragraph (around 150–300 words):",
    height=160,
    key="gap_input",
)

if st.button("Evaluate Research Gap"):
    # Enforce workflow: must run Stage 1 first
    if not st.session_state.get("title_checked", False):
        st.error("Please first run the **Title / Scope Relevance Check** (Step 1) before evaluating the research gap.")
        st.stop()

    if not gap_input.strip():
        st.error("Please enter a research gap paragraph.")
        st.stop()

    gap_emb = compute_embedding(gap_input)
    if gap_emb is None:
        st.error("Could not compute embedding for the gap. Please try again.")
        st.stop()

    sims_gap = cosine_similarity(gap_emb, embeddings)
    top10 = get_top_k(sims_gap, k=10)

    st.subheader("Most Relevant Studies for This Research Gap (Top 10)")

    cols_to_show_gap = [c for c in ["Title", "Year", "Journal", "DOI", "similarity"] if c in top10.columns]
    if not cols_to_show_gap:
        cols_to_show_gap = list(top10.columns)

    st.dataframe(top10[cols_to_show_gap])

    st.subheader("Studies That MUST Be Considered in the Literature Review")
    must_cols = [c for c in ["Title", "Year", "Journal", "DOI"] if c in top10.columns]
    if not must_cols:
        must_cols = list(top10.columns)
    st.table(top10[must_cols].head(5))

    st.subheader("AI Evaluation of Your Research Gap")
    with st.spinner("Evaluating research gap using GPT..."):
        feedback = gpt_evaluate_gap(gap_input, top10)

    st.write(feedback)
