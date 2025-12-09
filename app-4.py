import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI

# ------------------------------------------------------------
# Page Setup
# ------------------------------------------------------------
st.set_page_config(page_title="AI-BIM Research Gap Checker", layout="wide")
st.title("AI-BIM / Digital Construction Research Gap Checker")
st.write("Upload your data files and paste a research gap to receive evaluation based on similarity and LLM feedback.")

# ------------------------------------------------------------
# Sidebar Upload Section
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

# ------------------------------------------------------------
# Check all files exist
# ------------------------------------------------------------
if not uploaded_df1 or uploaded_df1.name != "dt_construction_filtered_topics (1).csv":
    st.warning("Please upload **dt_construction_filtered_topics (1).csv**")
    st.stop()

if not uploaded_df2 or uploaded_df2.name != "dt_topic_summary_reconstructed.csv":
    st.warning("Please upload **dt_topic_summary_reconstructed.csv**")
    st.stop()

if not uploaded_emb or uploaded_emb.name != "embeddings.npy":
    st.warning("Please upload **embeddings.npy**")
    st.stop()

# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------
df1 = pd.read_csv(uploaded_df1)
df2 = pd.read_csv(uploaded_df2)
corpus_df = pd.concat([df1, df2], ignore_index=True)

embeddings = np.load(uploaded_emb)

# ------------------------------------------------------------
# OpenAI Client
# ------------------------------------------------------------
client = OpenAI(api_key=st.sidebar.text_input("Enter your OpenAI API key:", type="password"))

# ------------------------------------------------------------
# Utility Functions
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
    """Compute cosine similarity against a matrix."""
    vec = vec / np.linalg.norm(vec)
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    return np.dot(mat, vec)

def get_top_k(similarities, k=25):
    idx = np.argsort(similarities)[::-1][:k]
    results = corpus_df.iloc[idx].copy()
    results["similarity"] = similarities[idx]
    return results

def gpt_evaluate_gap(gap_text, top10_df):
    """Generate structured academic feedback using GPT."""
    study_context = "\n\n".join(
        [
            f"Title: {r['Title']}\nAbstract: {r['Abstract']}"
            for _, r in top10_df.iterrows()
        ]
    )

    prompt = f"""
You are an academic reviewer. Evaluate the following research gap.

### Research Gap:
{gap_text}

### Relevant Studies:
{study_context}

Provide the following in structured form:
1. Validity of the research gap
2. Why it is valid or not
3. How to improve or make the gap more original
4. List of short improvement points
5. Final feedback summary
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )

    return completion.choices[0].message.content

# ------------------------------------------------------------
# MAIN INTERFACE
# ------------------------------------------------------------

st.subheader("Paste the research gap (up to 300 words):")
user_text = st.text_area("")

# ------------------------
# BUTTON: Evaluate
# ------------------------
if st.button("Evaluate Research Gap"):

    if not user_text.strip():
        st.error("Please paste text before evaluating.")
        st.stop()

    # Create embedding
    gap_embedding = compute_embedding(user_text)

    if gap_embedding is None:
        st.error("Embedding failed. Please try again.")
        st.stop()

    # Similarity search
    similarities = cosine_similarity(gap_embedding, embeddings)

    # Top 10 papers
    top10 = get_top_k(similarities, k=10)

    st.header("AI Evaluation Output")

    st.subheader("Most Relevant Studies (Top 10)")
    st.dataframe(
        top10[["Title", "Year", "Journal", "DOI", "similarity"]]
    )

    st.subheader("Studies That MUST Be Considered")
    st.table(
        top10[["Title", "Year", "Journal", "DOI"]].head(5)
    )

    # GPT evaluation
    st.subheader("AI Evaluation Output")
    with st.spinner("Evaluating research gap..."):
        result = gpt_evaluate_gap(user_text, top10)

    st.write(result)
