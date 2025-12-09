import streamlit as st
import numpy as np
import pandas as pd
from openai import OpenAI
from pathlib import Path

DATA_PATH_1 = "dt_construction_filtered_topics.csv"
DATA_PATH_2 = "dt_topic_summary_reconstructed.csv"
EMB_PATH = "embeddings.npy"

df1 = pd.read_csv(DATA_PATH_1)
df2 = pd.read_csv(DATA_PATH_2)
corpus_df = pd.concat([df1, df2], ignore_index=True)

embeddings = np.load(EMB_PATH)

client = OpenAI()

def compute_embedding(text):
    if not text.strip():
        return None
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return np.array(resp.data[0].embedding)

def cosine_similarity(vec, mat):
    vec = vec / np.linalg.norm(vec)
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    return np.dot(mat, vec)

def get_top_k(similarities, k=25):
    indices = np.argsort(similarities)[::-1][:k]
    results = corpus_df.iloc[indices].copy()
    results["similarity"] = similarities[indices]
    return results

def ask_gpt_evaluation(student_gap_text, top10_df):
    context = "\n\n".join(
        [
            f"Title: {row['Title']}\nAbstract: {row['Abstract']}"
            for _, row in top10_df.iterrows()
        ]
    )

    prompt = f"""
You are an academic reviewer. 
Evaluate the following research gap and provide structured feedback.

### Research Gap:
{student_gap_text}

### Most Relevant Studies:
{context}

Provide the following:
1. Clarity of the research problem
2. Novelty
3. Contribution potential
4. Feasibility
5. Missing components
6. Final recommended improved gap (rewrite)
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )
    return resp.choices[0].message.content

st.set_page_config(page_title="AI-BIM Research Gap Checker", layout="wide")
st.title("AI-BIM Research Gap Checker")
st.markdown("This tool assesses topic relevance and research gap quality using AI.")

st.header("1. Title / Scope Relevance Check")
title_input = st.text_area("Enter your dissertation title or scope:", height=120)

if st.button("Check Relevance (Top 25 Papers)"):
    emb = compute_embedding(title_input)
    if emb is None:
        st.error("Please enter a valid text.")
    else:
        sims = cosine_similarity(emb, embeddings)
        top25 = get_top_k(sims, k=25)

        st.subheader("Top 25 Most Relevant Studies")
        st.dataframe(top25[["Title", "Year", "Journal", "DOI", "similarity"]])

        csv = top25.to_csv(index=False).encode("utf-8")
        st.download_button("Download Top 25 as CSV", csv, "top25_relevant_studies.csv", "text/csv")

st.header("2. Full Research Gap Evaluation")
gap_input = st.text_area("Enter your research gap paragraph:", height=180)

if st.button("Evaluate Research Gap"):
    gap_emb = compute_embedding(gap_input)
    if gap_emb is None:
        st.error("Please enter a valid text.")
    else:
        sims = cosine_similarity(gap_emb, embeddings)
        top10 = get_top_k(sims, k=10)

        st.subheader("Most Relevant Studies for This Gap (Top 10)")
        st.dataframe(top10[["Title", "Year", "Journal", "DOI", "similarity"]])

        st.subheader("Studies That MUST Be Considered in the Literature Review")
        st.table(top10[["Title", "Year", "Journal", "DOI"]].head(5))

        st.subheader("GPT Evaluation")
        with st.spinner("Evaluating research gap..."):
            result = ask_gpt_evaluation(gap_input, top10)
        st.write(result)
