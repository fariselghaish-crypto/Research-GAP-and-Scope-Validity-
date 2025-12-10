###############################################
# BIM Topic Research Gap Checker (Clean Build)
# Using Parquet + NPY Embeddings Only
###############################################

import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import json

st.set_page_config(page_title="BIM Topic Research Gap Checker", layout="wide")

st.markdown("""
# BIM Topic Research Gap Checker  
Evaluate dissertation titles and research gaps using uploaded embeddings.
""")

# -----------------------------------------------------
# File Uploads
# -----------------------------------------------------
st.sidebar.header("Upload Required Files")

parquet_file = st.sidebar.file_uploader("Upload bert_documents_enriched.parquet", type=["parquet"])
embedding_file = st.sidebar.file_uploader("Upload bert_embeddings.npy", type=["npy"])
api_key = st.sidebar.text_input("OpenAI API Key (Optional)", type="password")

client = OpenAI(api_key=api_key) if api_key else None

if not parquet_file or not embedding_file:
    st.warning("Please upload the parquet + npy files.")
    st.stop()

# -----------------------------------------------------
# Load Data
# -----------------------------------------------------
df = pd.read_parquet(parquet_file)
embeddings = np.load(embedding_file)

if len(df) != len(embeddings):
    st.error(f"Row mismatch: parquet={len(df)} vs embeddings={len(embeddings)}")
    st.stop()

# -----------------------------------------------------
# Helper Functions
# -----------------------------------------------------
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

def gpt_feedback(title, gap, refs, top_titles):
    if not client:
        return {"note": "No API key provided – GPT evaluation skipped."}

    prompt = f"""
You are an academic supervisor. Return JSON only.

{{
"title_comment": "",
"clarity_comment": "",
"future_comment": "",
"originality_comment": "",
"weaknesses": [],
"suggestions": [],
"rewritten_gap": ""
}}

Title: {title}
Gap: {gap}
References: {refs}

Top matching titles:
{top_titles}
"""

    r = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        max_tokens=1800,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        return json.loads(r.choices[0].message.content)
    except:
        return {"error": "GPT response not JSON."}

# -----------------------------------------------------
# Main UI
# -----------------------------------------------------
st.subheader("Research Gap Evaluation")

title = st.text_input("Enter Dissertation Title")
gap = st.text_area("Paste Research Gap", height=160)
refs = st.text_area("Paste APA References", height=140)

if st.button("Run Evaluation"):
    with st.spinner("Computing similarities..."):
        # embed query using OpenAI if available
        if client:
            embed = client.embeddings.create(
                model="text-embedding-3-small",
                input=[title + " " + gap + " " + refs]
            )
            query_vec = np.array(embed.data[0].embedding)
        else:
            st.error("OpenAI key required for embedding.")
            st.stop()

        sims = [cosine_sim(query_vec, e) for e in embeddings]
        df["similarity"] = sims

        top25 = df.sort_values("similarity", ascending=False).head(25)

    st.success("Done!")

    # -----------------------------
    # Display results
    # -----------------------------
    st.subheader("Top 25 Relevant Papers")
    st.dataframe(top25[["Title", "Year", "Source title", "DOI", "similarity"]])

    # GPT
    st.subheader("GPT Evaluation (Optional)")

    if client:
        with st.spinner("Generating GPT evaluation..."):
            top_titles = "\n".join(top25["Title"].tolist())
            gpt_out = gpt_feedback(title, gap, refs, top_titles)
        st.write(gpt_out)
    else:
        st.info("Add API key to enable GPT feedback.")
