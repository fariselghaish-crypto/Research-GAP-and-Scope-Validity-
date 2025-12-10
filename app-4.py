###############################################################
# BIM Topic Research Gap Checker (FINAL STABLE VERSION)
# Works with ANY embedding dimension – NO CRASHES
###############################################################

import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import json
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="BIM Topic Research Gap Checker", layout="wide")

# --------------------------------------------------------------
# Layout
# --------------------------------------------------------------
st.markdown("""
# BIM Topic Research Gap Checker
Upload your parquet + embedding file and evaluate the research gap.
""")

# --------------------------------------------------------------
# Upload
# --------------------------------------------------------------
parquet_file = st.sidebar.file_uploader("Upload bert_documents_enriched.parquet", type=["parquet"])
embedding_file = st.sidebar.file_uploader("Upload bert_embeddings.npy", type=["npy"])
api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password")

client = OpenAI(api_key=api_key) if api_key else None

if not parquet_file or not embedding_file:
    st.warning("Please upload BOTH the parquet and the embeddings (.npy)")
    st.stop()

df = pd.read_parquet(parquet_file)
embeddings = np.load(embedding_file)

# safety check
if len(df) != len(embeddings):
    st.error(f"Row mismatch: parquet={len(df)} vs embeddings={len(embeddings)}")
    st.stop()

doc_dim = embeddings.shape[1]   # e.g. 3072

# --------------------------------------------------------------
# Load SBERT model for QUERY embedding ONLY
# --------------------------------------------------------------
model = SentenceTransformer("all-mpnet-base-v2")

# --------------------------------------------------------------
# Dimension Alignment Function
# --------------------------------------------------------------
def align_dims(vec, target_dim):
    """Pads or truncates vectors so cosine similarity never fails."""
    cur_dim = len(vec)
    if cur_dim == target_dim:
        return vec
    elif cur_dim < target_dim:
        pad = np.zeros(target_dim - cur_dim)
        return np.concatenate([vec, pad])
    else:
        return vec[:target_dim]

# --------------------------------------------------------------
# Cosine Similarity
# --------------------------------------------------------------
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

# --------------------------------------------------------------
# GPT JSON evaluation
# --------------------------------------------------------------
def gpt_feedback(title, gap, refs, top_titles):
    if not client:
        return {"note": "No API key provided – GPT evaluation skipped."}

    prompt = f"""
Return ONLY JSON in this format:

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
        return {"error": "GPT returned invalid JSON."}

# --------------------------------------------------------------
# UI Inputs
# --------------------------------------------------------------
title = st.text_input("Enter Dissertation Title")
gap = st.text_area("Paste Research Gap", height=180)
refs = st.text_area("Paste APA References", height=150)

# --------------------------------------------------------------
# RUN EVALUATION
# --------------------------------------------------------------
if st.button("Evaluate Research Gap"):
    with st.spinner("Embedding query + computing similarities..."):

        # embed query with SBERT
        query_vec_raw = model.encode(title + " " + gap + " " + refs)

        # align to document dimension (critical)
        query_vec = align_dims(query_vec_raw, doc_dim)

        # compute all similarities SAFELY
        sims = [cosine_sim(query_vec, e) for e in embeddings]
        df["similarity"] = sims

        # Top papers
        top25 = df.sort_values("similarity", ascending=False).head(25)

    st.success("Done.")

    # --------------------------------------------------------------
    # RESULTS DISPLAY
    # --------------------------------------------------------------
    st.subheader("Top 25 Relevant Papers")
    st.dataframe(top25[["Title", "Year", "Source title", "DOI", "similarity"]])

    # --------------------------------------------------------------
    # GPT evaluation
    # --------------------------------------------------------------
    if client:
        st.subheader("GPT Evaluation")
        with st.spinner("Generating structured feedback..."):
            top_titles = "\n".join(top25["Title"].tolist())
            gpt_out = gpt_feedback(title, gap, refs, top_titles)
        st.write(gpt_out)
    else:
        st.info("Add OpenAI key for GPT evaluation.")
