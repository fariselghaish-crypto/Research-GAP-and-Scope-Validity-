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
    "Step 1 checks title/scope relevance. Step 2 evaluates the research gap using similarity and GPT academic analysis."
)

# ------------------------------------------------------------
# Sidebar Uploads
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
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "title_checked" not in st.session_state:
    st.session_state["title_checked"] = False

# ------------------------------------------------------------
# Validate Files
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
df1 = pd.read_csv(uploaded_df1)   # Real research papers
df2 = pd.read_csv(uploaded_df2)   # Topic summaries
corpus_df = pd.concat([df1, df2], ignore_index=True)

embeddings = np.load(uploaded_emb)

# ------------------------------------------------------------
# GPT client
# ------------------------------------------------------------
client = OpenAI(api_key=api_key)

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def compute_embedding(text: str):
    if not text.strip():
        return None
    try:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return np.array(resp.data[0].embedding)
    except:
        return None


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
    # Build citations list for GPT
    study_context = "\n\n".join(
        [
            f"Study {i+1}:\nTitle: {row.get('Title', '')}\nAbstract: {row.get('Abstract', '')}"
            for i, (_, row) in enumerate(top10_df.iterrows())
        ]
    )

    # NEW STRICT + FAIR ACADEMIC PROMPT
    prompt = f"""
You are an experienced academic supervisor in digital construction and AI in the built environment.
Critically and fairly evaluate whether the text below represents a VALID MSc-level research gap.

### Research Gap to Evaluate:
{gap_text}

### Top Relevant Studies (use these for citations):
{study_context}

Your evaluation MUST strictly follow this structure:

------------------------------------------------------------
1. Validity Decision (choose ONLY one):
VALID  
NOT VALID  

Decision rule:
- Mark VALID ONLY if the gap is clear, specific, literature-grounded, and identifies a genuine missing academic element.
- Mark NOT VALID if it is vague, unsupported, already answered in literature, or lacks novelty.

------------------------------------------------------------
2. Clarity & Specificity Assessment
Evaluate whether the research gap is:
- clear and logically written,
- specific about the missing literature,
- academically justified,
- free of unsupported generalisations.

------------------------------------------------------------
3. Literature Integration (MANDATORY)
Critically evaluate how well the gap aligns with the findings of the top relevant studies.
You MUST:
- Cite AT LEAST 7 of the provided studies using natural-language in-text references
  (e.g., “According to Study 3…”).
- Show whether these studies already address the topic.
- Identify where gaps remain in the literature.

------------------------------------------------------------
4. Critical Weaknesses  
Provide 5–10 bullet points highlighting:
- missing detail,
- missing evidence,
- poor clarity,
- lack of novelty,
- unclear direction,
- missing methodological orientation.

------------------------------------------------------------
5. Novelty Assessment  
Explain the true novelty of the gap compared to the provided studies.

------------------------------------------------------------
6. Improvement Guidance  
Provide 5–10 precise, actionable improvements.

------------------------------------------------------------
7. Rewritten Research Gap (150–220 words)
Rewrite a complete, academically strong research gap that:
- cites AT LEAST 7 of the studies using “Study X” references,
- is clear, specific, and literature-grounded,
- defines a strong research direction,
- is suitable for an MSc dissertation.

Do NOT invent authors, dates, or DOIs.  
Use ONLY the studies provided above for citations.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1400,
    )
    return resp.choices[0].message.content


# ------------------------------------------------------------
# STEP 1 — Title / Scope Relevance
# ------------------------------------------------------------
st.header("1. Title / Scope Relevance Check")

title_input = st.text_area(
    "Enter your dissertation TITLE or SCOPE (1–3 sentences):",
    height=110
)

if st.button("Check Title Relevance (Top 25 Papers)"):
    if not title_input.strip():
        st.error("Please enter a title or scope.")
    else:
        title_emb = compute_embedding(title_input)
        if title_emb is None:
            st.error("Embedding failed. Try rewriting your text.")
            st.stop()

        sims = cosine_similarity(title_emb, embeddings)
        top25 = get_top_k(sims, k=25)

        st.subheader("Top 25 Most Relevant Studies")
        display_cols = [
            c for c in ["Title", "Year", "Source title", "DOI", "similarity"]
            if c in top25.columns
        ]
        st.dataframe(top25[display_cols])

        csv = top25.to_csv(index=False).encode("utf-8")
        st.download_button("Download Top 25 CSV", csv, "top25_relevance.csv")

        st.session_state["title_checked"] = True
        st.success("Title relevance checked. Step 2 unlocked.")


# ------------------------------------------------------------
# STEP 2 — Research Gap Evaluation
# ------------------------------------------------------------
st.header("2. Research Gap Evaluation")

gap_input = st.text_area(
    "Enter your RESEARCH GAP (150–300 words):",
    height=170
)

if st.button("Evaluate Research Gap"):
    if not st.session_state["title_checked"]:
        st.error("Please complete Step 1 first.")
        st.stop()

    if not gap_input.strip():
        st.error("Please enter a research gap paragraph.")
        st.stop()

    gap_emb = compute_embedding(gap_input)
    if gap_emb is None:
        st.error("Failed to generate embedding. Try rewriting.")
        st.stop()

    sims = cosine_similarity(gap_emb, embeddings)
    top10 = get_top_k(sims, k=10)

    # Filter OUT topic summaries (df2)
    top10_filtered = top10[top10["Title"].notna()].copy()
    top10_filtered = top10_filtered.reset_index(drop=True)

    st.subheader("Top 10 Most Relevant Research Papers")
    if top10_filtered.empty:
        st.warning("Top results matched topic summaries only. Try rewriting the gap.")
    else:
        display_cols = [
            c for c in ["Title", "Year", "Source title", "DOI", "similarity"]
            if c in top10_filtered.columns
        ]
        st.dataframe(top10_filtered[display_cols])

    st.subheader("Top 5 MUST-Consider Papers")
    must5 = top10_filtered.head(5)
    if must5.empty:
        st.warning("No real papers available.")
    else:
        must_cols = [c for c in ["Title", "Year", "Source title", "DOI"] if c in must5.columns]
        st.table(must5[must_cols])

    # GPT EVALUATION
    st.subheader("AI Evaluation Output")
    with st.spinner("Evaluating research gap..."):
        result = gpt_evaluate_gap(gap_input, top10_filtered)
    st.write(result)
