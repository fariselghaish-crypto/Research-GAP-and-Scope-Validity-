import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI-BIM Research Gap Checker",
    layout="wide"
)

# ------------------------------------------------------------
# CUSTOM CSS FOR MODERN DASHBOARD UI
# ------------------------------------------------------------
st.markdown("""
<style>
    body {
        background-color: #F5F5F5;
    }
    .banner {
        background-color: #003366;
        padding: 20px 40px;
        border-radius: 8px;
        color: white;
        margin-bottom: 25px;
    }
    .metric-card {
        background: white;
        padding: 18px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# BANNER HEADER
# ------------------------------------------------------------
st.markdown("""
<div class="banner">
    <h2>AI-BIM Research Gap & Title Quality Checker</h2>
    <p>AI-enabled analysis of dissertation titles and research gaps using semantic similarity and APA-based GPT evaluation.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# SIDEBAR FILE UPLOADS
# ------------------------------------------------------------
st.sidebar.header("Upload Required Files")

uploaded_df1 = st.sidebar.file_uploader(
    "Upload df1: dt_construction_filtered_topics (1).csv", type=["csv"]
)
uploaded_df2 = st.sidebar.file_uploader(
    "Upload df2: dt_topic_summary_reconstructed.csv", type=["csv"]
)
uploaded_emb = st.sidebar.file_uploader(
    "Upload embeddings.npy", type=["npy"]
)
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")

# ------------------------------------------------------------
# VALIDATE UPLOADS
# ------------------------------------------------------------
if not uploaded_df1 or uploaded_df1.name != "dt_construction_filtered_topics (1).csv":
    st.error("Please upload df1: dt_construction_filtered_topics (1).csv")
    st.stop()

if not uploaded_df2 or uploaded_df2.name != "dt_topic_summary_reconstructed.csv":
    st.error("Please upload df2: dt_topic_summary_reconstructed.csv")
    st.stop()

if not uploaded_emb or uploaded_emb.name != "embeddings.npy":
    st.error("Please upload embeddings.npy")
    st.stop()

if not api_key.strip():
    st.error("Please enter your OpenAI API key.")
    st.stop()

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df1 = pd.read_csv(uploaded_df1)      # REAL PAPERS
df2 = pd.read_csv(uploaded_df2)      # TOPIC SUMMARIES (not used for similarity)
emb_all = np.load(uploaded_emb)

df1_emb = emb_all[: len(df1)]        # USE ONLY REAL PAPER EMBEDDINGS

# ------------------------------------------------------------
# OPENAI CLIENT
# ------------------------------------------------------------
client = OpenAI(api_key=api_key)

# ------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------
def compute_embedding(text):
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


def get_top10_real(sim):
    idx = np.argsort(sim)[::-1][:10]
    results = df1.iloc[idx].copy()
    results["similarity"] = sim[idx]
    return results


# ------------------------------------------------------------
# NEW APA-BASED GPT EVALUATOR
# ------------------------------------------------------------
def gpt_evaluate(title, gap, top10):
    # Create APA citation metadata block
    apa_context = ""
    for _, row in top10.iterrows():
        apa_context += (
            f"- Authors: {row['Authors']}\n"
            f"  Year: {row['Year']}\n"
            f"  Title: {row['Title']}\n"
            f"  Source: {row['Source title']}\n"
            f"  DOI: {row['DOI']}\n\n"
        )

    prompt = f"""
You are an academic supervisor in Digital Construction.

You must evaluate BOTH the dissertation TITLE and the RESEARCH GAP using REAL APA citations.

### Student Title:
{title}

### Student Research Gap:
{gap}

------------------------------------------------------------
TOP 10 RELEVANT STUDIES FOR APA CITATION
You MUST use only these metadata fields. Do NOT invent authors or years.

{apa_context}

------------------------------------------------------------
0. TITLE QUALITY EVALUATION
- Clarity
- Specificity
- Scope appropriateness
- Academic alignment
- Score 0–10
- Provide 2–3 improved versions

------------------------------------------------------------
1. VALIDITY DECISION (STRICT)
VALID or NOT VALID

------------------------------------------------------------
2. MANDATORY APA CITATION REQUIREMENT
A research gap is ONLY VALID if:
- It cites at least FIVE (5) of the papers above
- Each citation uses APA format (Author, Year)
- The citations MUST be based *exactly* on the provided metadata

If fewer than 5 valid citations appear in the student's gap:
→ The research gap MUST be classified as NOT VALID.
→ No exceptions.

------------------------------------------------------------
3. CLARITY & SPECIFICITY ASSESSMENT

------------------------------------------------------------
4. LITERATURE INTEGRATION (APA REQUIRED)
- Compare the gap with the literature
- Integrate at least 5 APA citations
- Highlight alignment and missing contributions

------------------------------------------------------------
5. CRITICAL WEAKNESSES (5–10 points)

------------------------------------------------------------
6. NOVELTY ASSESSMENT

------------------------------------------------------------
7. IMPROVEMENT GUIDANCE (5–10 items)

------------------------------------------------------------
8. REWRITE THE RESEARCH GAP (150–220 words)
- Must cite at least 5 of the above papers in APA format
- Must be academically strong
- Must reflect REAL literature only
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1800,
    )
    return resp.choices[0].message.content


# ------------------------------------------------------------
# USER INPUTS
# ------------------------------------------------------------
st.header("Enter Your Inputs")

title_input = st.text_input("Dissertation Title")
gap_input = st.text_area("Research Gap (150–300 words)", height=160)

run_button = st.button("Run Full Analysis")

if run_button:

    if not title_input.strip() or not gap_input.strip():
        st.error("Please enter BOTH title and research gap.")
        st.stop()

    gap_emb = compute_embedding(gap_input)
    if gap_emb is None:
        st.error("Embedding failed. Try rephrasing your text.")
        st.stop()

    sims = cosine_similarity(gap_emb, df1_emb)
    top10 = get_top10_real(sims)

    # KPI CARDS
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"""<div class="metric-card"><h3>Matched Papers</h3><h2>10</h2></div>""", unsafe_allow_html=True)
    col2.markdown(f"""<div class="metric-card"><h3>Avg Similarity</h3><h2>{top10['similarity'].mean():.3f}</h2></div>""", unsafe_allow_html=True)
    col3.markdown(f"""<div class="metric-card"><h3>Strength</h3><h2>{"Strong" if top10['similarity'].mean()>0.50 else "Moderate"}</h2></div>""", unsafe_allow_html=True)

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "Top 10 Relevant Papers",
        "Must-Consider Papers",
        "Title Evaluation",
        "Gap Evaluation"
    ])

    # TAB 1
    with tab1:
        st.subheader("Top 10 Most Relevant Papers")
        for i, row in top10.iterrows():
            st.markdown(f"### {i+1}. {row['Title']}")
            st.markdown(f"**Source:** {row['Source title']} ({row['Year']})")
            st.markdown(f"**Authors:** {row['Authors']}")
            st.markdown(f"**DOI:** {row['DOI']}")
            st.progress(float(row["similarity"]))
            with st.expander("Abstract"):
                st.write(row["Abstract"])
            st.markdown("---")

    # TAB 2
    with tab2:
        st.subheader("Top 5 MUST-Consider Papers")
        st.table(top10.head(5)[["Title", "Year", "Source title", "DOI"]])

    # RUN GPT
    evaluation_output = gpt_evaluate(title_input, gap_input, top10)

    # TAB 3
    with tab3:
        st.subheader("Title Evaluation Output")
        st.write(evaluation_output)

    # TAB 4
    with tab4:
        st.subheader("Research Gap Evaluation Output")
        st.write(evaluation_output)
