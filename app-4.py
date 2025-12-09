import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import re

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI-BIM Research Gap Checker",
    layout="wide"
)

# ------------------------------------------------------------
# CSS DESIGN
# ------------------------------------------------------------
st.markdown("""
<style>
    body { background-color: #F5F5F5; }
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
# HEADER
# ------------------------------------------------------------
st.markdown("""
<div class="banner">
    <h2>AI-BIM Research Gap & Title Quality Checker</h2>
    <p>AI-enabled evaluation of dissertation titles, research gaps, and APA citations.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# FILE UPLOADS
# ------------------------------------------------------------
st.sidebar.header("Upload Required Files")

uploaded_df1 = st.sidebar.file_uploader("Upload df1: dt_construction_filtered_topics (1).csv", type=["csv"])
uploaded_df2 = st.sidebar.file_uploader("Upload df2: dt_topic_summary_reconstructed.csv", type=["csv"])
uploaded_emb = st.sidebar.file_uploader("Upload embeddings.npy", type=["npy"])
api_key = st.sidebar.text_input("Enter OpenAI API key:", type="password")

# ------------------------------------------------------------
# VALIDATE
# ------------------------------------------------------------
if not uploaded_df1:
    st.error("Upload df1: dt_construction_filtered_topics (1).csv")
    st.stop()

if not uploaded_df2:
    st.error("Upload df2: dt_topic_summary_reconstructed.csv")
    st.stop()

if not uploaded_emb:
    st.error("Upload embeddings.npy")
    st.stop()

if not api_key.strip():
    st.error("Enter OpenAI API key.")
    st.stop()

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df1 = pd.read_csv(uploaded_df1)
df2 = pd.read_csv(uploaded_df2)
emb_all = np.load(uploaded_emb)
df1_emb = emb_all[: len(df1)]

client = OpenAI(api_key=api_key)

# ------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------
def compute_embedding(text):
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

def get_top10(sim):
    idx = np.argsort(sim)[::-1][:10]
    res = df1.iloc[idx].copy()
    res["similarity"] = sim[idx]
    return res

# ------------------------------------------------------------
# NEW: Citation extraction function
# ------------------------------------------------------------
def extract_citations(text):
    citations = []
    lines = text.split("\n")

    # Detect references block
    for line in lines:
        clean = line.strip()
        if re.search(r"\(\d{4}\)", clean):  # APA year
            citations.append(clean)
        if clean.lower().startswith("references"):
            continue

    return list(set(citations))

# ------------------------------------------------------------
# GPT EVALUATOR
# ------------------------------------------------------------
def gpt_evaluate(title, gap, refs_text, top10):

    student_refs = extract_citations(refs_text)
    refs_list_text = "\n".join(student_refs) if student_refs else "No APA references detected."

    top10_context = "\n".join([
        f"- {row['Title']} (DOI: {row['DOI']})"
        for _, row in top10.iterrows()
    ])

    prompt = f"""
You are an academic supervisor evaluating BOTH the dissertation TITLE and RESEARCH GAP.

============================================================
STUDENT TITLE:
{title}

STUDENT RESEARCH GAP:
{gap}

STUDENT APA REFERENCES (EXTRACTED):
{refs_list_text}

============================================================
TOP 10 RELEVANT PAPERS:
{top10_context}

============================================================
SCORING RUBRIC (100%):
1. Clarity of Research Gap – 20%
2. Citation Relevance (APA references matching top papers) – 40%
3. Future Direction & Contribution – 20%
4. Originality & Significance – 20%

Citation scoring rules:
- If 0 matching citations → 0/40.
- 1–2 matches → 10–20/40.
- 3–4 matches → 20–30/40.
- 5+ matches → 30–40/40.
- A match means APA reference aligns with a top-10 title.

============================================================
REQUIRED OUTPUT FORMAT:
0. TITLE EVALUATION (0–10)
- Clarity:
- Specificity:
- Scope:
- Academic alignment:
- Score:
- Improved titles:

1. CLARITY SCORE (20%)
Score:
Justification:

2. CITATION SCORE (40%)
Score:
Justification:
- Total APA references:
- Matching references:
- Matching Titles:

3. FUTURE DIRECTION (20%)
Score:
Justification:

4. ORIGINALITY (20%)
Score:
Justification:

5. TOTAL SCORE (0–100)
6. FINAL VERDICT (VALID / BORDERLINE / NOT VALID)

7. CRITICAL WEAKNESSES
(list)

8. IMPROVEMENT SUGGESTIONS
(list)

9. REWRITTEN RESEARCH GAP (150–220 words)
- Must reference missing areas compared to top-10 titles.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1800
    )

    return resp.choices[0].message.content


# ------------------------------------------------------------
# USER INPUTS
# ------------------------------------------------------------
st.header("Enter Your Inputs")

title_input = st.text_input("1. Dissertation Title")
gap_input = st.text_area("2. Research Gap Statement (150–300 words)", height=180)
refs_input = st.text_area("3. APA Reference List (copy/paste your references)", height=200)

if st.button("Run Analysis"):
    if not title_input.strip() or not gap_input.strip():
        st.error("Title and Research Gap are required.")
        st.stop()

    emb = compute_embedding(gap_input)
    if emb is None:
        st.error("Embedding failed. Try again.")
        st.stop()

    sim = cosine_similarity(emb, df1_emb)
    top10 = get_top10(sim)

    col1, col2, col3 = st.columns(3)
    col1.markdown('<div class="metric-card"><h3>Matched Papers</h3><h2>10</h2></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card"><h3>Avg Similarity</h3><h2>{top10["similarity"].mean():.3f}</h2></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-card"><h3>Strength</h3><h2>{"Strong" if top10["similarity"].mean()>0.5 else "Moderate"}</h2></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Top 10 Papers",
        "Must-Consider",
        "Title Evaluation",
        "Gap Evaluation"
    ])

    with tab1:
        st.subheader("Top 10 Most Relevant Papers")
        for _, row in top10.iterrows():
            st.markdown(f"### {row['Title']}")
            st.markdown(f"**Source:** {row['Source title']} ({row['Year']})")
            st.progress(float(row["similarity"]))
            with st.expander("Abstract"):
                st.write(row["Abstract"])
            st.markdown("---")

    with tab2:
        st.subheader("Top 5 Must-Consider Papers")
        cols = ["Title", "Year", "Source title", "DOI"]
        st.table(top10.head(5)[cols])

    result = gpt_evaluate(title_input, gap_input, refs_input, top10)

    with tab3:
        st.subheader("Title Evaluation")
        st.write(result)

    with tab4:
        st.subheader("Research Gap Evaluation")
        st.write(result)
