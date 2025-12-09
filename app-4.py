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
    <p>AI-enabled analysis of dissertation titles and research gaps using semantic similarity and a strict academic rubric.</p>
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

# embeddings: use only df1 for similarity
df1_emb = emb_all[: len(df1)]

# ------------------------------------------------------------
# OPENAI CLIENT
# ------------------------------------------------------------
client = OpenAI(api_key=api_key)

# ------------------------------------------------------------
# HELPER FUNCTIONS
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
    except Exception:
        return None


def cosine_similarity(vec: np.ndarray, mat: np.ndarray):
    vec = vec / np.linalg.norm(vec)
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    return np.dot(mat, vec)


def get_top10_real(similarities: np.ndarray) -> pd.DataFrame:
    idx = np.argsort(similarities)[::-1][:10]
    results = df1.iloc[idx].copy()
    results["similarity"] = similarities[idx]
    return results


def gpt_evaluate(title: str, gap: str, top10: pd.DataFrame) -> str:
    # Build study context with titles & abstracts
    study_context = "\n\n".join([
        f"- Title: {row['Title']}\n  Abstract: {row['Abstract']}"
        for _, row in top10.iterrows()
    ])

    prompt = f"""
You are an experienced academic supervisor in Digital Construction.

You must evaluate BOTH the dissertation TITLE and the RESEARCH GAP using a strict weighted rubric, based on the Top-10 most relevant papers.

============================================================
STUDENT INPUT
============================================================

Title:
{title}

Research Gap:
{gap}

============================================================
TOP 10 RELEVANT PAPERS (for comparison)
============================================================
Use ONLY these titles and abstracts to judge alignment and originality. Do NOT invent any papers.

{study_context}

============================================================
EVALUATION RUBRIC (100% TOTAL)
============================================================

1. CLARITY OF THE RESEARCH GAP (20%)
- Is the gap clearly stated, specific, and academically coherent?
- Does it identify what is missing in the literature?
- Score 0–20.

2. CITATION RELEVANCE / ALIGNMENT WITH TOP PAPERS (40%)
- The student will later add formal references; your task is to check conceptual alignment.
- Compare the gap to the titles and abstracts of the Top-10 papers.
- You MUST reference at least 5–7 paper titles (exact titles) in your justification.
- Check if the gap truly extends beyond what these papers already cover.
- Score 0–40.

3. FUTURE DIRECTION & CONTRIBUTION (20%)
- Does the research gap clearly indicate what future work should address?
- Does it explain what new contribution this study will make beyond existing literature?
- Score 0–20.

4. ORIGINALITY & SIGNIFICANCE (20%)
- Is the gap genuinely original relative to the 10 papers?
- Is the topic significant for AI in construction / SMEs / digital transformation?
- Score 0–20.

============================================================
SCORING AND DECISION RULE
============================================================

Total Score = Clarity + Citation Relevance + Future Direction + Originality.

Final Verdict:
- VALID: score ≥ 70
- BORDERLINE: 50–69
- NOT VALID: < 50

============================================================
TITLE QUALITY CHECK (SEPARATE)
============================================================
Before you apply the rubric to the research gap, briefly evaluate the TITLE:

- Clarity (is it understandable and precise?)
- Specificity (does it indicate topic, context, and focus?)
- Scope (too broad, too narrow, or appropriate?)
- Academic alignment (does it sound like an MSc dissertation title?)
- Title score: 0–10
- Suggest 2–3 improved title variants if necessary.

============================================================
REQUIRED OUTPUT FORMAT
============================================================

Follow this structure exactly:

0. TITLE QUALITY EVALUATION
- Title clarity:
- Title specificity:
- Scope and academic alignment:
- Title score (0–10):
- 2–3 improved titles:

1. CLARITY SCORE (20%)
Score: X/20
Justification:

2. CITATION RELEVANCE SCORE (40%)
Score: X/40
Justification:
(Must reference at least 5–7 paper titles exactly as they appear above.)

3. FUTURE DIRECTION & CONTRIBUTION (20%)
Score: X/20
Justification:

4. ORIGINALITY & SIGNIFICANCE (20%)
Score: X/20
Justification:

5. TOTAL SCORE (0–100)

6. FINAL VERDICT
(VALID / BORDERLINE / NOT VALID)

7. CRITICAL WEAKNESSES
(List 5–10 bullet points.)

8. IMPROVEMENT SUGGESTIONS
(List 5–10 bullet points.)

9. REWRITTEN RESEARCH GAP (150–220 words)
- Must be clearer, more specific, more rigorous.
- Must explicitly reference what is missing in relation to the Top-10 paper titles.
- Must NOT invent any new papers; use only the ideas implied by the titles/abstracts above.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
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
        st.error("Please enter BOTH a title and a research gap.")
        st.stop()

    gap_emb = compute_embedding(gap_input)
    if gap_emb is None:
        st.error("Embedding failed. Try rephrasing your research gap text.")
        st.stop()

    sims = cosine_similarity(gap_emb, df1_emb)
    top10 = get_top10_real(sims)

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        '<div class="metric-card"><h3>Matched Papers</h3><h2>10</h2></div>',
        unsafe_allow_html=True
    )
    col2.markdown(
        f'<div class="metric-card"><h3>Avg Similarity</h3><h2>{top10["similarity"].mean():.3f}</h2></div>',
        unsafe_allow_html=True
    )
    col3.markdown(
        f'<div class="metric-card"><h3>Relevance Strength</h3><h2>{"Strong" if top10["similarity"].mean() > 0.50 else "Moderate"}</h2></div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Top 10 Relevant Papers",
        "Must-Consider Papers",
        "Title Evaluation",
        "Gap Evaluation"
    ])

    # TAB 1 — TOP 10
    with tab1:
        st.subheader("Top 10 Most Relevant Papers")
        for i, row in top10.iterrows():
            st.markdown(f"### {i+1}. {row['Title']}")
            st.markdown(f"**Source:** {row['Source title']} ({row['Year']})")
            if "Authors" in row:
                st.markdown(f"**Authors:** {row['Authors']}")
            st.markdown(f"**DOI:** {row['DOI']}")
            st.progress(float(row["similarity"]))
            with st.expander("Abstract"):
                st.write(row["Abstract"])
            st.markdown("---")

    # TAB 2 — MUST-CONSIDER
    with tab2:
        st.subheader("Top 5 MUST-Consider Papers")
        cols_to_show = [c for c in ["Title", "Year", "Source title", "DOI"] if c in top10.columns]
        st.table(top10.head(5)[cols_to_show])

    # GPT EVALUATION (one call)
    evaluation_output = gpt_evaluate(title_input, gap_input, top10)

    # TAB 3 — TITLE EVALUATION
    with tab3:
        st.subheader("Title & Gap Evaluation (Full Text)")
        st.info("The first section focuses on the title. Scroll for full rubric evaluation.")
        st.write(evaluation_output)

    # TAB 4 — GAP EVALUATION
    with tab4:
        st.subheader("Research Gap Evaluation (Same Output)")
        st.write(evaluation_output)
