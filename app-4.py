import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="BIM Topic Research Gap Checker",
    layout="wide",
)

st.title("BIM Topic Research Gap Checker")
st.subheader("Evaluate dissertation titles and research gaps using uploaded embeddings.")

# ---------------------------------------------------------
# FILE UPLOADS
# ---------------------------------------------------------
st.sidebar.header("Upload Required Files")

parquet_file = st.sidebar.file_uploader(
    "Upload bert_documents_enriched.parquet",
    type=["parquet"]
)

embeddings_file = st.sidebar.file_uploader(
    "Upload bert_embeddings.npy",
    type=["npy"]
)

query_embedding_file = st.sidebar.file_uploader(
    "Upload query_embedding.npy (from BERT)",
    type=["npy"]
)

st.sidebar.write("---")

# Optional OpenAI for commentary
api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password")

# ---------------------------------------------------------
# MAIN INPUT FORM
# ---------------------------------------------------------
st.header("Research Gap Evaluation")

title_input = st.text_input(
    "Enter Dissertation Title",
    placeholder="Enter your dissertation working title..."
)

gap_input = st.text_area(
    "Paste Research Gap",
    height=180,
    placeholder="Paste your research gap text here..."
)

refs_input = st.text_area(
    "Paste APA References",
    height=140,
    placeholder="Paste your APA references here..."
)

run_button = st.button("Run Evaluation")

# ---------------------------------------------------------
# PROCESSING
# ---------------------------------------------------------
if run_button:

    # VALIDATION
    if parquet_file is None or embeddings_file is None or query_embedding_file is None:
        st.error("Please upload all required files.")
        st.stop()

    with st.spinner("Loading data and embeddings..."):

        # Load DF
        df = pd.read_parquet(parquet_file)
        df = df.fillna("")

        # Load embeddings
        embeddings = np.load(embeddings_file)
        query_vec = np.load(query_embedding_file)

        # Convert to float32 for faster computation if needed
        embeddings = embeddings.astype(np.float32)
        query_vec = query_vec.astype(np.float32)

    st.success("Files loaded successfully.")

    # ---------------------------------------------------------
    # VECTORISED SIMILARITY (FAST)
    # ---------------------------------------------------------
    with st.spinner("Computing similarities... (few seconds)"):

        emb_norms = np.linalg.norm(embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)

        sims = np.dot(embeddings, query_vec) / ((emb_norms * query_norm) + 1e-9)
        df["similarity"] = sims

    st.success("Similarity computed.")

    # ---------------------------------------------------------
    # SHOW TOP PAPERS
    # ---------------------------------------------------------
    st.header("Top 25 Most Relevant Papers")

    top_df = df.sort_values("similarity", ascending=False).head(25)

    st.dataframe(
        top_df[[
            "Title",
            "Authors",
            "Year",
            "Source title",
            "DOI",
            "Abstract",
            "similarity",
        ]],
        use_container_width=True
    )

    # ---------------------------------------------------------
    # OPTIONAL GPT COMMENTARY
    # ---------------------------------------------------------
    if api_key.strip() != "":
        import openai
        client = openai.OpenAI(api_key=api_key)

        st.header("GPT Commentary on Research Gap (Optional)")

        with st.spinner("Generating expert commentary..."):
            prompt = f"""
            Provide a concise expert commentary (no more than 2 paragraphs)
            evaluating this research gap and title.

            Title: {title_input}

            Research Gap:
            {gap_input}

            APA References Provided:
            {refs_input}
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350
            )

            st.write(response.choices[0].message["content"])

    else:
        st.info("Enter an OpenAI API key in the sidebar to generate GPT commentary.")
