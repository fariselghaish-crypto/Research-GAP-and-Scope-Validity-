import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="BIM Topic Research Gap Checker",
    layout="wide"
)

st.title("BIM Topic Research Gap Checker")
st.write("Evaluate dissertation titles and research gaps using uploaded precomputed embeddings.")

# ---------------------------------------------------------
# SIDEBAR
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
    "Upload query_embedding.npy",
    type=["npy"]
)

api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password")

# ---------------------------------------------------------
# MAIN INPUTS
# ---------------------------------------------------------
st.header("Research Gap Evaluation")

title_input = st.text_input("Enter Dissertation Title")

gap_input = st.text_area("Paste Research Gap", height=180)

refs_input = st.text_area("Paste APA References", height=140)

run_button = st.button("Run Evaluation")

# ---------------------------------------------------------
# RUN EVALUATION
# ---------------------------------------------------------
if run_button:

    # Validate
    if parquet_file is None or embeddings_file is None or query_embedding_file is None:
        st.error("Please upload all required files first.")
        st.stop()

    # Load Data
    with st.spinner("Loading data files..."):
        try:
            df = pd.read_parquet(parquet_file, engine="pyarrow")
            df = df.fillna("")
        except Exception as e:
            st.error(f"Failed to load Parquet file: {e}")
            st.stop()

        try:
            embeddings = np.load(embeddings_file, allow_pickle=False)
            query_vec = np.load(query_embedding_file, allow_pickle=False)
        except Exception as e:
            st.error(f"Failed to load embeddings: {e}")
            st.stop()

        # Type safety
        embeddings = embeddings.astype(np.float32)
        query_vec = query_vec.astype(np.float32)

        # Validate shape
        if embeddings.shape[1] != query_vec.shape[0]:
            st.error(f"Embedding dimension mismatch: embeddings={embeddings.shape}, query={query_vec.shape}")
            st.stop()

    st.success("Files loaded successfully.")

    # ---------------------------------------------------------
    # COMPUTE SIMILARITY (SAFE VECTORISED)
    # ---------------------------------------------------------
    with st.spinner("Computing similarities..."):

        try:
            norms = np.linalg.norm(embeddings, axis=1)
            qn = np.linalg.norm(query_vec)

            sims = np.dot(embeddings, query_vec) / (norms * qn + 1e-9)
            df["similarity"] = sims

        except Exception as e:
            st.error(f"Similarity calculation failed: {e}")
            st.stop()

    st.success("Similarity computed.")

    # ---------------------------------------------------------
    # SHOW TOP PAPERS
    # ---------------------------------------------------------
    st.header("Top 25 Most Relevant Papers")

    try:
        top_df = df.sort_values("similarity", ascending=False).head(25)
    except Exception as e:
        st.error(f"Sorting failed: {e}")
        st.stop()

    # Shorten abstract for safe display
    def short_abs(x):
        return x[:400] + "..." if len(x) > 400 else x

    top_df["short_abstract"] = top_df["Abstract"].apply(short_abs)

    st.dataframe(
        top_df[[
            "Title",
            "Authors",
            "Year",
            "Source title",
            "DOI",
            "short_abstract",
            "similarity"
        ]],
        use_container_width=True
    )

    # ---------------------------------------------------------
    # OPTIONAL GPT COMMENTARY
    # ---------------------------------------------------------
    if api_key:

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            st.header("GPT Commentary (Optional)")

            with st.spinner("Generating expert commentary..."):

                prompt = f"""
                Provide a concise academic commentary (max 2 paragraphs)
                evaluating the following dissertation proposal:

                Title:
                {title_input}

                Research Gap:
                {gap_input}

                Student's References:
                {refs_input}
                """

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=350
                )

                st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"GPT commentary failed: {e}")
