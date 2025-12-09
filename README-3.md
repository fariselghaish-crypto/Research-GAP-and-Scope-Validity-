# AI-BIM Research Gap Checker

A Streamlit tool for evaluating dissertation topic relevance and research gap quality.

## Deployment Notes (Streamlit Cloud)

This version is optimized for **direct file uploads to Streamlit Cloud**.

### You MUST upload these files manually via:
**Manage App → Files → Upload**

- `dt_construction_filtered_topics (1).csv`
- `dt_topic_summary_reconstructed.csv`
- `embeddings.npy`

## Workflow

### Stage 1 — Title/Scope Relevance Check
- User enters dissertation title or scope
- The system returns Top 25 most relevant papers

### Stage 2 — Research Gap Evaluation
- User inputs research gap
- Top 10 relevant papers retrieved
- GPT-4o-mini provides structured evaluation
- Includes “Studies That MUST Be Considered”

## Running Locally
```
pip install -r requirements.txt
streamlit run app.py
```
