# AI-BIM Research Gap Checker

A Streamlit tool that evaluates dissertation topics and research gaps using embedding similarity and GPT-based academic analysis.

## Workflow

### Stage 1 — Title/Scope Relevance Check
- User enters dissertation title or scope
- Embedding similarity vs. full corpus
- Returns Top 25 relevant studies
- CSV download option

### Stage 2 — Full Research Gap Evaluation
- Embeds research gap
- Retrieves Top 10 relevant papers
- Generates structured GPT-4o-mini academic evaluation
- Includes key new feature:
  **“Studies That Should Be Considered in the Literature Review”**

## Required Files
- app.py
- embeddings.npy
- dt_construction_filtered_topics.csv
- dt_topic_summary_reconstructed.csv
- requirements.txt

## Installation
```
pip install -r requirements.txt
```

## Run
```
streamlit run app.py
```
