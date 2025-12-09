def gpt_evaluate(title, gap, top10):
    # Build study context with titles & abstracts only
    study_context = "\n\n".join([
        f"- Title: {row['Title']}\n  Abstract: {row['Abstract']}"
        for _, row in top10.iterrows()
    ])

    prompt = f"""
You are an academic supervisor evaluating a dissertation research gap using a strict weighted rubric.

### Student Title:
{title}

### Student Research Gap:
{gap}

### Top 10 Relevant Papers (use ONLY their titles for comparison):
{study_context}

============================================================
⭐ EVALUATION RUBRIC (100% Total)
============================================================

1. CLARITY OF THE RESEARCH GAP (20%)
- Is the gap clearly stated, specific, and academically coherent?
- Does it identify what is missing in the literature?
- Score 0–20.

2. CITATION RELEVANCE / ALIGNMENT WITH TOP PAPERS (40%)
- The student will add citations later; your role is to check conceptual alignment.
- Compare the gap to the titles and abstracts of the Top-10 papers.
- You MUST reference at least 5–7 paper titles in your justification.
- Check if the gap extends beyond what these papers already cover.
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
⭐ SCORING AND DECISION RULE
============================================================
Total Score = Clarity + Citation Relevance + Future Direction + Originality.

Final Verdict:
- VALID: score ≥ 70
- BORDERLINE: 50–69
- NOT VALID: < 50

============================================================
⭐ REQUIRED OUTPUT FORMAT
============================================================

1. Clarity Score (20%)
Score: X/20  
Justification:

2. Citation Relevance Score (40%)
Score: X/40  
Justification:  
(Must reference at least 5–7 paper titles exactly as provided.)

3. Future Direction & Contribution (20%)
Score: X/20  
Justification:

4. Originality & Significance (20%)
Score: X/20  
Justification:

5. Total Score (0–100)

6. Final Verdict (VALID / BORDERLINE / NOT VALID)

7. Critical Weaknesses  
(List 5–10 points)

8. Improvement Suggestions  
(List 5–10 points)

9. Rewrite the Research Gap (150–220 words)
- Must be clearer, more specific, more rigorous.
- Must explicitly reference missing areas relative to Top-10 titles.
- Must NOT invent any paper; use titles only.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )

    return resp.choices[0].message.content
