#############################################################
# AI-BIM / Digital Construction Research Gap Checker
# QUB Edition – With PDF Export
# FULL APP – DEC 2025 RELEASE (RUBRIC SCORES + APA REFERENCES)
#############################################################

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from io import BytesIO
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ==========================================================
# PAGE CONFIG + BRANDING
# ==========================================================
st.set_page_config(
    page_title="AI-BIM Research Gap Checker",
    layout="wide",
    page_icon="📘"
)

QUB_RED = "#CC0033"
QUB_DARK = "#002147"
QUB_LIGHT = "#F5F5F5"

# ==========================================================
# GLOBAL CSS STYLE
# ==========================================================
st.markdown(f"""
<style>
body {{
    background-color: {QUB_LIGHT};
}}
.metric-card {{
    background-color: white;
    border-left: 6px solid {QUB_RED};
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #ddd;
    text-align: center;
}}
.metric-title {{
    font-size: 18px;
    color: {QUB_DARK};
    margin-bottom: 8px;
    font-weight: 600;
}}
.metric-value {{
    font-size: 32px;
    font-weight: 700;
    color: {QUB_RED};
}}
.header {{
    background-color: {QUB_DARK};
    padding: 25px 40px;
    border-radius: 10px;
    color: white;
}}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================
st.markdown(f"""
<div class="header">
<h2>AI-BIM / Digital Construction Research Gap Checker</h2>
<p>Evaluate dissertation titles and research gaps using AI, similarity analysis, and evidence-based scoring.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("Upload Dataset")

DATA_PATH_1 = st.sidebar.file_uploader("Upload dt_construction_filtered_topics.csv", type=["csv"])
DATA_PATH_2 = st.sidebar.file_uploader("Upload dt_topic_summary_reconstructed.csv", type=["csv"])
EMB_PATH = st.sidebar.file_uploader("Upload embeddings.npy", type=["npy"])
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

client = OpenAI(api_key=api_key) if api_key else None

if not (DATA_PATH_1 and DATA_PATH_2 and EMB_PATH):
    st.warning("Please upload all required dataset files.")
    st.stop()

df1 = pd.read_csv(DATA_PATH_1).fillna("")
df2 = pd.read_csv(DATA_PATH_2).fillna("")
embeddings = np.load(EMB_PATH)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def compute_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9)

def extract_keywords(text, n=10):
    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    if not tokens:
        return pd.Series([])
    freq = pd.Series(tokens).value_counts()
    return freq.head(n)

def generate_pdf(title, avg_sim, citation_cov, keyword_score, verdict, gpt_eval):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    def write_line(text, font="Helvetica", size=11, step=15):
        nonlocal y
        c.setFont(font, size)
        for line in text.split("\n"):
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont(font, size)
            c.drawString(40, y, line[:100])
            y -= step

    write_line("AI-BIM Research Gap Evaluation Report", "Helvetica-Bold", 16, 25)
    write_line(f"Title: {title}", "Helvetica-Bold", 12, 20)
    write_line("Summary Metrics:", "Helvetica-Bold", 12, 20)
    write_line(f"• Avg Similarity: {avg_sim:.3f}")
    write_line(f"• Citation Coverage: {citation_cov}")
    write_line(f"• Keyword Score: {keyword_score}/20")
    write_line(f"• Verdict: {verdict}", "Helvetica-Bold", 12, 20)

    write_line("\nGPT Evaluation:", "Helvetica-Bold", 13, 25)
    write_line("Title Evaluation:", "Helvetica-Bold", 12)
    write_line(gpt_eval.get("title_comment", ""))
    write_line("\nClarity:", "Helvetica-Bold", 12)
    write_line(gpt_eval.get("clarity_comment", ""))
    write_line("\nFuture Contribution:", "Helvetica-Bold", 12)
    write_line(gpt_eval.get("future_comment", ""))
    write_line("\nOriginality:", "Helvetica-Bold", 12)
    write_line(gpt_eval.get("originality_comment", ""))
    write_line("\nWeaknesses:", "Helvetica-Bold", 12)
    write_line("\n".join(gpt_eval.get("weaknesses", [])))
    write_line("\nSuggestions:", "Helvetica-Bold", 12)
    write_line("\n".join(gpt_eval.get("suggestions", [])))
    write_line("\nRewritten Gap:", "Helvetica-Bold", 12)
    write_line(gpt_eval.get("rewritten_gap", ""))

    # Optional: references in PDF (can be expanded later)
    if "references_list" in gpt_eval and gpt_eval["references_list"]:
        write_line("\nReferences (APA style):", "Helvetica-Bold", 12, 20)
        for ref in gpt_eval["references_list"]:
            write_line(ref)

    c.save()
    buffer.seek(0)
    return buffer

# ==========================================================
# UPDATED GPT EVALUATION FUNCTION
# WITH SCORES + CAREFUL CITATIONS + APA REFERENCES
# ==========================================================
def gpt_evaluate_json(title, gap, refs, top10_text):
    prompt = f"""
You are an academic supervisor in AI-driven Digital Construction, evaluating a dissertation research gap.

TASKS:
1) Critically evaluate the title, clarity of the gap, future research directions, and originality.
2) Rewrite the research gap using ONLY the Top-Matched Literature provided.
3) Provide full APA-style references for ALL 10 top-matched sources.

REWRITTEN GAP REQUIREMENTS:
- Length: 250–320 words.
- Integrate findings, limitations, and methods from the abstracts.
- Use at least SIX different numbered citations.
- Use NUMBERED in-text citations in this exact format: (1), (2), (3).
- Numbers MUST match the order of the Top-Matched Literature.
- Do NOT mention authors or years in the rewritten gap.
- Writing MUST be academically strong, critical, and evidence-driven.
- Do NOT generalise or guess content. Only cite what clearly appears in a provided abstract.

CITATION QUALITY RULES:
- Every citation MUST support the sentence it is attached to.
- Do NOT cite a paper unless the abstract content explicitly supports that claim.
- Distribute citations across the rewritten gap logically.
- Avoid over-citing the same source.
- Avoid invented, implied, or fabricated findings.

SCORING RULES:
You MUST provide numeric scores according to this rubric:
- title_score: 0–10 (clarity, scope, specificity, academic suitability)
- clarity_score: 0–20 (coherence, structure, precision of the research gap)
- future_score: 0–20 (specificity and realism of future research directions)
- originality_score: 0–20 (strength and distinctiveness of the research gap)

Your JSON RESPONSE MUST FOLLOW THIS EXACT STRUCTURE:

{{
"title_comment": "",
"title_score": 0,

"clarity_comment": "",
"clarity_score": 0,

"future_comment": "",
"future_score": 0,

"originality_comment": "",
"originality_score": 0,

"weaknesses": [],
"suggestions": [],

"rewritten_gap": "",
"references_list": []
}}

Student Input:
Title: {title}
Research Gap: {gap}
References: {refs}

Top-Matched Literature (use these as numbered citations):
{top10_text}

IMPORTANT:
- "references_list" MUST contain ALL 10 sources in APA-7 style, one per item, in the same order as the Top-Matched Literature.
- The rewritten_gap MUST strictly follow citation rules.
"""
    resp = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3200
    )

    content = resp.choices[0].message.content

    try:
        parsed = json.loads(content)
    except Exception:
        parsed = {
            "title_comment": "JSON formatting error.",
            "title_score": 0,
            "clarity_comment": "Error.",
            "clarity_score": 0,
            "future_comment": "Error.",
            "future_score": 0,
            "originality_comment": "Error.",
            "originality_score": 0,
            "weaknesses": [],
            "suggestions": [],
            "rewritten_gap": gap,
            "references_list": []
        }
    return parsed

# ==========================================================
# MAIN UI
# ==========================================================
st.title("Research Gap Evaluation")

title_input = st.text_input("Enter Dissertation Title")
gap_input = st.text_area("Paste Research Gap", height=180)
refs_input = st.text_area("Paste APA References", height=150)

if st.button("Evaluate Research Gap"):
    with st.spinner("Processing evaluation..."):

        full_text = title_input + " " + gap_input + " " + refs_input
        embed = client.embeddings.create(
            model="text-embedding-3-small",
            input=[full_text]
        )
        query_vec = np.array(embed.data[0].embedding)

        df1["similarity"] = [compute_similarity(query_vec, v) for v in embeddings[:len(df1)]]
        top10 = df1.sort_values("similarity", ascending=False).head(10)
        avg_sim = top10["similarity"].mean()

        # ===== Citation Coverage (0–40) =====
        ref_lines = [r.lower() for r in refs_input.split("\n") if r.strip()]
        match_count = 0
        for r in ref_lines:
            for t in top10["Title"]:
                if t.lower()[:25] in r:
                    match_count += 1
                    break
        cov_ratio = match_count / max(len(ref_lines), 1)
        citation_score = int(cov_ratio * 40)

        # ===== Keyword Score (0–20, diagnostic only) =====
        gap_kw = extract_keywords(gap_input)
        lit_kw = extract_keywords(" ".join(df1["Abstract"].tolist()))
        overlap = set(gap_kw.index).intersection(lit_kw.index)
        keyword_score = int(len(overlap) / max(len(gap_kw.index), 1) * 20)

        # ===== GPT Evaluation (UPDATED) =====
        top10_text = "\n\n".join([
            f"({idx+1}) TITLE: {row['Title']}\nABSTRACT: {row['Abstract']}"
            for idx, (_, row) in enumerate(top10.iterrows())
        ])

        gpt_eval = gpt_evaluate_json(title_input, gap_input, refs_input, top10_text)

        # Safely extract scores from GPT output
        title_score = int(gpt_eval.get("title_score", 0))
        clarity_score = int(gpt_eval.get("clarity_score", 0))
        future_score = int(gpt_eval.get("future_score", 0))
        originality_score = int(gpt_eval.get("originality_score", 0))

        # ===== Final Score (Rubric: 20+40+20+20 = 100) =====
        total = clarity_score + citation_score + future_score + originality_score

        if total >= 70:
            verdict = "VALID"
            badge = "🟢 VALID"
        elif total >= 50:
            verdict = "BORDERLINE"
            badge = "🟡 BORDERLINE"
        else:
            verdict = "NOT VALID"
            badge = "🔴 NOT VALID"

        # ==================================================
        # DASHBOARD
        # ==================================================
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"""<div class="metric-card">
            <div class="metric-title">Avg Similarity</div>
            <div class="metric-value">{avg_sim:.3f}</div></div>""", unsafe_allow_html=True)
        col2.markdown(f"""<div class="metric-card">
            <div class="metric-title">Citation Coverage</div>
            <div class="metric-value">{match_count}/{len(ref_lines)}</div></div>""", unsafe_allow_html=True)
        col3.markdown(f"""<div class="metric-card">
            <div class="metric-title">Keyword Score</div>
            <div class="metric-value">{keyword_score}/20</div></div>""", unsafe_allow_html=True)
        col4.markdown(f"""<div class="metric-card">
            <div class="metric-title">Verdict</div>
            <div class="metric-value">{badge}</div></div>""", unsafe_allow_html=True)

        st.markdown(f"**Total Rubric Score:** {total}/100")

        # ==================================================
        # TABS
        # ==================================================
        tab1, tab2, tab3, tab4 = st.tabs([
            "Top Literature",
            "GPT Evaluation",
            "Weaknesses",
            "Rewritten Gap"
        ])

        with tab1:
            st.subheader("Top 10 Relevant Papers")
            st.dataframe(top10[["Title", "Year", "DOI", "similarity"]])

        with tab2:
            st.subheader("GPT Evaluation (Expert Rubric)")
            st.write("### Title")
            st.write(f"Score: {title_score}/10")
            st.write(gpt_eval.get("title_comment", ""))

            st.write("### Clarity of Research Gap")
            st.write(f"Score: {clarity_score}/20")
            st.write(gpt_eval.get("clarity_comment", ""))

            st.write("### Future Contribution")
            st.write(f"Score: {future_score}/20")
            st.write(gpt_eval.get("future_comment", ""))

            st.write("### Originality")
            st.write(f"Score: {originality_score}/20")
            st.write(gpt_eval.get("originality_comment", ""))

            st.write("### Citation Score (from system)")
            st.write(f"{citation_score}/40")

        with tab3:
            st.subheader("Critical Weaknesses")
            for w in gpt_eval.get("weaknesses", []):
                st.write(f"- {w}")
            st.subheader("Suggestions")
            for s in gpt_eval.get("suggestions", []):
                st.write(f"- {s}")

        with tab4:
            st.subheader("Rewritten Research Gap")
            st.write(gpt_eval.get("rewritten_gap", ""))

            st.markdown("---")
            st.subheader("References Used in Rewritten Gap (APA Style)")
            if "references_list" in gpt_eval and gpt_eval["references_list"]:
                for ref in gpt_eval["references_list"]:
                    st.markdown(f"- {ref}")
            else:
                st.info("No reference list returned.")

        # ==================================================
        # PDF DOWNLOAD
        # ==================================================
        pdf_buffer = generate_pdf(
            title_input,
            avg_sim,
            f"{match_count}/{len(ref_lines)}",
            keyword_score,
            verdict,
            gpt_eval
        )

        st.download_button(
            label="📄 Download Full Evaluation as PDF",
            data=pdf_buffer,
            file_name="gap_evaluation_report.pdf",
            mime="application/pdf"
        )
