#############################################################
# AI-BIM / Digital Construction Research Gap Checker
# FULL WORKING VERSION + REVISION-AWARE SCORING (+2.5 uplift at >=70% coverage)
#
# Preserves your original UI/flow and fixes the errors you hit.
# Main additions:
# - objective citation metrics blended into citation score
# - strict GPT reviewer prompt
# - revision-aware uplift:
#     if revised gap covers >=70% of extracted improvement keywords:
#         Novelty +2.5, Significance +2.5, Clarity +1.0 (capped at 10)
#
# NOTE: No pd.DataFrame type hints anywhere (avoids NameError in Streamlit).
#############################################################

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from io import BytesIO
from openai import OpenAI
import difflib
from datetime import datetime

#############################################################
# PAGE CONFIG – QUB BRANDING
#############################################################
st.set_page_config(page_title="AI-BIM Research Gap Checker", layout="wide")

QUB_RED = "#CC0033"
QUB_DARK = "#002147"
QUB_LIGHT = "#F5F5F5"

#############################################################
# CSS FIXED
#############################################################
st.markdown(f"""
<style>
body {{
    background-color: {QUB_LIGHT};
    font-family: Arial, sans-serif;
}}
.header {{
    background-color: {QUB_DARK};
    padding: 25px 40px;
    border-radius: 10px;
    color: white;
}}
</style>
""", unsafe_allow_html=True)

#############################################################
# HEADER
#############################################################
st.markdown(f"""
<div class="header">
<h2>🎓 AI-BIM / Digital Construction Research Gap Checker</h2>
<p>Evaluate research gaps using Top-10 literature, abstracts, Scopus metadata, and GPT reviewer analysis.</p>
</div>
""", unsafe_allow_html=True)

#############################################################
# SIDEBAR
#############################################################
st.sidebar.header("Upload Required Files")

PARQUET = st.sidebar.file_uploader("Upload bert_documents_enriched.parquet", type=["parquet"])
EMB_PATH = st.sidebar.file_uploader("Upload bert_embeddings.npy", type=["npy"])
SCOPUS = st.sidebar.file_uploader("Upload Scopus.csv", type=["csv"])
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

style_choice = st.sidebar.selectbox(
    "Journal Style for Review",
    ["Automation in Construction", "ECAM", "ITcon"]
)

show_debug = st.sidebar.checkbox("Show scoring diagnostics", value=False)
revision_mode = st.sidebar.checkbox("Revision-aware scoring (rewards addressing feedback)", value=True)

if not (PARQUET and EMB_PATH and SCOPUS and api_key):
    st.warning("Please upload all 3 files and enter your API key.")
    st.stop()

client = OpenAI(api_key=api_key)

#############################################################
# LOADERS
#############################################################
@st.cache_resource
def load_docs(parquet_file, emb_file):
    df = pd.read_parquet(parquet_file).fillna("")
    emb = np.load(emb_file)
    return df, emb

@st.cache_resource
def load_scopus(csv_file):
    raw = csv_file.read()
    encodings = ["utf-8", "iso-8859-1", "utf-16"]
    for enc in encodings:
        try:
            df = pd.read_csv(BytesIO(raw), encoding=enc, low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            return df
        except:
            pass
    df = pd.read_csv(BytesIO(raw), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

df_docs, embeddings = load_docs(PARQUET, EMB_PATH)
df_scopus = load_scopus(SCOPUS)

#############################################################
# ALIGN ROW COUNTS
#############################################################
if len(df_docs) != embeddings.shape[0]:
    min_len = min(len(df_docs), embeddings.shape[0])
    st.warning(f"Mismatch detected. Using first {min_len} rows.")
    df_docs = df_docs.iloc[:min_len].reset_index(drop=True)
    embeddings = embeddings[:min_len, :]

#############################################################
# EMBEDDING CALL
#############################################################
def embed_query(text):
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(resp.data[0].embedding)

#############################################################
# APA BUILDER
#############################################################
def build_apa(row):
    authors = row.get("Authors", "")
    year = str(row.get("Year", "n.d."))
    title = row.get("Title", "")
    journal = row.get("Source title", "")
    volume = row.get("Volume", "")
    issue = row.get("Issue", "")
    p1 = str(row.get("Page start", ""))
    p2 = str(row.get("Page end", ""))
    art = str(row.get("Art. No.", ""))
    doi = str(row.get("DOI", ""))

    pages = ""
    if p1 and p2 and p1 != "nan" and p2 != "nan":
        pages = f"{p1}-{p2}"
    elif art and art != "nan":
        pages = f"Article {art}"

    apa = f"{authors} ({year}). {title}. {journal}"
    if volume and volume != "nan":
        apa += f", {volume}"
    if issue and issue != "nan":
        apa += f"({issue})"
    if pages:
        apa += f", {pages}"
    if doi and doi != "nan":
        apa += f". https://doi.org/{doi}"

    return apa

#############################################################
# SIMILARITY
#############################################################
def vector_similarity(qv, mat):
    qn = np.linalg.norm(qv)
    dn = np.linalg.norm(mat, axis=1)
    return mat @ qv / (dn * qn + 1e-9)

#############################################################
# CITATION METRICS (OBJECTIVE SIGNALS)
#############################################################
CURRENT_YEAR = datetime.now().year
DOI_REGEX = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)
YEAR_REGEX = re.compile(r"\b(19\d{2}|20\d{2})\b")

def extract_dois(text):
    if not text:
        return set()
    return set(d.lower().rstrip(".") for d in DOI_REGEX.findall(text))

def extract_years(text):
    if not text:
        return []
    years = [int(y) for y in YEAR_REGEX.findall(text)]
    years = [y for y in years if 1950 <= y <= CURRENT_YEAR]
    return years

def normalize_title(t):
    t = (t or "").lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return t.strip()

def fuzzy_match(user_titles, candidate, cutoff=0.86):
    best = 0.0
    for ut in user_titles:
        r = difflib.SequenceMatcher(None, ut, candidate).ratio()
        best = max(best, r)
    return best >= cutoff, best

def citation_metrics(user_refs_text, top10_df):
    user_dois = extract_dois(user_refs_text)
    user_years = extract_years(user_refs_text)

    recent_3y = sum(1 for y in user_years if y >= CURRENT_YEAR - 2)
    recent_5y = sum(1 for y in user_years if y >= CURRENT_YEAR - 4)

    top10_dois = set()
    if "DOI" in top10_df.columns:
        top10_dois = set(
            str(d).lower().strip()
            for d in top10_df["DOI"].fillna("").tolist()
            if str(d).strip()
        )

    overlap_doi = len(user_dois.intersection(top10_dois))

    user_titles_norm = [
        normalize_title(x)
        for x in re.split(r"\n|•|\r", user_refs_text or "")
        if len(x.strip()) > 20
    ]

    overlap_title = 0
    if "Title" in top10_df.columns:
        for t in top10_df["Title"].fillna("").tolist():
            ok, _ = fuzzy_match(user_titles_norm, normalize_title(t))
            if ok:
                overlap_title += 1

    return {
        "n_refs_lines": len([x for x in (user_refs_text or "").splitlines() if x.strip()]),
        "n_dois_user": len(user_dois),
        "recent_3y_count": recent_3y,
        "recent_5y_count": recent_5y,
        "top10_overlap_doi_count": overlap_doi,
        "top10_overlap_title_count": overlap_title
    }

def citation_score_objective(m):
    score = 2.0

    if m["n_refs_lines"] >= 8:
        score += 1.0
    if m["n_refs_lines"] >= 15:
        score += 0.5

    if m["n_dois_user"] >= 3:
        score += 1.0
    if m["n_dois_user"] >= 8:
        score += 0.5

    if m["recent_3y_count"] >= 3:
        score += 1.5
    if m["recent_3y_count"] >= 6:
        score += 0.5
    if m["recent_5y_count"] >= 6:
        score += 0.5

    if m["top10_overlap_doi_count"] >= 1:
        score += 1.5
    if m["top10_overlap_doi_count"] >= 2:
        score += 0.5
    if m["top10_overlap_title_count"] >= 2:
        score += 0.5

    return min(round(score, 1), 9.0)

#############################################################
# CAPS (KEEP SCORING REALISTIC)
#############################################################
def caps_for_novelty_significance(gap_text):
    gap = (gap_text or "").lower()
    generic_terms = [
        "framework", "interoperability", "lifecycle", "sustainability", "digital twin",
        "circular", "predictive", "explainable", "decision"
    ]
    generic_hits = sum(1 for g in generic_terms if g in gap)

    has_mechanism = any(k in gap for k in ["workflow", "governance", "validation", "traceab", "audit", "operational"])
    has_eval = any(k in gap for k in ["case study", "experiment", "benchmark", "dataset", "evaluation"])

    novelty_cap = 10
    significance_cap = 10

    if generic_hits >= 6 and not has_mechanism:
        novelty_cap = 6
    if not has_eval:
        significance_cap = 6 if not has_mechanism else 7

    return novelty_cap, significance_cap

#############################################################
# REVISION-AWARE SCORING (REWARDS ADDRESSING FEEDBACK)
# RULE: >=70% improvement keywords present in revised gap -> +2.5 novelty/significance
#############################################################
STOPWORDS = set("""
a an the and or but if then else when while to of in on for with without by from as is are was were be been being
this that these those it its their there here into across within between also may might can could should would
""".split())

def tokenize_keywords(text, min_len=4):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t.strip() for t in text.split() if t.strip()]
    toks = [t for t in toks if len(t) >= min_len and t not in STOPWORDS]
    return toks

def extract_improvement_keywords(improvements_list, top_k=18):
    joined = " ".join(improvements_list or [])
    toks = tokenize_keywords(joined)
    freq = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, c in ranked[:top_k]]

def revision_coverage(new_gap_text, prev_keywords):
    gap = (new_gap_text or "").lower()
    found = []
    missing = []
    for kw in (prev_keywords or []):
        if kw in gap:
            found.append(kw)
        else:
            missing.append(kw)
    total = max(len(prev_keywords or []), 1)
    coverage = len(found) / total
    return coverage, found, missing

def apply_revision_uplift(scores_dict, coverage):
    """
    >= 0.70 coverage -> strong reward:
        Novelty +2.5
        Significance +2.5
        Clarity +1.0
    """
    novelty_uplift = 0.0
    significance_uplift = 0.0
    clarity_uplift = 0.0

    if coverage >= 0.70:
        novelty_uplift = 2.5
        significance_uplift = 2.5
        clarity_uplift = 1.0
    elif coverage >= 0.40:
        novelty_uplift = 1.2
        significance_uplift = 1.0
        clarity_uplift = 0.6
    elif coverage >= 0.25:
        novelty_uplift = 0.6
        significance_uplift = 0.4
        clarity_uplift = 0.4

    # apply safely with caps
    try:
        scores_dict["novelty_score"] = min(10, round(float(scores_dict.get("novelty_score", 0)) + novelty_uplift, 1))
    except:
        pass
    try:
        scores_dict["significance_score"] = min(10, round(float(scores_dict.get("significance_score", 0)) + significance_uplift, 1))
    except:
        pass
    try:
        scores_dict["clarity_score"] = min(10, round(float(scores_dict.get("clarity_score", 0)) + clarity_uplift, 1))
    except:
        pass

    return {
        "coverage": round(coverage, 2),
        "novelty_uplift": novelty_uplift,
        "significance_uplift": significance_uplift,
        "clarity_uplift": clarity_uplift
    }

#############################################################
# GPT REVIEW (UPDATED "GUTS": evidence-based + metrics-aware)
#############################################################
def gpt_review(title, gap, refs, top10_titles, top10_abstracts, style_choice, cite_m):

    combined_abstracts = "\n\n".join(
        [f"PAPER {i+1}:\nTITLE: {t}\nABSTRACT:\n{a}"
         for i, (t, a) in enumerate(zip(top10_titles, top10_abstracts))]
    )

    prompt = f"""
You are a senior academic reviewer writing in the style of {style_choice}.
Evaluate the STUDENT GAP only. Do not rewrite it.

Return STRICT JSON ONLY:
{{
  "novelty_score": 0,
  "significance_score": 0,
  "clarity_score": 0,
  "citation_score": 0,
  "good_points": [],
  "improvements": [],
  "novelty_comment": "",
  "significance_comment": "",
  "citation_comment": "",
  "evidence_papers": {{
    "overlap": [],
    "supporting": [],
    "missing": []
  }}
}}

SCORING RULES:
- 8–10 extremely rare and only for publication-ready work.
- Typical student gaps: 3–6.

NOVELTY: judge against TOP-10 abstracts, cite paper numbers in evidence_papers.overlap.
SIGNIFICANCE: high only if solving the gap materially changes research/practice.
CLARITY: focus, precision, logical structure.
CITATION QUALITY: use the metrics provided; keep <=6 if weak.

If any score > 7, justify it strongly; otherwise reduce to 7 or below.

CITATION METRICS:
{json.dumps(cite_m, ensure_ascii=False)}

STUDENT TITLE:
{title}

STUDENT GAP:
{gap}

REFERENCES:
{refs}

TOP-10 ABSTRACTS:
{combined_abstracts}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content

    try:
        return json.loads(raw)
    except:
        cleaned = raw[raw.find("{"): raw.rfind("}") + 1]
        try:
            return json.loads(cleaned)
        except:
            return {
                "novelty_score": 0,
                "significance_score": 0,
                "clarity_score": 0,
                "citation_score": 0,
                "good_points": [],
                "improvements": [],
                "novelty_comment": "",
                "significance_comment": "",
                "citation_comment": "",
                "evidence_papers": {"overlap": [], "supporting": [], "missing": []}
            }

#############################################################
# INPUT BOXES
#############################################################
st.markdown("### 📌 Dissertation Title")
title = st.text_input("", placeholder="Enter dissertation title")

st.markdown("### 📌 Research Gap")
gap = st.text_area("", height=200, placeholder="Paste the research gap here")

st.markdown("### 📌 APA References")
refs = st.text_area("", height=200, placeholder="Paste APA references here")

#############################################################
# RUN BUTTON
#############################################################
if st.button("Run Evaluation"):
    with st.spinner("Processing..."):

        q_vec = embed_query(f"{title} {gap} {refs}")
        sims = vector_similarity(q_vec, embeddings)
        df_docs["similarity"] = sims

        top10 = df_docs.sort_values("similarity", ascending=False).head(10)
        top10_titles = top10["Title"].tolist()
        top10_abstracts = top10["Abstract"].fillna("").tolist()

        apa_list = []
        for t in top10_titles:
            row = df_scopus[df_scopus["Title"] == t]
            apa_list.append(build_apa(row.iloc[0]) if len(row) else f"{t} (metadata not found)")

        # Citation metrics + blended citation score
        cite_m = citation_metrics(refs, top10)
        cite_obj = citation_score_objective(cite_m)

        # GPT review
        gpt_out = gpt_review(title, gap, refs, top10_titles, top10_abstracts, style_choice, cite_m)

        try:
            cite_gpt = float(gpt_out.get("citation_score", 0))
        except:
            cite_gpt = 0.0
        gpt_out["citation_score"] = round(0.6 * cite_obj + 0.4 * cite_gpt, 1)

        # Apply conservative caps
        nov_cap, sig_cap = caps_for_novelty_significance(gap)
        try:
            gpt_out["novelty_score"] = min(float(gpt_out.get("novelty_score", 0)), nov_cap)
        except:
            gpt_out["novelty_score"] = 0
        try:
            gpt_out["significance_score"] = min(float(gpt_out.get("significance_score", 0)), sig_cap)
        except:
            gpt_out["significance_score"] = 0

        # Revision-aware uplift based on previous improvements
        revision_info = {"enabled": False}

        if revision_mode and "prev_run" in st.session_state:
            prev = st.session_state["prev_run"]
            prev_improvements = prev.get("improvements", [])
            prev_keywords = extract_improvement_keywords(prev_improvements, top_k=18)

            cov, found, missing = revision_coverage(gap, prev_keywords)
            uplift_info = apply_revision_uplift(gpt_out, cov)

            revision_info = {
                "enabled": True,
                "coverage": uplift_info["coverage"],
                "keywords": prev_keywords,
                "found": found,
                "missing": missing,
                "uplifts": uplift_info
            }

        # Store current run
        st.session_state["prev_run"] = {
            "title": title,
            "gap": gap,
            "refs": refs,
            "scores": {
                "novelty": gpt_out.get("novelty_score", 0),
                "significance": gpt_out.get("significance_score", 0),
                "clarity": gpt_out.get("clarity_score", 0),
                "citation": gpt_out.get("citation_score", 0),
            },
            "improvements": gpt_out.get("improvements", []),
            "good_points": gpt_out.get("good_points", []),
            "cite_metrics": cite_m
        }

        #############################################################
        # METRIC CARDS
        #############################################################
        col1, col2, col3, col4 = st.columns(4)

        card_html = """
        <div style="background-color:white; border-left:6px solid #CC0033;
                    padding:16px; border-radius:12px; border:1px solid #ddd;
                    text-align:center; margin-bottom:10px;">
            <div style="font-size:16px; font-weight:600; color:#002147;">{title}</div>
            <div style="font-size:28px; font-weight:700; color:#CC0033;">{value}</div>
        </div>
        """

        col1.markdown(card_html.format(title="Novelty", value=f"{gpt_out['novelty_score']}/10"), unsafe_allow_html=True)
        col2.markdown(card_html.format(title="Significance", value=f"{gpt_out['significance_score']}/10"), unsafe_allow_html=True)
        col3.markdown(card_html.format(title="Clarity", value=f"{gpt_out['clarity_score']}/10"), unsafe_allow_html=True)
        col4.markdown(card_html.format(title="Citation Quality", value=f"{gpt_out['citation_score']}/10"), unsafe_allow_html=True)

        #############################################################
        # RESULTS TABS
        #############################################################
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📚 Top 10 Literature",
            "⭐ Good Points",
            "🚧 Improvements",
            "🔎 Reviewer Comments",
            "📑 APA References"
        ])

        with tab1:
            cols = [c for c in ["Title", "Year", "DOI", "similarity"] if c in top10.columns]
            st.write(top10[cols])

        with tab2:
            for p in gpt_out.get("good_points", []):
                st.write("•", p)

        with tab3:
            for p in gpt_out.get("improvements", []):
                st.write("•", p)

        with tab4:
            st.write("### Novelty Comment")
            st.write(gpt_out.get("novelty_comment", ""))

            st.write("### Significance Comment")
            st.write(gpt_out.get("significance_comment", ""))

            st.write("### Citation Comment")
            st.write(gpt_out.get("citation_comment", ""))

            if revision_mode and revision_info.get("enabled"):
                st.write("### Revision-aware scoring")
                st.write(f"Coverage of previous improvement keywords: {revision_info.get('coverage', 0)}")
                st.write("Applied uplifts:")
                st.json(revision_info.get("uplifts", {}))

                if show_debug:
                    st.write("Keywords extracted from previous improvements:")
                    st.write(revision_info.get("keywords", []))
                    st.write("Found in revised gap:")
                    st.write(revision_info.get("found", []))
                    st.write("Still missing:")
                    st.write(revision_info.get("missing", []))

            if show_debug:
                st.write("### Evidence (Paper numbers)")
                st.json(gpt_out.get("evidence_papers", {}))

                st.write("### Citation Metrics (objective)")
                st.json(cite_m)

                st.write("Objective citation score (0–9):", cite_obj)
                st.write("Novelty cap applied:", nov_cap)
                st.write("Significance cap applied:", sig_cap)

        with tab5:
            st.subheader("APA References")
            for ref in apa_list:
                st.markdown(f"<p>• {ref}</p>", unsafe_allow_html=True)
