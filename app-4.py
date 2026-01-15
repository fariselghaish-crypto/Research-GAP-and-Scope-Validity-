#############################################################
# AI-BIM / Digital Construction Research Gap Checker
# FULL WORKING VERSION + FIXED NOVELTY/SIGNIFICANCE SCORING
#
# What was fixed:
# 1) GPT output is forced into JSON mode when possible (response_format=json_object).
# 2) Prompt is shortened (abstracts truncated) to prevent JSON truncation.
# 3) GPT no longer "guesses" novelty/significance as final numbers.
#    Instead, GPT returns STRUCTURED SIGNALS + overlap/missing evidence,
#    then the app computes novelty/significance deterministically.
# 4) Caps are softened: instead of hard ceilings, we apply small penalties when generic.
# 5) Debug now shows:
#    - raw GPT JSON (if requested)
#    - signals used to compute scores
#    - deterministic scoring breakdown
#
# Your original UI/flow preserved as much as possible.
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
# CSS
#############################################################
st.markdown(
    f"""
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
""",
    unsafe_allow_html=True,
)

#############################################################
# HEADER
#############################################################
st.markdown(
    f"""
<div class="header">
<h2>🎓 AI-BIM / Digital Construction Research Gap Checker</h2>
<p>Evaluate research gaps using Top-10 literature, abstracts, Scopus metadata, and GPT reviewer analysis.</p>
</div>
""",
    unsafe_allow_html=True,
)

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
    ["Automation in Construction", "ECAM", "ITcon"],
)

show_debug = st.sidebar.checkbox("Show scoring diagnostics", value=False)
revision_mode = st.sidebar.checkbox("Revision-aware scoring (B + D)", value=True)

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
        except Exception:
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
    resp = client.embeddings.create(model="text-embedding-3-large", input=text)
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
# OBJECTIVE CITATION METRICS
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
        "top10_overlap_title_count": overlap_title,
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
# B: SOFTER GENERICITY PENALTY (NOT HARD CAPS)
#############################################################
GENERIC_TERMS = [
    "framework",
    "interoperability",
    "lifecycle",
    "sustainability",
    "digital twin",
    "circular",
    "predictive",
    "explainable",
    "decision",
]


def genericity_penalty(gap_text):
    gap = (gap_text or "").lower()
    generic_hits = sum(1 for g in GENERIC_TERMS if g in gap)

    has_mechanism = any(k in gap for k in ["workflow", "governance", "validation", "traceab", "audit", "operational"])
    has_eval = any(k in gap for k in ["case study", "experiment", "benchmark", "dataset", "evaluation"])

    # small penalties (soft)
    nov_pen = 0.0
    sig_pen = 0.0

    # broad + no mechanism -> likely "generic novelty"
    if generic_hits >= 6 and not has_mechanism:
        nov_pen += 1.2

    # no evaluation signals -> significance should be reduced a bit
    if not has_eval:
        sig_pen += 1.2 if not has_mechanism else 0.8

    return {
        "generic_hits": generic_hits,
        "has_mechanism": has_mechanism,
        "has_eval": has_eval,
        "novelty_penalty": nov_pen,
        "significance_penalty": sig_pen,
    }


#############################################################
# REVISION-AWARE KEYWORD COVERAGE (UNCHANGED)
#############################################################
STOPWORDS = set(
    """
a an the and or but if then else when while to of in on for with without by from as is are was were be been being
this that these those it its their there here into across within between also may might can could should would
""".split()
)


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


def revision_tier(coverage):
    if coverage >= 0.70:
        return "major_revision"
    if coverage >= 0.40:
        return "minor_revision"
    return "no_revision"


def apply_revision_uplift(scores_dict, coverage):
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

    # bounded
    try:
        scores_dict["novelty_score"] = min(10, round(float(scores_dict.get("novelty_score", 0)) + novelty_uplift, 1))
    except Exception:
        pass
    try:
        scores_dict["significance_score"] = min(
            10, round(float(scores_dict.get("significance_score", 0)) + significance_uplift, 1)
        )
    except Exception:
        pass
    try:
        scores_dict["clarity_score"] = min(10, round(float(scores_dict.get("clarity_score", 0)) + clarity_uplift, 1))
    except Exception:
        pass

    return {
        "coverage": round(coverage, 2),
        "novelty_uplift": novelty_uplift,
        "significance_uplift": significance_uplift,
        "clarity_uplift": clarity_uplift,
    }


#############################################################
# GPT REVIEW (SIGNALS-FIRST) + SHORTER CONTEXT + JSON MODE
#############################################################
def _truncate(text, max_chars=1200):
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 20].rstrip() + " ...[truncated]"


def gpt_review_signals(title, gap, refs, top10_titles, top10_abstracts, style_choice, cite_m):
    # Keep context smaller to avoid truncated JSON outputs
    blocks = []
    for i, (t, a) in enumerate(zip(top10_titles, top10_abstracts)):
        blocks.append(
            f"PAPER {i+1}:\nTITLE: {t}\nABSTRACT (truncated):\n{_truncate(a, 900)}"
        )
    combined_abstracts = "\n\n".join(blocks)

    # JSON schema we expect (keep minimal and stable)
    expected = {
        "signals": {
            "clear_point_of_departure": False,
            "explicit_missing_mechanism": False,
            "explicit_missing_validation_or_evaluation": False,
            "explicit_boundary_or_scope": False,
            "explicit_contribution_claim": False,
            "has_evaluation_design": False,
            "has_metrics_or_measures": False,
            "has_stakeholder_or_practical_impact": False,
            "has_generalisability_or_scalability": False,
            "clarity_good_structure": False,
            "clarity_defines_terms": False,
            "clarity_low_ambiguity": False,
        },
        "evidence_papers": {
            "overlap": [],     # list of paper numbers
            "supporting": [],  # list of paper numbers
            "missing": [],     # list of paper numbers
        },
        "good_points": [],
        "improvements": [],
        "novelty_comment": "",
        "significance_comment": "",
        "citation_comment": "",
        "clarity_comment": "",
        "citation_score": 0,  # GPT's view, later blended
    }

    prompt = f"""
You are a senior academic reviewer writing in the style of {style_choice}.
Evaluate ONLY the STUDENT GAP. Do not rewrite it.

Return STRICT JSON ONLY with this exact structure and keys. Do not add extra keys.
If uncertain, keep booleans as false.

JSON TEMPLATE:
{json.dumps(expected, ensure_ascii=False)}

SCORING/LOGIC RULES:
- Do NOT output novelty_score or significance_score. The system will compute them.
- You MUST justify overlap/support/missing with paper numbers (1-10) based on TOP-10 abstracts.
- Set signal booleans to true ONLY if the STUDENT GAP explicitly contains that element.

CITATION METRICS (objective):
{json.dumps(cite_m, ensure_ascii=False)}

STUDENT TITLE:
{title}

STUDENT GAP:
{gap}

REFERENCES:
{refs}

TOP-10 ABSTRACTS:
{combined_abstracts}
""".strip()

    # Try JSON mode if supported by the model/endpoint; fallback if not.
    raw = ""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            temperature=0,
            max_tokens=2200,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content
        return json.loads(raw), raw
    except Exception:
        # fallback: no response_format
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                temperature=0,
                max_tokens=2200,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content
            # robust extraction of JSON object
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start : end + 1]), raw
            return expected, raw
        except Exception:
            return expected, raw


#############################################################
# DETERMINISTIC NOVELTY / SIGNIFICANCE / CLARITY SCORING
#############################################################
def _clamp(v, lo=0.0, hi=10.0):
    try:
        return float(min(hi, max(lo, v)))
    except Exception:
        return lo


def compute_scores_from_signals(signals_dict, evidence_papers, gap_text):
    s = signals_dict or {}
    ev = evidence_papers or {"overlap": [], "supporting": [], "missing": []}

    overlap_n = len(ev.get("overlap", []) or [])
    missing_n = len(ev.get("missing", []) or [])

    # Novelty rubric (0-10)
    novelty = 2.0
    breakdown_n = []

    if s.get("clear_point_of_departure"):
        novelty += 1.8
        breakdown_n.append("+1.8 clear point of departure")
    if s.get("explicit_missing_mechanism"):
        novelty += 2.0
        breakdown_n.append("+2.0 explicit missing mechanism")
    if s.get("explicit_missing_validation_or_evaluation"):
        novelty += 1.4
        breakdown_n.append("+1.4 explicit missing validation/evaluation")
    if s.get("explicit_boundary_or_scope"):
        novelty += 1.0
        breakdown_n.append("+1.0 explicit boundary/scope")
    if s.get("explicit_contribution_claim"):
        novelty += 1.0
        breakdown_n.append("+1.0 explicit contribution claim")

    # Evidence adjustment: many overlaps reduce novelty; missing papers increase confidence of novelty claim
    if overlap_n >= 3:
        novelty -= 1.0
        breakdown_n.append("-1.0 high overlap (>=3 papers)")
    elif overlap_n == 2:
        novelty -= 0.6
        breakdown_n.append("-0.6 moderate overlap (2 papers)")
    elif overlap_n == 1:
        novelty -= 0.3
        breakdown_n.append("-0.3 some overlap (1 paper)")

    if missing_n >= 3:
        novelty += 0.6
        breakdown_n.append("+0.6 multiple missing papers (>=3)")
    elif missing_n == 2:
        novelty += 0.4
        breakdown_n.append("+0.4 missing papers (2)")
    elif missing_n == 1:
        novelty += 0.2
        breakdown_n.append("+0.2 missing papers (1)")

    # Soft genericity penalty
    gp = genericity_penalty(gap_text)
    if gp["novelty_penalty"] > 0:
        novelty -= gp["novelty_penalty"]
        breakdown_n.append(f"-{gp['novelty_penalty']:.1f} genericity penalty")

    novelty = _clamp(round(novelty, 1))

    # Significance rubric (0-10)
    significance = 2.0
    breakdown_s = []

    if s.get("has_evaluation_design"):
        significance += 2.2
        breakdown_s.append("+2.2 has evaluation design")
    if s.get("has_metrics_or_measures"):
        significance += 1.2
        breakdown_s.append("+1.2 has metrics/measures")
    if s.get("has_stakeholder_or_practical_impact"):
        significance += 1.4
        breakdown_s.append("+1.4 stakeholder/practical impact")
    if s.get("has_generalisability_or_scalability"):
        significance += 1.0
        breakdown_s.append("+1.0 generalisability/scalability")
    if s.get("explicit_boundary_or_scope"):
        significance += 0.6
        breakdown_s.append("+0.6 clear scope supports feasibility")

    # If no evaluation design at all, significance should stay low
    if not s.get("has_evaluation_design"):
        significance -= 0.6
        breakdown_s.append("-0.6 no evaluation design")

    # Soft genericity penalty
    if gp["significance_penalty"] > 0:
        significance -= gp["significance_penalty"]
        breakdown_s.append(f"-{gp['significance_penalty']:.1f} genericity penalty")

    significance = _clamp(round(significance, 1))

    # Clarity rubric (0-10)
    clarity = 2.0
    breakdown_c = []
    if s.get("clarity_good_structure"):
        clarity += 2.2
        breakdown_c.append("+2.2 good structure")
    if s.get("clarity_defines_terms"):
        clarity += 1.6
        breakdown_c.append("+1.6 defines terms")
    if s.get("clarity_low_ambiguity"):
        clarity += 1.8
        breakdown_c.append("+1.8 low ambiguity")

    # If the gap has explicit boundary/scope, clarity benefits
    if s.get("explicit_boundary_or_scope"):
        clarity += 0.8
        breakdown_c.append("+0.8 clear boundary/scope")

    clarity = _clamp(round(clarity, 1))

    return {
        "novelty_score": novelty,
        "significance_score": significance,
        "clarity_score": clarity,
        "breakdown": {
            "novelty": breakdown_n,
            "significance": breakdown_s,
            "clarity": breakdown_c,
            "genericity": gp,
            "evidence_counts": {"overlap_n": overlap_n, "missing_n": missing_n},
        },
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
        # Reduce retrieval noise: do NOT let references dominate similarity
        # (title+gap repeated acts as weight)
        q_vec = embed_query(f"{title}\n{gap}\n{title}\n{gap}")
        sims = vector_similarity(q_vec, embeddings)
        df_docs["similarity"] = sims

        top10 = df_docs.sort_values("similarity", ascending=False).head(10)
        top10_titles = top10["Title"].tolist()
        top10_abstracts = top10["Abstract"].fillna("").tolist()

        apa_list = []
        for t in top10_titles:
            row = df_scopus[df_scopus["Title"] == t]
            apa_list.append(build_apa(row.iloc[0]) if len(row) else f"{t} (metadata not found)")

        # Objective citation metrics + objective citation score
        cite_m = citation_metrics(refs, top10)
        cite_obj = citation_score_objective(cite_m)

        # GPT review -> signals + evidence + comments
        gpt_out, gpt_raw = gpt_review_signals(title, gap, refs, top10_titles, top10_abstracts, style_choice, cite_m)
        if show_debug:
            st.session_state["last_gpt_raw"] = gpt_raw

        # Deterministic novelty/significance/clarity from signals
        signals = (gpt_out or {}).get("signals", {}) or {}
        evidence_papers = (gpt_out or {}).get("evidence_papers", {}) or {}
        computed = compute_scores_from_signals(signals, evidence_papers, gap)

        # Use deterministic scores
        final_scores = {
            "novelty_score": computed["novelty_score"],
            "significance_score": computed["significance_score"],
            "clarity_score": computed["clarity_score"],
        }

        # Citation score is still blended (objective dominates)
        try:
            cite_gpt = float((gpt_out or {}).get("citation_score", 0))
        except Exception:
            cite_gpt = 0.0
        final_scores["citation_score"] = round(0.6 * cite_obj + 0.4 * cite_gpt, 1)

        # Attach comments, good_points, improvements, evidence
        final_payload = {
            **final_scores,
            "good_points": (gpt_out or {}).get("good_points", []) or [],
            "improvements": (gpt_out or {}).get("improvements", []) or [],
            "novelty_comment": (gpt_out or {}).get("novelty_comment", "") or "",
            "significance_comment": (gpt_out or {}).get("significance_comment", "") or "",
            "citation_comment": (gpt_out or {}).get("citation_comment", "") or "",
            "clarity_comment": (gpt_out or {}).get("clarity_comment", "") or "",
            "evidence_papers": evidence_papers,
            "signals": signals,
            "score_breakdown": computed.get("breakdown", {}),
        }

        #############################################################
        # REVISION COVERAGE (computed BEFORE uplift)
        #############################################################
        revision_info = {"enabled": False, "tier": "no_revision"}

        cov = 0.0
        found = []
        missing = []
        prev_keywords = []

        if revision_mode and "prev_run" in st.session_state:
            prev = st.session_state["prev_run"]
            prev_improvements = prev.get("improvements", [])
            prev_keywords = extract_improvement_keywords(prev_improvements, top_k=18)
            cov, found, missing = revision_coverage(gap, prev_keywords)

        tier = revision_tier(cov)

        # Apply deterministic revision uplift after computed scores
        uplift_info = {"coverage": round(cov, 2), "novelty_uplift": 0, "significance_uplift": 0, "clarity_uplift": 0}
        if revision_mode and tier in ["major_revision", "minor_revision"]:
            uplift_info = apply_revision_uplift(final_payload, cov)

        revision_info = {
            "enabled": revision_mode and ("prev_run" in st.session_state),
            "tier": tier,
            "coverage": round(cov, 2),
            "keywords": prev_keywords,
            "found": found,
            "missing": missing,
            "uplifts": uplift_info,
        }

        # Store current run for next attempt
        st.session_state["prev_run"] = {
            "title": title,
            "gap": gap,
            "refs": refs,
            "scores": {
                "novelty": final_payload.get("novelty_score", 0),
                "significance": final_payload.get("significance_score", 0),
                "clarity": final_payload.get("clarity_score", 0),
                "citation": final_payload.get("citation_score", 0),
            },
            "improvements": final_payload.get("improvements", []),
            "good_points": final_payload.get("good_points", []),
            "cite_metrics": cite_m,
            "signals": final_payload.get("signals", {}),
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

        col1.markdown(
            card_html.format(title="Novelty", value=f"{final_payload.get('novelty_score', 0)}/10"),
            unsafe_allow_html=True,
        )
        col2.markdown(
            card_html.format(title="Significance", value=f"{final_payload.get('significance_score', 0)}/10"),
            unsafe_allow_html=True,
        )
        col3.markdown(
            card_html.format(title="Clarity", value=f"{final_payload.get('clarity_score', 0)}/10"),
            unsafe_allow_html=True,
        )
        col4.markdown(
            card_html.format(title="Citation Quality", value=f"{final_payload.get('citation_score', 0)}/10"),
            unsafe_allow_html=True,
        )

        #############################################################
        # RESULTS TABS
        #############################################################
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📚 Top 10 Literature",
                "⭐ Good Points",
                "🚧 Improvements",
                "🔎 Reviewer Comments",
                "📑 APA References",
            ]
        )

        with tab1:
            cols = [c for c in ["Title", "Year", "DOI", "similarity"] if c in top10.columns]
            st.write(top10[cols])

        with tab2:
            for p in final_payload.get("good_points", []):
                st.write("•", p)

        with tab3:
            for p in final_payload.get("improvements", []):
                st.write("•", p)

        with tab4:
            st.write("### Novelty Comment")
            st.write(final_payload.get("novelty_comment", ""))

            st.write("### Significance Comment")
            st.write(final_payload.get("significance_comment", ""))

            st.write("### Clarity Comment")
            st.write(final_payload.get("clarity_comment", ""))

            st.write("### Citation Comment")
            st.write(final_payload.get("citation_comment", ""))

            st.write("### Revision reassessment (B + D)")
            st.write(f"Revision tier: **{revision_info.get('tier')}**")
            st.write(f"Keyword coverage: **{revision_info.get('coverage')}**")
            st.write("Uplifts applied:", revision_info.get("uplifts", {}))

            if show_debug:
                st.write("### Deterministic scoring signals")
                st.json(final_payload.get("signals", {}))

                st.write("### Deterministic scoring breakdown")
                st.json(final_payload.get("score_breakdown", {}))

                st.write("### Evidence (Paper numbers)")
                st.json(final_payload.get("evidence_papers", {}))

                st.write("### Citation Metrics (objective)")
                st.json(cite_m)
                st.write("Objective citation score (0–9):", cite_obj)

                st.write("### GPT raw output (last call)")
                st.write(st.session_state.get("last_gpt_raw", ""))

                st.write("### Revision keyword diagnostics")
                st.write("Keywords extracted from previous improvements:")
                st.write(revision_info.get("keywords", []))
                st.write("Found in revised gap:")
                st.write(revision_info.get("found", []))
                st.write("Still missing:")
                st.write(revision_info.get("missing", []))

        with tab5:
            st.subheader("APA References")
            for ref in apa_list:
                st.markdown(f"<p>• {ref}</p>", unsafe_allow_html=True)
