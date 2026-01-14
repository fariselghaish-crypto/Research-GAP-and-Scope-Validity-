#############################################################
# ENHANCEMENTS FOR MORE RELIABLE SCORING (NOVELTY / SIGNIFICANCE / CITATION)
# Drop-in updates: (A) compute objective citation metrics, (B) force GPT to
# ground scores in Top-10 evidence, (C) combine GPT score + metric-based score.
#############################################################

import difflib
from datetime import datetime

CURRENT_YEAR = datetime.now().year  # will use runtime year

#############################################################
# 1) STRONGER REFERENCE PARSING + METRICS (RECENCY, DOI COVERAGE, TOP-10 OVERLAP)
#############################################################
DOI_REGEX = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)
YEAR_REGEX = re.compile(r"\b(19\d{2}|20\d{2})\b")

def extract_dois(text: str):
    return list({d.lower().rstrip(".") for d in DOI_REGEX.findall(text or "")})

def extract_years(text: str):
    years = [int(y) for y in YEAR_REGEX.findall(text or "")]
    # remove obvious false positives outside plausible academic window
    years = [y for y in years if 1950 <= y <= CURRENT_YEAR]
    return years

def normalize_title(t: str):
    t = (t or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return t

def fuzzy_title_match(user_titles_norm, cand_norm, cutoff=0.86):
    # best match ratio
    best = 0.0
    for ut in user_titles_norm:
        r = difflib.SequenceMatcher(None, ut, cand_norm).ratio()
        if r > best:
            best = r
    return best >= cutoff, best

def citation_metrics(user_refs_text: str, top10_df: pd.DataFrame):
    user_dois = set(extract_dois(user_refs_text))
    user_years = extract_years(user_refs_text)

    # recency
    recent_3y = sum(1 for y in user_years if y >= CURRENT_YEAR - 2)  # last 3 yrs inclusive
    recent_5y = sum(1 for y in user_years if y >= CURRENT_YEAR - 4)

    # top10 overlap via DOI
    top10_dois = set([str(d).lower().strip() for d in top10_df.get("DOI", []).fillna("").tolist() if str(d).strip()])
    overlap_doi = len(user_dois.intersection(top10_dois))

    # top10 overlap via title fuzzy match (backup)
    user_titles_norm = [normalize_title(x) for x in re.split(r"\n|•|\r", user_refs_text or "") if len(x.strip()) > 20]
    top10_titles = top10_df.get("Title", pd.Series([])).fillna("").tolist()
    top10_titles_norm = [normalize_title(t) for t in top10_titles]

    overlap_title = 0
    best_title_matches = []
    for i, cand_norm in enumerate(top10_titles_norm):
        ok, score = fuzzy_title_match(user_titles_norm, cand_norm)
        if ok:
            overlap_title += 1
            best_title_matches.append((i+1, top10_titles[i], round(score, 3)))

    metrics = {
        "n_refs_lines": len([x for x in (user_refs_text or "").splitlines() if x.strip()]),
        "n_dois_user": len(user_dois),
        "n_years_found": len(user_years),
        "recent_3y_count": recent_3y,
        "recent_5y_count": recent_5y,
        "top10_overlap_doi_count": overlap_doi,
        "top10_overlap_title_count": overlap_title,
        "top10_title_matches": best_title_matches[:10],  # keep short
        "top10_doi_list": sorted(list(top10_dois))[:10], # debug aid (short)
    }
    return metrics


#############################################################
# 2) METRIC-BASED CITATION SCORE (0–10) + NOVELTY/SIGNIFICANCE HEURISTICS
#############################################################
def clamp(x, lo=0, hi=10):
    return max(lo, min(hi, x))

def score_citations_from_metrics(m):
    """
    Conservative scoring:
    - Recency matters (last 3 years)
    - Direct relevance: overlap with top-10 (DOI/title)
    - DOI coverage (traceability)
    """
    base = 2.0

    # volume proxy (avoid rewarding spam)
    if m["n_refs_lines"] >= 8: base += 1.0
    if m["n_refs_lines"] >= 15: base += 0.5

    # DOI coverage
    if m["n_dois_user"] >= 3: base += 1.0
    if m["n_dois_user"] >= 8: base += 0.5

    # Recency
    if m["recent_3y_count"] >= 3: base += 1.5
    if m["recent_3y_count"] >= 6: base += 0.5
    if m["recent_5y_count"] >= 6: base += 0.5

    # Overlap with top10
    # DOI overlap is strongest, title overlap is weaker
    if m["top10_overlap_doi_count"] >= 1: base += 1.5
    if m["top10_overlap_doi_count"] >= 2: base += 0.5
    if m["top10_overlap_title_count"] >= 2: base += 0.5

    # Cap hard: a perfect 10 is extremely rare
    return clamp(round(base, 1), 0, 9)


def novelty_significance_heuristics(gap_text: str, top10_df: pd.DataFrame):
    """
    Light-touch heuristics to reduce inflated GPT scoring:
    - If gap reuses common phrases heavily, reduce novelty ceiling
    - If gap lacks explicit 'missing mechanism' + 'measurable contribution', reduce significance ceiling
    """
    gap_l = (gap_text or "").lower()

    common_claims = [
        "interoperability", "lifecycle", "sustainability", "digital twin",
        "framework", "decision-making", "circular", "explainable", "predictive"
    ]
    hits = sum(1 for w in common_claims if w in gap_l)

    # crude specificity signals
    has_mechanism = any(k in gap_l for k in ["operationalis", "implement", "workflow", "governance", "validation", "traceab", "audit"])
    has_eval = any(k in gap_l for k in ["evaluate", "experiment", "case study", "benchmark", "metrics", "dataset", "protocol"])

    novelty_cap = 10
    significance_cap = 10

    if hits >= 6 and not has_mechanism:
        novelty_cap = 6
    if not has_eval:
        significance_cap = 7 if has_mechanism else 6

    return novelty_cap, significance_cap


#############################################################
# 3) UPDATE GPT PROMPT: EVIDENCE-BASED SCORING + REQUIRED CITED PAPER NUMBERS
#############################################################
def gpt_review(title, gap, refs, top10_titles, top10_abstracts, style_choice, cite_metrics):

    combined_abstracts = "\n\n".join(
        [f"PAPER {i+1}:\nTITLE: {t}\nABSTRACT:\n{a}"
         for i, (t, a) in enumerate(zip(top10_titles, top10_abstracts))]
    )

    prompt = f"""
You are a senior academic reviewer for {style_choice}.
Your job is to SCORE the gap with high reliability.

CRITICAL RULES:
1) Ground your scoring in evidence from the TOP-10 abstracts. If you claim overlap, cite paper numbers.
2) Do NOT rewrite the gap. Only evaluate.
3) Use the full scale conservatively. Scores 8–10 are extremely rare.
4) Novelty and citation scores MUST explicitly reference TOP-10 Paper numbers to justify the score.
5) If evidence is insufficient, lower the score.

Return STRICT JSON ONLY (no markdown, no extra text):
{{
  "novelty_score": 0,
  "significance_score": 0,
  "clarity_score": 0,
  "citation_score": 0,
  "evidence": {{
    "overlap_papers": [1,2],
    "supporting_papers": [5,10],
    "missing_recent_citations": true
  }},
  "good_points": ["..."],
  "improvements": ["..."],
  "novelty_comment": "Must cite Paper numbers and why overlap exists or not.",
  "significance_comment": "State why it is incremental vs transformative, with rationale.",
  "citation_comment": "Refer to citation metrics and Paper numbers to recommend additions."
}}

SCORING RUBRIC:
- Novelty (0–10): assess distinctiveness vs TOP-10.
- Significance (0–10): assess value if solved; avoid inflated scores for well-known issues.
- Clarity (0–10): structure, precision, and lack of vagueness.
- Citation quality (0–10): recency + relevance + linkage to TOP-10.

REFERENCE METRICS (objective signals):
{json.dumps(cite_metrics, ensure_ascii=False)}

STUDENT TITLE:
{title}

STUDENT GAP:
{gap}

REFERENCES PROVIDED:
{refs}

TOP-10 MOST RELEVANT ABSTRACTS:
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
        out = json.loads(raw)
    except:
        cleaned = raw[raw.find("{"): raw.rfind("}")+1]
        out = json.loads(cleaned) if cleaned else {}

    # harden output defaults
    out = out if isinstance(out, dict) else {}
    for k in ["novelty_score","significance_score","clarity_score","citation_score"]:
        out.setdefault(k, 0)

    out.setdefault("evidence", {"overlap_papers": [], "supporting_papers": [], "missing_recent_citations": False})
    out.setdefault("good_points", [])
    out.setdefault("improvements", [])
    out.setdefault("novelty_comment", "")
    out.setdefault("significance_comment", "")
    out.setdefault("citation_comment", "")

    return out


#############################################################
# 4) COMBINE GPT + METRICS INTO FINAL SCORES (REDUCE BIAS / INFLATION)
#############################################################
def combine_scores(gpt_out, cite_m, gap_text, top10_df):
    # objective citation score
    citation_obj = score_citations_from_metrics(cite_m)

    # caps to reduce inflated novelty/significance when vague
    novelty_cap, significance_cap = novelty_significance_heuristics(gap_text, top10_df)

    novelty = min(float(gpt_out.get("novelty_score", 0)), novelty_cap)
    significance = min(float(gpt_out.get("significance_score", 0)), significance_cap)
    clarity = clamp(float(gpt_out.get("clarity_score", 0)), 0, 10)

    # blended citation score: 60% objective, 40% GPT
    citation_gpt = float(gpt_out.get("citation_score", 0))
    citation = clamp(round(0.6 * citation_obj + 0.4 * citation_gpt, 1), 0, 10)

    gpt_out["novelty_score_final"] = clamp(round(novelty, 1), 0, 10)
    gpt_out["significance_score_final"] = clamp(round(significance, 1), 0, 10)
    gpt_out["clarity_score_final"] = clamp(round(clarity, 1), 0, 10)
    gpt_out["citation_score_final"] = citation
    gpt_out["citation_score_objective"] = citation_obj
    gpt_out["caps"] = {"novelty_cap": novelty_cap, "significance_cap": significance_cap}
    return gpt_out


#############################################################
# 5) IN YOUR "Run Evaluation" BLOCK, ADD METRICS + USE FINAL SCORES
#############################################################
# After you create top10, do:
# cite_m = citation_metrics(refs, top10)
# gpt_out = gpt_review(title, gap, refs, top10_titles, top10_abstracts, style_choice, cite_m)
# gpt_out = combine_scores(gpt_out, cite_m, gap, top10)

# Then in metric cards use *_final:
# col1 ... gpt_out['novelty_score_final']
# col2 ... gpt_out['significance_score_final']
# col3 ... gpt_out['clarity_score_final']
# col4 ... gpt_out['citation_score_final']

# And optionally show an "Evidence" panel:
# st.json(gpt_out.get("evidence", {}))
# st.write("Objective citation score:", gpt_out.get("citation_score_objective"))
# st.write("Applied caps:", gpt_out.get("caps"))
