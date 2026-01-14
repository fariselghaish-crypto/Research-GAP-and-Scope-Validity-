#############################################################
# RELIABLE METRICS + STRICT GPT REVIEW (SAFE / NO TYPE HINTS)
#############################################################

import difflib
from datetime import datetime

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
    return [y for y in years if 1950 <= y <= CURRENT_YEAR]


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
    for t in top10_df["Title"].fillna("").tolist():
        ok, _ = fuzzy_match(user_titles_norm, normalize_title(t))
        if ok:
            overlap_title += 1

    return {
        "n_refs": len([x for x in (user_refs_text or "").splitlines() if x.strip()]),
        "n_dois": len(user_dois),
        "recent_3y": recent_3y,
        "recent_5y": recent_5y,
        "top10_doi_overlap": overlap_doi,
        "top10_title_overlap": overlap_title,
    }


def citation_score_from_metrics(m):
    score = 2.0

    if m["n_refs"] >= 8:
        score += 1
    if m["n_refs"] >= 15:
        score += 0.5

    if m["n_dois"] >= 3:
        score += 1
    if m["n_dois"] >= 8:
        score += 0.5

    if m["recent_3y"] >= 3:
        score += 1.5
    if m["recent_3y"] >= 6:
        score += 0.5

    if m["top10_doi_overlap"] >= 1:
        score += 1.5
    if m["top10_title_overlap"] >= 2:
        score += 0.5

    return min(round(score, 1), 9)


def novelty_significance_caps(gap_text):
    gap = (gap_text or "").lower()

    generic_terms = [
        "framework", "interoperability", "lifecycle",
        "sustainability", "digital twin", "decision-making",
        "circular", "predictive", "explainable"
    ]

    generic_hits = sum(1 for g in generic_terms if g in gap)

    has_mechanism = any(
        k in gap for k in
        ["operational", "workflow", "governance", "validation", "traceable", "auditable"]
    )

    has_evaluation = any(
        k in gap for k in
        ["case study", "experiment", "benchmark", "dataset", "evaluation"]
    )

    novelty_cap = 10
    significance_cap = 10

    if generic_hits >= 6 and not has_mechanism:
        novelty_cap = 6

    if not has_evaluation:
        significance_cap = 6 if not has_mechanism else 7

    return novelty_cap, significance_cap


#############################################################
# DROP-IN REPLACEMENT GPT REVIEW (STRICT + EVIDENCE BASED)
#############################################################

def gpt_review(title, gap, refs, top10_titles, top10_abstracts, style_choice, cite_metrics):

    combined_abstracts = "\n\n".join(
        f"PAPER {i+1}\nTITLE: {t}\nABSTRACT:\n{a}"
        for i, (t, a) in enumerate(zip(top10_titles, top10_abstracts))
    )

    prompt = f"""
You are a senior reviewer for {style_choice}.
Evaluate the research gap ONLY. Do not rewrite.

STRICT RULES:
- Scores 8–10 are extremely rare.
- Ground novelty and citation judgements in the TOP-10 papers.
- Cite paper numbers when claiming overlap or support.
- If evidence is weak, reduce the score.

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
  "citation_comment": ""
}}

REFERENCE METRICS:
{cite_metrics}

STUDENT TITLE:
{title}

RESEARCH GAP:
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
            "citation_comment": ""
        }
