#############################################################
# UPDATED HARD VALIDITY RULES (200 words + reference ranges)
#############################################################

# --- Rule 1: Rewritten Gap Word Count ---
gap_word_count = len(gpt_out["rewritten_gap"].split())

if gap_word_count >= 200:
    length_flag = "valid"
elif 150 <= gap_word_count < 200:
    length_flag = "borderline"
else:
    length_flag = "invalid"

# --- Rule 2: Number of References ---
num_refs = len([r for r in refs.split("\n") if r.strip()])

if num_refs >= 7:
    ref_flag = "valid"
elif 5 <= num_refs <= 6:
    ref_flag = "borderline"
else:
    ref_flag = "invalid"

# --- Apply Penalties ---
length_penalty = 0
ref_penalty = 0

if length_flag == "borderline":
    length_penalty = 5
elif length_flag == "invalid":
    length_penalty = 15

if ref_flag == "borderline":
    ref_penalty = 5
elif ref_flag == "invalid":
    ref_penalty = 15

# --- Final Score Calculation ---
total_raw = (
    gpt_out["novelty_score"]
    + gpt_out["significance_score"]
    + gpt_out["clarity_score"]
    + gpt_out["citation_score"]
    - length_penalty
    - ref_penalty
)

total_score = max(0, min(40, total_raw))

# --- VERDICT LOGIC ---
if length_flag == "invalid" or ref_flag == "invalid":
    verdict = "❌ NOT VALID"
elif total_score >= 30:
    verdict = "🟢 VALID"
elif total_score >= 20:
    verdict = "🟡 BORDERLINE"
else:
    verdict = "❌ NOT VALID"
