"""
Centralized configuration for the AI Job Market Intelligence app.

This exists as a direct fix to a real gap in the original project: every
weight, threshold, and file path was a magic number scattered inline through
app.py, which meant tuning the recommendation logic required hunting through
UI code, and there was no single place to see "what does this app actually
assume?" All of that lives here now.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_PATH = Path(__file__).resolve().parent.parent  # project root, one level up from src/
DATA_PATH = BASE_PATH / "dataset"
MODELS_PATH = BASE_PATH / "models"

ANALYTICS_DATA_FILE = DATA_PATH / "analytics_jobs.csv"
SALARY_MODEL_FILE = MODELS_PATH / "salary_model.pkl"

# NOTE: tfidf_vectorizer.pkl and group_mappings.pkl live in dataset/, not
# models/ — this is where Notebook 3 actually saved them. The original
# project's intended folder layout implied a models/ copy too; rather than
# maintaining two copies of the same artifact that could silently drift out
# of sync, there is exactly one copy, and every consumer points at it here.
TFIDF_VECTORIZER_FILE = DATA_PATH / "tfidf_vectorizer.pkl"
GROUP_MAPPINGS_FILE = DATA_PATH / "group_mappings.pkl"
MEDIANS_FILE = MODELS_PATH / "medians.pkl"

# ---------------------------------------------------------------------------
# Recommendation engine — hybrid scoring weights
# ---------------------------------------------------------------------------
# Sensitivity-checked in Notebook 5, Section 6: alternative weightings
# (skill-dominant, fully balanced) retained 70-100% overlap in top-10 results
# with this default, which is the evidence these weights are a reasonable
# choice, not proof they are optimal. True optimization would require real
# user feedback data (clicks/applications), which this project does not have
# — documented as future work, not faked here.
RECOMMENDATION_WEIGHTS = {
    "skill": 0.50,
    "salary": 0.20,
    "experience": 0.15,
    "city": 0.15,
}

# Below this max skill-similarity score across the whole candidate pool, a
# query is treated as a cold-start case (see Notebook 5, Section 8) — the
# skill signal is essentially absent, and the UI must say so rather than
# silently presenting a confident-looking ranked list.
COLD_START_SKILL_SCORE_THRESHOLD = 0.05

# Undisclosed-salary jobs get this neutral score (Notebook 5, Section 5) —
# not a 0 or a 1, since an undisclosed salary is neither confirmed to match
# nor confirmed to mismatch the user's predicted salary.
UNDISCLOSED_SALARY_NEUTRAL_SCORE = 0.5

TOP_K_RECOMMENDATIONS = 10
MAX_MISSING_SKILLS_SHOWN = 8

# ---------------------------------------------------------------------------
# Experience buckets — must match Notebook 3's definition exactly, since the
# salary model was trained on these exact bucket boundaries.
# ---------------------------------------------------------------------------
EXPERIENCE_BUCKETS = [
    (2, "Entry"),
    (5, "Mid"),
    (10, "Senior"),
]
EXPERIENCE_BUCKET_DEFAULT = "Leadership"  # anything above the last threshold
