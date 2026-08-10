"""
Canonical, single-source-of-truth logic for skill cleaning, salary
prediction, and job recommendation.

This module exists to fix the root cause of a real bug found in the original
project: app.py maintained its own, slightly different copy of the
skill-cleaning function than the one used to prepare training data in
Notebook 3 — a classic train/serve skew bug that silently degraded matching
quality for exactly the technical skill tokens (C++, C#, .NET) most likely
to matter. Every function below is copied verbatim from Notebook 5, which
verified this exact logic against the real trained model and real data. If
this logic ever needs to change, it must change in Notebook 3, Notebook 5,
and here together — not independently.
"""

import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    EXPERIENCE_BUCKETS,
    EXPERIENCE_BUCKET_DEFAULT,
    UNDISCLOSED_SALARY_NEUTRAL_SCORE,
    RECOMMENDATION_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Skill text cleaning — must stay byte-for-byte identical to Notebook 3/5
# ---------------------------------------------------------------------------
def clean_skills_text(text: str) -> str:
    """Canonical skill-text cleaner. Preserves +, #, . so technical tokens
    like C++, C#, and .NET survive cleaning intact — the original app.py's
    version stripped these characters, causing train/serve skew."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+.# ]", " ", text)
    return text


def extract_skill_set(text: str) -> set:
    cleaned = clean_skills_text(text)
    return set(t for t in cleaned.split() if t)


def exp_bucket(exp: float) -> str:
    for threshold, label in EXPERIENCE_BUCKETS:
        if exp <= threshold:
            return label
    return EXPERIENCE_BUCKET_DEFAULT


# ---------------------------------------------------------------------------
# Salary prediction
# ---------------------------------------------------------------------------
def predict_salary(user_skills_cleaned, user_exp, user_city, model, tfidf, group_mappings):
    """Build a model-aligned feature vector for a user profile and predict
    salary. Fixes a real bug in the original app.py: the user's city was
    collected in the UI but never actually reached the salary model's
    feature vector — only used for the recommendation-ranking city score.
    Here, city is mapped through the same group_mappings saved in
    Notebook 3, so it's bucketed identically to how training data was.

    Known limitation (documented, not hidden): the user profile has no
    role_group or industry_group input, so those one-hot columns are left
    at zero (the reference/unknown category) — the prediction is
    conditioned on experience + city + skills only.
    """
    feature_names = model.feature_names_in_
    user_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    if "exp_avg" in user_df.columns:
        user_df.loc[0, "exp_avg"] = user_exp
    if "exp_squared" in user_df.columns:
        user_df.loc[0, "exp_squared"] = user_exp ** 2

    bucket_col = f"exp_bucket_{exp_bucket(user_exp)}"
    if bucket_col in user_df.columns:
        user_df.loc[0, bucket_col] = 1

    mapped_city = group_mappings["city_map"].get(user_city, user_city)
    city_group = mapped_city if mapped_city in group_mappings["top_cities"] else "Other"
    city_col = f"city_group_{city_group}"
    if city_col in user_df.columns:
        user_df.loc[0, city_col] = 1

    user_tfidf_vec = tfidf.transform([user_skills_cleaned])
    tfidf_terms = tfidf.get_feature_names_out()
    for i, term in enumerate(tfidf_terms):
        if term in user_df.columns:
            user_df.loc[0, term] = user_tfidf_vec[0, i]

    pred_log_salary = model.predict(user_df)[0]
    predicted_salary = float(np.expm1(pred_log_salary))
    return predicted_salary, user_tfidf_vec


# ---------------------------------------------------------------------------
# Hybrid recommendation scoring
# ---------------------------------------------------------------------------
def salary_match_score(job_salary, predicted_salary):
    """Undisclosed salary gets a neutral score — not penalized, not
    rewarded, since the data simply doesn't confirm a match either way."""
    if pd.isna(job_salary):
        return UNDISCLOSED_SALARY_NEUTRAL_SCORE
    diff_ratio = abs(job_salary - predicted_salary) / predicted_salary
    return float(np.exp(-2 * diff_ratio))


def recommend_jobs(
    user_skills,
    user_exp,
    user_city,
    analytics_df,
    job_tfidf_matrix,
    salary_model,
    tfidf,
    group_mappings,
    top_k=10,
    weights=None,
):
    """Return (predicted_salary, ranked_dataframe, max_skill_score).

    max_skill_score is returned explicitly so the caller can detect and
    surface a cold-start warning (Notebook 5, Section 8) rather than
    silently presenting a confident-looking ranked list for a query with no
    real skill signal in the corpus.
    """
    if weights is None:
        weights = RECOMMENDATION_WEIGHTS

    user_skills_cleaned = clean_skills_text(user_skills)
    predicted_salary, user_tfidf_vec = predict_salary(
        user_skills_cleaned, user_exp, user_city, salary_model, tfidf, group_mappings
    )

    scored = analytics_df.copy()
    scored["skill_score"] = cosine_similarity(user_tfidf_vec, job_tfidf_matrix).flatten()
    scored["salary_score"] = scored["salary_avg"].apply(lambda s: salary_match_score(s, predicted_salary))

    max_exp = analytics_df["exp_avg"].max()
    scored["exp_score"] = np.clip(1 - abs(scored["exp_avg"] - user_exp) / max_exp, 0, 1)

    mapped_city = group_mappings["city_map"].get(user_city, user_city)
    user_city_group = mapped_city if mapped_city in group_mappings["top_cities"] else "Other"
    scored["city_score"] = (scored["city_group"] == user_city_group).astype(int)

    scored["final_score"] = (
        weights["skill"] * scored["skill_score"]
        + weights["salary"] * scored["salary_score"]
        + weights["experience"] * scored["exp_score"]
        + weights["city"] * scored["city_score"]
    )

    max_skill_score = float(scored["skill_score"].max())
    top_jobs = scored.sort_values("final_score", ascending=False).head(top_k)

    return predicted_salary, top_jobs, max_skill_score
