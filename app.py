# =====================================================
# AI JOB MARKET INTELLIGENCE & SMART RECOMMENDER
# =====================================================
#
# This file intentionally does NOT reimplement skill-cleaning, salary
# prediction, or recommendation scoring inline. All of that logic lives in
# src/utils.py, copied verbatim from Notebook 5 where it was built and
# verified against the real trained model. The original version of this app
# had its own, slightly different skill-cleaning function than the one used
# to prepare training data — a train/serve skew bug that silently degraded
# matching quality. Importing the same module the notebooks use is what
# prevents that from happening again.

import logging
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src import config
from src.utils import recommend_jobs, extract_skill_set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Job Market Intelligence",
    page_icon="💼",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Artifact loading — with real error handling, unlike the original version
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    if not config.ANALYTICS_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Job data not found at {config.ANALYTICS_DATA_FILE}. "
            "Run Notebook 3 (Feature Engineering) to generate it."
        )
    df = pd.read_csv(config.ANALYTICS_DATA_FILE)
    df["key_skills_cleaned"] = df["key_skills_cleaned"].fillna("")
    return df


@st.cache_resource
def load_models():
    missing = [
        p for p in [
            config.SALARY_MODEL_FILE,
            config.TFIDF_VECTORIZER_FILE,
            config.GROUP_MAPPINGS_FILE,
        ]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifact(s): {[str(p) for p in missing]}. "
            "Run Notebooks 3 and 4 to generate them."
        )
    salary_model = joblib.load(config.SALARY_MODEL_FILE)
    tfidf = joblib.load(config.TFIDF_VECTORIZER_FILE)
    group_mappings = joblib.load(config.GROUP_MAPPINGS_FILE)
    return salary_model, tfidf, group_mappings


try:
    analytics_df = load_data()
    salary_model, tfidf, group_mappings = load_models()
    job_tfidf_matrix = tfidf.transform(analytics_df["key_skills_cleaned"])
    startup_error = None
except FileNotFoundError as e:
    startup_error = str(e)
    logger.error("Startup failed: %s", startup_error)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("💼 AI-Powered Job Market Intelligence & Smart Recommender")

if startup_error:
    st.error(
        "This app couldn't load its data or model files, so it can't run right now.\n\n"
        f"Details: {startup_error}"
    )
    st.stop()

st.caption(
    "Salary predictions are trained on postings that disclosed a salary in this dataset "
    "(roughly a quarter of all postings) — not a random sample of the full job market. "
    "See the project README for the full limitations."
)

st.sidebar.header("👤 Enter Your Profile")

user_skills = st.sidebar.text_area(
    "Enter your skills:",
    placeholder="python sql machine learning pandas",
    help="Comma or space separated. Technical tokens like C++, C#, and .NET are supported.",
)

user_exp = st.sidebar.slider(
    "Years of Experience:",
    min_value=0,
    max_value=20,
    value=2,
)

city_options = sorted(set(group_mappings["top_cities"]) | {"Other"})
user_city = st.sidebar.selectbox("Preferred City", options=city_options)

submitted = st.sidebar.button("Get Recommendations")

if submitted:
    if not user_skills.strip():
        st.warning("Please enter your skills to get recommendations.")
    elif len(user_skills.strip()) < 3:
        st.warning("Please enter a more complete list of skills — that looks too short to match against.")
    else:
        try:
            predicted_salary, top_jobs, max_skill_score = recommend_jobs(
                user_skills,
                user_exp,
                user_city,
                analytics_df,
                job_tfidf_matrix,
                salary_model,
                tfidf,
                group_mappings,
                top_k=config.TOP_K_RECOMMENDATIONS,
            )
        except Exception as e:
            logger.exception("Recommendation failed")
            st.error(
                "Something went wrong generating recommendations. This has been logged. "
                "Please try again, or try different skills/experience/city values."
            )
            st.stop()

        st.subheader("💰 Predicted Suitable Salary")
        st.success(f"₹ {int(predicted_salary):,} per year")
        st.caption(
            "This prediction is conditioned on your experience, city, and skills. "
            "It does not account for a target industry or role, since those aren't "
            "collected above — see the README for why."
        )

        if max_skill_score < config.COLD_START_SKILL_SCORE_THRESHOLD:
            st.warning(
                "⚠️ Your listed skills don't closely match anything in this job market's "
                "data. The recommendations below are ranked mostly by salary/experience/city "
                "fit, **not** genuine skill matches — treat them as a weak starting point, "
                "not a confident shortlist. Try broader or more standard skill terms."
            )

        st.subheader("🎯 Top Job Recommendations")

        user_skill_set = extract_skill_set(user_skills)

        for _, row in top_jobs.iterrows():
            with st.container():
                st.markdown(f"### {row.job_title}")
                st.write(f"📍 City: {row.city_group}")
                st.write(f"🏭 Industry: {row.industry_group}")
                st.write(f"💼 Role: {row.role_group}")
                st.write(
                    f"💰 Salary: ₹ {int(row.salary_avg):,}" if pd.notna(row.salary_avg) else "💰 Salary: Not Disclosed"
                )
                st.write(f"🧠 Required Experience: {row.exp_avg} years")

                st.write(f"🔹 Skill Match: {round(row.skill_score * 100, 1)}%")
                st.write(f"🔹 Salary Match: {round(row.salary_score * 100, 1)}%")
                st.write(f"🔹 Experience Match: {round(row.exp_score * 100, 1)}%")

                job_skill_set = extract_skill_set(str(row.key_skills_cleaned))
                missing_skills = job_skill_set - user_skill_set

                if missing_skills:
                    st.write("📚 Skills to Learn:")
                    st.write(", ".join(sorted(list(missing_skills))[: config.MAX_MISSING_SKILLS_SHOWN]))
                else:
                    st.write("✅ Strong Skill Match!")

                st.markdown("---")

        st.caption(
            f"Ranking formula: {int(config.RECOMMENDATION_WEIGHTS['skill']*100)}% Skill + "
            f"{int(config.RECOMMENDATION_WEIGHTS['salary']*100)}% Salary + "
            f"{int(config.RECOMMENDATION_WEIGHTS['experience']*100)}% Experience + "
            f"{int(config.RECOMMENDATION_WEIGHTS['city']*100)}% City Preference "
            "(see Notebook 5 for the sensitivity check behind these weights)."
        )
