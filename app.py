# =====================================================
# AI JOB MARKET INTELLIGENCE & SMART RECOMMENDER
# Fully Production-Ready Version
# =====================================================

# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ===============================
# 2. DEFINE PROJECT PATHS
# ===============================
BASE_PATH = Path(__file__).resolve().parent
DATA_PATH = BASE_PATH / "dataset"
MODELS_PATH = BASE_PATH / "models"

# ===============================
# 3. LOAD DATA & MODELS
# ===============================


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH / "analytics_jobs.csv")


@st.cache_resource
def load_models():
    salary_model = joblib.load(MODELS_PATH / "salary_model.pkl")
    tfidf = joblib.load(MODELS_PATH / "tfidf_vectorizer.pkl")
    medians = joblib.load(MODELS_PATH / "medians.pkl")
    return salary_model, tfidf, medians


analytics_df = load_data()
salary_model, tfidf, medians = load_models()

job_tfidf_matrix = tfidf.transform(
    analytics_df["key_skills_cleaned"].fillna("")
)

scaler = MinMaxScaler()

# ===============================
# 4. UTILITY FUNCTIONS
# ===============================


def clean_skills(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return " ".join(tokens)


def extract_skill_set(text):
    text = clean_skills(text)
    return set(text.split())


def create_exp_bucket(exp):
    if exp <= 2:
        return "Entry"
    elif exp <= 5:
        return "Mid"
    elif exp <= 10:
        return "Senior"
    else:
        return "Leadership"

# ===============================
# 5. CORE RECOMMENDATION ENGINE
# ===============================


def recommend_jobs(user_skills, user_exp, user_city):

    user_skills_cleaned = clean_skills(user_skills)
    feature_names = salary_model.feature_names_in_

    # Create feature-aligned dataframe
    user_df = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    # Numeric Features
    if "exp_avg" in user_df.columns:
        user_df.loc[0, "exp_avg"] = user_exp

    if "exp_squared" in user_df.columns:
        user_df.loc[0, "exp_squared"] = user_exp ** 2

    # Experience bucket
    exp_bucket = create_exp_bucket(user_exp)
    bucket_col = f"exp_bucket_{exp_bucket}"
    if bucket_col in user_df.columns:
        user_df.loc[0, bucket_col] = 1

    # TF-IDF mapping
    user_tfidf = tfidf.transform([user_skills_cleaned])
    tfidf_features = tfidf.get_feature_names_out()

    for i, word in enumerate(tfidf_features):
        if word in user_df.columns:
            user_df.loc[0, word] = user_tfidf[0, i]

    # Salary Prediction
    pred_log_salary = salary_model.predict(user_df)[0]
    predicted_salary = np.expm1(pred_log_salary)

    # ===============================
    # SCORING SYSTEM
    # ===============================
    df = analytics_df.copy()

    # Skill similarity
    skill_similarity = cosine_similarity(
        user_tfidf,
        job_tfidf_matrix
    ).flatten()

    df["skill_score"] = skill_similarity

    # Salary score (improved smooth logic)
    def salary_score(job_salary):
        if pd.isna(job_salary):
            return 0.5
        diff_ratio = abs(job_salary - predicted_salary) / predicted_salary
        return np.exp(-2 * diff_ratio)

    df["salary_score"] = df["salary_avg"].apply(salary_score)

    # Experience score
    max_exp = df["exp_avg"].max()
    df["exp_score"] = np.clip(
        1 - abs(df["exp_avg"] - user_exp) / max_exp,
        0,
        1
    )

    # City score
    df["city_score"] = (
        df["city_group"] == user_city
    ).astype(int)

    # Normalize skill similarity
    df[["skill_score"]] = scaler.fit_transform(
        df[["skill_score"]]
    )

    # Final Hybrid Score
    df["final_score"] = (
        0.50 * df["skill_score"] +
        0.20 * df["salary_score"] +
        0.15 * df["exp_score"] +
        0.15 * df["city_score"]
    )

    top_jobs = df.sort_values(
        by="final_score",
        ascending=False
    ).head(10)

    return predicted_salary, top_jobs


# ===============================
# 6. STREAMLIT UI
# ===============================
st.set_page_config(
    page_title="AI Job Market Intelligence",
    page_icon="💼",
    layout="wide"
)

st.title("💼 AI-Powered Job Market Intelligence & Smart Recommender")

st.sidebar.header("👤 Enter Your Profile")

user_skills = st.sidebar.text_area(
    "Enter your skills:",
    placeholder="python sql machine learning pandas"
)

user_exp = st.sidebar.slider(
    "Years of Experience:",
    min_value=0,
    max_value=20,
    value=2
)

user_city = st.sidebar.selectbox(
    "Preferred City",
    options=sorted(analytics_df["city_group"].unique())
)

if st.sidebar.button("Get Recommendations"):

    if user_skills.strip() == "":
        st.warning("Please enter your skills.")
    else:
        predicted_salary, top_jobs = recommend_jobs(
            user_skills,
            user_exp,
            user_city
        )

        st.subheader("💰 Predicted Suitable Salary")
        st.success(f"₹ {int(predicted_salary):,} per year")

        st.subheader("🎯 Top 10 Job Recommendations")

        user_skill_set = extract_skill_set(user_skills)

        for _, row in top_jobs.iterrows():

            with st.container():

                st.markdown(f"### {row.job_title}")
                st.write(f"📍 City: {row.city_group}")
                st.write(f"🏭 Industry: {row.industry_group}")
                st.write(f"💼 Role: {row.role_group}")
                st.write(
                    f"💰 Salary: ₹ {int(row.salary_avg) if not pd.isna(row.salary_avg) else 'Not Disclosed'}"
                )
                st.write(f"🧠 Required Experience: {row.exp_avg} years")

                # Explainable AI
                st.write(f"🔹 Skill Match: {round(row.skill_score * 100, 1)}%")
                st.write(
                    f"🔹 Salary Match: {round(row.salary_score * 100, 1)}%")
                st.write(
                    f"🔹 Experience Match: {round(row.exp_score * 100, 1)}%")

                # Skill Gap Analysis (Fixed)
                job_skill_set = extract_skill_set(
                    str(row.key_skills_cleaned)
                )

                missing_skills = job_skill_set - user_skill_set

                if missing_skills:
                    st.write("📚 Skills to Learn:")
                    st.write(", ".join(list(missing_skills)[:8]))
                else:
                    st.write("✅ Strong Skill Match!")

                st.markdown("---")

        st.caption(
            "Ranking Formula: 50% Skill + 20% Salary + "
            "15% Experience + 15% City Preference"
        )
