# AI Job Market Intelligence & Smart Recommender

A hybrid salary-prediction and job-recommendation system built end-to-end on a real, messy, public job-postings dataset — from raw data to a deployed Streamlit app.

This README documents what was built, how, why each major decision was made, what the honest results are, and — explicitly — what this project's limitations are. Nothing below is a claim the code and notebooks in this repo can't back up with an actual number.

## Table of Contents
- [Problem](#problem)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Methodology & Results, Notebook by Notebook](#methodology--results-notebook-by-notebook)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)
- [Documentation](#documentation)

## Problem

A job seeker looking at public job-board postings can't easily answer two questions:
1. What salary should I expect, given my skills and experience?
2. Which currently posted jobs are actually the best fit for me — not just the most recent or the most keyword-relevant?

This project builds a pipeline that answers both, and is transparent about the real, sometimes unflattering, limits of the answer.

**Full problem framing, scope, and formal requirements are in [`docs/01_Project_Charter_BRD.md`](docs/01_Project_Charter_BRD.md).**

## Architecture

```
Raw Naukri scrape (30,000 postings, 2019)
        │
        ▼
Notebook 1 — Data Understanding & Cleaning
  Parse salary/experience text → structured numeric fields
  Explicit sanity-check assertions (caught 2 real data-corruption bugs)
        │
        ▼
Notebook 2 — Exploratory Data Analysis
  Statistical significance testing (Pearson, ANOVA), not just charts
        │
        ▼
Notebook 3 — Feature Engineering
  Train/test split FIRST, then every leakage-prone statistic derived
  from train only (outlier bounds, top-N categories, TF-IDF vocabulary)
        │
        ▼
Notebook 4 — Salary Prediction Model
  Baseline vs. Linear Regression vs. Random Forest vs. XGBoost
  Model card saved as a permanent artifact, not just notebook output
        │
        ▼
Notebook 5 — Job Recommendation Engine
  Hybrid scoring (skill + salary + experience + city)
  Evaluated with a real, quantified proxy metric (leave-one-out)
        │
        ▼
src/utils.py + src/config.py — canonical, single-source-of-truth logic
        │
        ▼
app.py — Streamlit deployment
```

## Project Structure

```
AI_Job_Market_Intelligence/
│
├── dataset/
│   ├── naukri_dataset.csv        # raw source data
│   ├── naukri_cleaned.csv        # output of Notebook 1
│   ├── analytics_jobs.csv        # output of Notebook 3 — full job pool for recommendations
│   ├── X_train.pkl / X_test.pkl / y_train.pkl / y_test.pkl
│   ├── tfidf_vectorizer.pkl      # fit on train only (Notebook 3)
│   └── group_mappings.pkl        # top cities/industries, derived from train only (Notebook 3)
│
├── models/
│   ├── salary_model.pkl          # final XGBoost model (Notebook 4)
│   ├── scaler.pkl                # Linear Regression baseline only — NOT used by the deployed app
│   ├── medians.pkl               # defensive inference-time imputation (Notebook 4)
│   └── model_card.json           # permanent record of final metrics/hyperparameters
│
├── notebooks/
│   ├── 1_data_understanding.ipynb
│   ├── 2_exploratory_data_analysis.ipynb
│   ├── 3_feature_engineering.ipynb
│   ├── 4_salary_prediction_model.ipynb
│   └── 5_job_recommendation_engine.ipynb
│
├── src/
│   ├── config.py                 # centralized weights, thresholds, paths
│   └── utils.py                  # canonical skill-cleaning, salary prediction, recommendation logic
│
├── docs/
│   ├── 01_Project_Charter_BRD.md
│   └── 02_Data_Dictionary.md
│
├── app.py                        # Streamlit app — imports src/, does not reimplement logic
├── requirements.txt              # pinned versions
├── .gitignore
└── README.md
```

## Setup

```bash
git clone <this-repo>
cd AI_Job_Market_Intelligence
pip install -r requirements.txt
```

**Data and model artifacts are not committed to this repo** (see `.gitignore` — they're large, and binary diffs are useless in git history). To regenerate them:

1. Download the raw dataset (Naukri.com job postings, 2019 — [Kaggle link to be added]) into `dataset/naukri_dataset.csv`.
2. Run the notebooks in order: `1_data_understanding.ipynb` → `2_exploratory_data_analysis.ipynb` → `3_feature_engineering.ipynb` → `4_salary_prediction_model.ipynb` → `5_job_recommendation_engine.ipynb`.
3. `streamlit run app.py`

## Methodology & Results, Notebook by Notebook

All numbers below are real output from actually executing each notebook against the full dataset — not illustrative placeholders.

### Notebook 1 — Data Understanding & Cleaning
- 30,000 raw postings → structured, parsed salary and experience fields.
- Sanity-check assertions caught two real data-corruption issues: stray concatenated digits producing invalid ranges, and phone numbers/free text being parsed as salaries in the billions of rupees. Both fixed at the parser level, not patched after the fact.
- Only ~26.8% of postings have a parseable, plausible salary.

### Notebook 2 — Exploratory Data Analysis
- Experience–salary correlation: **r = 0.717, p < 0.001** (Pearson) — a real, statistically significant relationship, not an assumption.
- Salary differs significantly by industry: **one-way ANOVA, F = 11.61, p = 6.33e-86**.
- Identified and quantified location fragmentation (NCR/Mumbai satellite cities), role imbalance, and industry-field text contamination — all addressed in Notebook 3.

### Notebook 3 — Feature Engineering
- **The core leakage fix:** train/test split happens before any statistic (outlier bounds, top-N cities/industries, TF-IDF vocabulary) is computed. Every such value is derived from train only.
- Removed the original `exp_city_interaction`/`exp_industry_interaction` features — they multiplied experience by an arbitrary category code with no real interpretation. Documented reasoning: tree-based models learn interactions natively.
- Model-usable population: 7,734 rows (25.8% of the full dataset) — train 6,062 / test 1,515 after outlier and contamination filtering.

### Notebook 4 — Salary Prediction Model

| Model | R² | RMSE (₹/yr) | MAE (₹/yr) |
|---|---|---|---|
| Baseline (median predictor) | -0.04 | 558,090 | 339,016 |
| Baseline (mean predictor) | -0.00 | 540,242 | 341,772 |
| Linear Regression | 0.67 | 325,258 | 188,038 |
| Random Forest | 0.66 | 329,540 | 191,613 |
| **XGBoost (final model)** | **0.68** | 332,414 | 189,537 |

Every real model clears the naive baseline by a wide margin (R² jumps from ~0 to ~0.66–0.68) — that gap is the evidence the features carry real signal.

### Notebook 5 — Job Recommendation Engine
- Hybrid score: 50% skill similarity (TF-IDF cosine) + 20% salary fit + 15% experience fit + 15% city fit.
- **Evaluated with a leave-one-out proxy metric** (not present in earlier versions of this project): recommendations match the source posting's role category 56.6% of the time vs. a 26.7% pool base rate, and industry category 43.6% of the time vs. a 20.5% base rate — roughly **2x lift over random** on both.
- Cold-start queries (skills with no real overlap in the job market) are detected and surfaced as an explicit warning, not silently ranked as if meaningful.

## Key Engineering Decisions

- **One canonical logic module (`src/utils.py`), not duplicated copies.** An earlier version of this app had its own, slightly different skill-cleaning function than the one used to train the model — a train/serve skew bug that silently degraded matching quality for technical skills like `C++`/`C#`/`.NET`. Every notebook and the app now share identical, tested logic.
- **Outlier handling is a training-only decision.** The test set is deliberately left at its native, unfiltered distribution — a harder but more honest evaluation than filtering both splits identically.
- **Undisclosed salary gets a neutral score, not a penalty.** A job that didn't disclose its salary isn't assumed to be a bad match.

## Known Limitations

Stated here plainly, not discovered by reading the code:

- **Selection bias in the salary model.** It's trained only on the ~26% of postings that disclosed a plausible salary — a non-random subset. Predictions describe that subset, not the whole market.
- **2019 source data.** Absolute salary figures are not current market rates. The value of this project is the methodology and relative patterns, not live salary numbers.
- **User industry/role are unknown at prediction time.** The app only collects skills, experience, and city — the salary model's industry/role one-hot features default to the reference category for every user query.
- **Recommendation weights are sensitivity-checked, not learned.** No real user feedback (clicks/applications) exists to formally optimize against.
- **Role grouping is rule-based keyword matching** — roughly 45% of postings fall into a catch-all "Other" bucket, a real simplification cost.

## Future Work

- Learn recommendation weights from real user interaction data, if this were ever deployed with actual users.
- Replace rule-based role/industry grouping with a learned classifier.
- Add a leakage-safe target-encoded interaction feature (city × experience) as an alternative to the removed arbitrary interaction terms, and compare against the current tree-native approach.
- Re-scrape or source a current job-postings dataset to replace the 2019 snapshot.

## Documentation

- [`docs/01_Project_Charter_BRD.md`](docs/01_Project_Charter_BRD.md) — problem statement, scope, objectives, personas, assumptions, constraints.
- [`docs/02_Data_Dictionary.md`](docs/02_Data_Dictionary.md) — every column, raw through engineered, with type and definition.
- [`models/model_card.json`](models/model_card.json) — permanent record of the final model's hyperparameters and evaluation metrics.

## Author

Akhil T V
Data Scientist | ML 