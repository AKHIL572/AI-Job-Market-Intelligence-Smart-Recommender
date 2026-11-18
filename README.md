# AI Job Market Intelligence & Smart Recommender

A hybrid salary-prediction and job-recommendation system built end-to-end on a real, messy, public job-postings dataset — from raw data to a deployed Streamlit app.

This README documents what was built, how, why each major decision was made, what the honest results are, and — explicitly — what this project's limitations are. Every metric and result cited in this documentation is backed by executable notebooks and evaluation scripts in this repository.

## Table of Contents
- [Problem](#problem)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Quickstart](#setup--quickstart)
- [Methodology & Results, Notebook by Notebook](#methodology--results-notebook-by-notebook)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)
- [Documentation](#documentation)
- [Author](#author)

---

## Problem

A job seeker looking at public job-board postings can't easily answer two critical questions:
1. **What salary should I expect**, given my specific skills, experience, and location?
2. **Which currently posted jobs are actually the best fit for me** — not just the most recent or the most keyword-stuffed?

This project builds an end-to-end machine learning pipeline and decision-support web application that answers both questions, with transparent disclosure of model assumptions, trade-offs, and empirical limits.

**Full problem framing, scope, and formal requirements are documented in [`docs/01_Project_Charter_BRD.md`](docs/01_Project_Charter_BRD.md).**

---

## Architecture

```
Raw Naukri scrape (~30,000 postings, 2019)
        │
        ▼
Notebook 1 — Data Understanding & Cleaning
  • Parse unstructured salary & experience text into numeric fields
  • Defensive assertions (caught 2 real data-corruption anomalies)
        │
        ▼
Notebook 2 — Exploratory Data Analysis
  • Statistical hypothesis testing (Pearson correlation, one-way ANOVA)
  • Analyzed location fragmentation, role distributions & industry contamination
        │
        ▼
Notebook 3 — Leakage-Free Feature Engineering
  • Train/test split performed FIRST
  • Outlier boundaries, top-N categories, and TF-IDF vocabulary fitted on train only
        │
        ▼
Notebook 4 — Salary Prediction Modeling
  • Naive baselines vs. Linear Regression vs. Random Forest vs. XGBoost
  • Model card saved as a reproducible artifact (`models/model_card.json`)
        │
        ▼
Notebook 5 — Hybrid Job Recommendation Engine
  • Multi-criteria scoring (50% skills + 20% salary + 15% experience + 15% city)
  • Cold-start detection & quantitative leave-one-out proxy evaluation
        │
        ▼
src/utils.py + src/config.py — Single-source-of-truth production logic
        │
        ▼
app.py — Interactive Streamlit Web Application
```

---

## Project Structure

```
AI-Job-Market-Intelligence-Smart-Recommender/
│
├── dataset/
│   ├── naukri_dataset.csv        # Raw source dataset (~30,000 postings)
│   ├── naukri_cleaned.csv        # Cleaned dataset (output of Notebook 1)
│   ├── analytics_jobs.csv        # Full job recommendation pool (output of Notebook 3)
│   ├── X_train.pkl / X_test.pkl  # Leakage-safe feature train/test splits
│   ├── y_train.pkl / y_test.pkl  # Target splits (log salary)
│   ├── tfidf_vectorizer.pkl      # TF-IDF vocabulary fit on train set only
│   └── group_mappings.pkl        # Top cities/industries/roles from train set
│
├── models/
│   ├── salary_model.pkl          # Final trained XGBoost regression model
│   ├── scaler.pkl                # Scaler artifact (used for Linear Regression baseline)
│   ├── medians.pkl               # Defensive inference-time imputation values
│   └── model_card.json           # Machine-readable record of metrics & parameters
│
├── notebooks/
│   ├── 1_data_understanding.ipynb
│   ├── 2_exploratory_data_analysis.ipynb
│   ├── 3_feature_engineering.ipynb
│   ├── 4_salary_prediction_model.ipynb
│   └── 5_job_recommendation_engine.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Centralized thresholds, paths, and recommendation weights
│   └── utils.py                  # Canonical skill-cleaning, salary prediction & matching logic
│
├── docs/
│   ├── 01_Project_Charter_BRD.md # Business requirements & project charter
│   └── 02_Data_Dictionary.md     # Column-level descriptions & transformations
│
├── app.py                        # Streamlit web application
├── requirements.txt              # Pinned Python dependencies
├── .gitignore                    # Git ignore file
└── README.md
```

---

## Setup & Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/AKHIL572/AI-Job-Market-Intelligence-Smart-Recommender.git
cd AI-Job-Market-Intelligence-Smart-Recommender
pip install -r requirements.txt
```

### 2. Launch the Streamlit App

All preprocessed data and trained model artifacts are included in the repository, so the web interface runs immediately without having to retrain:

```bash
streamlit run app.py
```

### 3. Pipeline Reproduction (Optional)

If you wish to re-execute the pipeline from raw data to final models, run the notebooks in order:

1. `notebooks/1_data_understanding.ipynb`
2. `notebooks/2_exploratory_data_analysis.ipynb`
3. `notebooks/3_feature_engineering.ipynb`
4. `notebooks/4_salary_prediction_model.ipynb`
5. `notebooks/5_job_recommendation_engine.ipynb`

---

## Methodology & Results, Notebook by Notebook

All numbers below represent real outputs from executing each notebook against the complete dataset.

### Notebook 1 — Data Understanding & Cleaning
- **30,000 raw postings** parsed from unstructured text into structured numeric fields (`salary_min`, `salary_max`, `salary_avg`, `exp_min`, `exp_max`, `exp_avg`).
- Programmatic assertion checks identified and resolved real data-corruption anomalies: concatenated numeric ranges and phone numbers parsed as salaries in billions of rupees.
- Only ~26.8% of raw postings contained a plausible, parseable salary range (reflecting real-world undisclosed compensation patterns).

### Notebook 2 — Exploratory Data Analysis
- **Experience vs. Salary:** Demonstrates strong, statistically significant correlation (**r = 0.717, p < 0.001** via Pearson correlation).
- **Industry Variance:** Significant compensation differences across industries confirmed via **one-way ANOVA (F = 11.61, p = 6.33e-86)**.
- Systematic isolation of NCR/Mumbai satellite city fragmentation and industry free-text noise.

### Notebook 3 — Feature Engineering
- **Leakage Prevention:** Train/test split (80/20) executed *prior* to computing summary statistics, outlier bounds, TF-IDF representations, or frequency groupings.
- Cleaned technical tokens preserve domain symbols (e.g., `C++`, `C#`, `.NET`).
- Final usable population for salary modeling: **7,577 rows** (train: 6,062 / test: 1,515) after outlier removal on the training split.

### Notebook 4 — Salary Prediction Model

Evaluated on the held-out test set:

| Model | Log R² | Log RMSE | Log MAE | Test RMSE (₹/yr) | Test MAE (₹/yr) |
|---|---|---|---|---|---|
| Baseline (Median Predictor) | -0.042 | 0.761 | 0.609 | ₹558,090 | ₹339,016 |
| Baseline (Mean Predictor) | -0.001 | 0.745 | 0.615 | ₹540,242 | ₹341,772 |
| Linear Regression | 0.671 | 0.427 | 0.321 | ₹325,499 | ₹188,213 |
| Random Forest | 0.661 | 0.434 | 0.329 | ₹329,433 | ₹191,689 |
| **XGBoost (Final Model)** | **0.686** | **0.417** | **0.315** | **₹323,975** | **₹186,912** |

Every trained model substantially outperforms the naive baselines, lifting R² from ~0 to ~0.67–0.69, proving strong learned signal from experience, location, role, and skill features.

### Notebook 5 — Job Recommendation Engine
- **Hybrid Scoring Formula:**
  $$\text{Match Score} = 0.50 \cdot S_{\text{skills}} + 0.20 \cdot S_{\text{salary}} + 0.15 \cdot S_{\text{experience}} + 0.15 \cdot S_{\text{city}}$$
- **Quantitative Proxy Evaluation:** Leave-one-out testing demonstrates top recommendations match the target posting's role category **56.6%** of the time (vs. 26.7% random base rate) and industry category **43.6%** of the time (vs. 20.5% base rate) — representing an approximate **2x lift over random**.
- **Cold-Start Safeguards:** Skill vectors falling below similarity thresholds trigger explicit low-confidence alerts rather than misleading top matches.
- **Skill Gap Discovery:** Compares candidate skills against job requirements to display actionable "Skills to Learn".

---

## Key Engineering Decisions

- **Single Canonical Module (`src/utils.py`):** UI and notebook code share the identical preprocessing and tokenization logic, eliminating train-serve skew.
- **Training-Only Outlier Filtering:** Outlier bounds are calculated strictly from training data, leaving the test set intact for unbiased evaluation.
- **Fair Scoring for Undisclosed Salaries:** Postings with undisclosed compensation receive a neutral score (0.50) rather than an arbitrary penalty.
- **Config Centralization (`src/config.py`):** Weights, cold-start thresholds, and paths are decoupled from application and modeling scripts.

---

## Known Limitations

- **Selection Bias:** The salary model is trained strictly on postings that disclosed salary (~26%), which is not an unbiased sample of the full market.
- **Temporal Baseline (2019):** Absolute compensation figures reflect historical 2019 rates. The system's value lies in its relative patterns and modular architecture.
- **Unobserved Inference Inputs:** Target industry and role are omitted in the end-user profile input and defaulted to baseline reference categories.
- **Static Weights:** Recommendation weights were validated via sensitivity analysis rather than live conversion/click feedback.

---

## Future Work

- Incorporate real-time compensation adjustments and current market datasets.
- Learn recommendation ranking weights through implicit user feedback (clicks, bookmarks, applications).
- Replace rule-based role/industry categorization with fine-tuned semantic embeddings (e.g., Sentence-Transformers).

---

## Documentation

- [`docs/01_Project_Charter_BRD.md`](docs/01_Project_Charter_BRD.md) — Business requirements, persona definitions, and project charter.
- [`docs/02_Data_Dictionary.md`](docs/02_Data_Dictionary.md) — Comprehensive attribute definitions and transformations.
- [`models/model_card.json`](models/model_card.json) — Structured metadata, evaluation metrics, and feature lists.

---

## Author

**Akhil T V**  
Data Scientist | ML Engineer  
GitHub: [@AKHIL572](https://github.com/AKHIL572)
