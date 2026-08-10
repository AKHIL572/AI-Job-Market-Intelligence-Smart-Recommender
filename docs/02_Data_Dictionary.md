# Data Dictionary
## AI Job Market Intelligence & Smart Recommender

This dictionary covers all three data stages in the pipeline: raw source data, cleaned data, and engineered/analytics data. Every column that appears anywhere in the project is documented here — this is the canonical reference; notebooks should not redefine column meaning inline.

---

## Stage 1 — Raw Data: `naukri_dataset.csv`

Source: Naukri.com job postings scrape, 2019, ~30,000 rows.

| Column | Type (raw) | Description | Known Data Quality Issues |
|---|---|---|---|
| `Uniq Id` | string | Unique identifier for the job posting | None observed |
| `Crawl Timestamp` | string (datetime) | Date/time the posting was scraped | Needs parsing to datetime |
| `Job Title` | string | Free-text job title as posted | High cardinality, no standardization |
| `Job Salary` | string | Free-text salary, e.g. "3,00,000 - 5,00,000 PA", or "Not Disclosed by Recruiter" | ~73% "Not Disclosed"; mixed formats (LPA, PA, ranges, single values) |
| `Job Experience Required` | string | Free-text experience range, e.g. "2 - 5 yrs" | Mixed formats; some unparseable |
| `Key Skills` | string | Pipe/comma/slash-delimited free text of required skills | Recruiter free text, not a taxonomy; mixes technical skills with job functions (e.g. "management," "bpo") |
| `Role Category` | string | Recruiter-assigned role category | Moderate cardinality |
| `Location` | string | Free-text location, sometimes multiple cities comma-separated | City name variants (e.g. "Bangalore" vs "Bengaluru") |
| `Functional Area` | string | Recruiter-assigned functional area | Not deeply used in this project (documented as a future-work opportunity) |
| `Industry` | string | Recruiter-assigned industry | **Contaminated**: some entries contain full company-description paragraphs instead of an industry label |
| `Role` | string | Recruiter-assigned specific role | High cardinality, heavily imbalanced (e.g. "Software Developer" >> most other roles) |

---

## Stage 2 — Cleaned Data: `naukri_cleaned.csv` (output of Notebook 1)

All raw columns are retained (lowercased, underscore-separated names) plus the following derived columns:

| Column | Type | Derived From | Description |
|---|---|---|---|
| `salary_disclosed` | int (0/1) | `job_salary` | 1 if a numeric salary was provided, 0 if "Not Disclosed" |
| `salary_min` / `salary_max` | float | `job_salary` | Parsed lower/upper salary bound, in INR/year. NaN if undisclosed or unparseable |
| `salary_avg` | float | `salary_min`, `salary_max` | Midpoint of min/max. Values below ₹10,000/year treated as parsing errors and set to NaN. Capped at the 99th percentile to remove extreme outliers |
| `exp_min` / `exp_max` | float | `job_experience_required` | Parsed lower/upper experience bound in years. Rows with only one number found are treated as unparseable (NaN) — see Notebook 1 limitations |
| `exp_avg` | float | `exp_min`, `exp_max` | Midpoint of min/max experience |
| `location_cleaned` | string | `location` | Primary city extracted from multi-city entries; "Bangalore"→"Bengaluru", "Delhi NCR"→"Delhi" mapping applied |
| `key_skills_cleaned` | string | `key_skills` | Lowercased, delimiter-normalized (`|`/`/`→`,`), deduplicated skill list. **Known limitation:** strips all characters except `[a-z0-9, ]`, which destroys skill tokens like `C++`, `C#`, `.NET` — flagged for correction |

---

## Stage 3 — Analytics / Model Data: `analytics_jobs.csv` and model feature set (output of Notebook 3)

| Column | Type | Derived From | Description |
|---|---|---|---|
| `log_salary` | float | `salary_avg` | `log1p(salary_avg)` — model target variable, log-transformed to reduce right-skew |
| `exp_squared` | float | `exp_avg` | `exp_avg ** 2` — non-linear experience term |
| `exp_bucket` | categorical | `exp_avg` | Entry (≤2y) / Mid (≤5y) / Senior (≤10y) / Leadership (>10y) |
| `city_cleaned` | string | `location_cleaned` | Further consolidated (Noida/Gurgaon/New Delhi/Delhi NCR → Delhi; Navi Mumbai → Mumbai) |
| `city_group` | categorical | `city_cleaned` | Top 7 cities kept individually; all others collapsed to "Other" |
| `industry` (cleaned) | string | `industry` (raw) | Rows with industry text ≥100 characters (contaminated company-description entries) removed |
| `industry_group` | categorical | `industry` (cleaned) | Top 10 industries kept individually; all others collapsed to "Other" |
| `role_group` | categorical | `role` | Rule-based grouping into Tech / Sales / HR / Finance / Operations / Other via keyword matching |
| `industry_role` | string | `industry_group`, `role_group` | Concatenated interaction category |
| `exp_city_interaction` | float | `exp_avg` × `city_group` category code | **Known modeling flaw — flagged for removal/redesign.** Multiplies experience by an arbitrary integer category code with no ordinal meaning; the resulting feature has no valid real-world interpretation |
| `exp_industry_interaction` | float | `exp_avg` × `industry_group` category code | Same flaw as above |
| `skills_cleaned` (train/test only) | string | `key_skills_cleaned` | Further regex-cleaned for TF-IDF input |
| TF-IDF features (`tfidf_0` … `tfidf_299`) | float (sparse) | `skills_cleaned` | Up to 300 TF-IDF terms fit on **training data only** (no leakage) |

---

## Model Artifacts (`/models`)

| File | Contents |
|---|---|
| `salary_model.pkl` | Tuned XGBoost regressor, trained on `log_salary` |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer (fit on train split only) |
| `scaler.pkl` | StandardScaler fitted on training features (used for the Linear Regression baseline only) |
| `medians.pkl` | Per-feature training-set medians, used to impute missing values consistently at inference time |

---

## Change Log

| Date | Change |
|---|---|
| [Date] | Initial data dictionary created |
