# Project Charter & Business Requirement Document (BRD)
## AI Job Market Intelligence & Smart Recommender

**Document Owner:** [Your Name]
**Status:** Draft v1.0
**Last Updated:** [Date]

---

## 1. Background

Job seekers on the Indian job market face two disconnected problems: they don't know what salary to expect for their skill and experience level, and they have no data-driven way to find roles that actually match their profile beyond keyword search on a job board. Recruitment platforms surface listings, but rarely tell a candidate *where they stand* relative to the market.

This project simulates a data team engagement: given a raw scrape of job postings, build an analytics pipeline and a decision-support tool that (a) predicts a fair market salary for a candidate's profile, and (b) recommends the jobs best matched to that profile.

## 2. Problem Statement

A job seeker cannot easily answer two questions from public job-board data:
1. "What salary should I expect, given my skills and experience?"
2. "Which currently posted jobs are the best fit for me, not just the most recent or most keyword-relevant?"

## 3. Objectives

| # | Objective | Success Criterion |
|---|-----------|-------------------|
| 1 | Clean and structure a raw, noisy job-postings dataset into an analysis-ready form | Documented, reproducible cleaning pipeline with <5% unexplained data loss at each stage |
| 2 | Identify the structural drivers of salary in the Indian job market (experience, skills, location, industry, role) | EDA delivers at least 5 statistically supported, actionable insights |
| 3 | Build a salary prediction model | Model beats a naive baseline (median/mean predictor) by a documented, meaningful margin, with train/test performance reported honestly (including known bias from disclosed-salary-only training data) |
| 4 | Build a hybrid job recommendation engine | Recommendations are evaluated against a defined quality metric, not just "it runs" |
| 5 | Ship a usable interface | A non-technical user can input skills/experience/location and receive a predicted salary + ranked job list with explainable match scores |

## 4. Scope

**In scope:**
- Data cleaning and structuring of the raw Naukri job-postings dataset
- Exploratory data analysis of salary, experience, location, industry, and skill patterns
- Feature engineering for a supervised regression model
- Salary prediction modeling (baseline + tuned models, compared)
- Content-based / hybrid job recommendation engine
- A deployed Streamlit interface

**Out of scope (explicitly, and stated as such rather than silently omitted):**
- Real-time job data ingestion (dataset is a static 2019 scrape — treated as a point-in-time market snapshot, not current market data)
- Multi-country / non-Indian job markets
- User accounts, saved searches, or persistent user profiles
- A/B testing or live user feedback loop on recommendation quality (acknowledged as future work)

## 5. Data Source

- **Source:** Naukri.com job postings dataset (public, Kaggle-hosted scrape)
- **Collection period:** 2019
- **Size:** ~30,000 raw job postings
- **License/attribution:** [link to be added]
- **Key limitation to disclose upfront:** This is historical (2019) data. Absolute salary figures are not representative of current market rates. The project's value is in the *methodology* (cleaning, modeling, recommending) and *relative* patterns (which factors drive salary), not in using this as a live salary calculator.

## 6. Target Users (Personas)

1. **Job seeker** — wants a fair salary expectation and a shortlist of well-matched roles.
2. **Recruiter/analyst** — wants market intelligence: which skills, cities, and industries are in demand.

## 7. Assumptions

- Salary figures, where disclosed, are annual CTC in INR as listed by the recruiter (not verified against actual offers).
- "Key Skills" field is recruiter-entered free text, not a standardized taxonomy — some noise is unavoidable and is documented rather than assumed away.
- Only ~27% of postings disclose salary; this project explicitly assumes and states that the salary model is trained on a *non-random subset* of the data (selection bias), and treats this as a documented limitation, not a hidden flaw.

## 8. Constraints

- Single-developer solo project — no dedicated QA, no stakeholder review cycle. Self-review checklist used instead (see `TESTING_AND_VALIDATION.md`).
- No access to ground-truth salary offers for validation — model quality is assessed against held-out disclosed-salary data only.
- Static dataset — no ability to re-scrape or refresh.

## 9. Deliverables

1. Cleaned, documented dataset + data dictionary
2. Five structured, documented notebooks (understanding → EDA → feature engineering → modeling → recommendation)
3. Trained salary prediction model with documented evaluation
4. Job recommendation engine with documented evaluation methodology
5. Deployed Streamlit application
6. Full project documentation (this document, data dictionary, README, limitations & future work)

## 10. Known Limitations (stated upfront, not discovered by the reader)

- Dataset is from 2019 — not current market data.
- Salary model is trained only on the subset of postings that disclosed salary (~13–16% after cleaning), which is not a random sample of all postings — this is a selection bias limitation, disclosed here rather than hidden.
- Recommendation engine weights are heuristically chosen, not learned or optimized against user feedback (no live user data exists for this project).
- Industry and role taxonomies are recruiter free-text, imperfectly normalized.

## 11. Success Definition for This Project (as a portfolio piece)

This project succeeds if a reviewer can read the documentation, understand exactly what was done, why, what its limitations are, and can verify every claim by re-running the pipeline — not if the model achieves a particular accuracy number in isolation.
