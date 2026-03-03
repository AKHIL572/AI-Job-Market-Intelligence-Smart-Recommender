# 💼 AI-Powered Job Market Intelligence & Smart Job Recommendation System

An end-to-end Data Science project that transforms raw job posting data into actionable job market intelligence and delivers personalized job recommendations using Machine Learning & NLP.

---

## 🚀 Project Overview

This project simulates a real-world product used by job portals, HR tech companies, and workforce analytics firms.

It performs:

- 📊 Job Market Intelligence Analysis  
- 💰 Salary Prediction using ML  
- 🔥 In-demand Skill Identification  
- 📍 Hiring Hotspot Detection (City Trends)  
- 🎯 AI-Powered Hybrid Job Recommendation Engine  

The system combines:

- NLP (TF-IDF)
- Machine Learning (Salary Prediction Model)
- Hybrid Recommendation System (Content + Scoring Based)
- Streamlit Production App Deployment

---

## 🏗️ Business Problem

Recruiters and job seekers face challenges such as:

- Unstructured job postings
- Salary transparency issues
- Skill mismatch
- Lack of data-driven career guidance

This project solves that by:

1. Structuring raw job data
2. Predicting fair salary ranges
3. Identifying market-demanded skills
4. Recommending personalized jobs based on:
   - Skills
   - Experience
   - Salary alignment
   - Location preference

---

## 🧠 System Architecture

```
Raw Job Dataset
↓
Data Cleaning & Feature Engineering
↓
TF-IDF Skill Vectorization
↓
Salary Prediction Model (Log-Transformed Regression)
↓
Hybrid Recommendation Engine
↓
Streamlit Production App

```
---

## 📂 Project Structure

```
job_AI_system/
│
├── dataset/
│ ├── naukri_dataset.csv
│ ├── naukri_cleaned.csv
│ └── analytics_jobs.csv
│
├── models/
│ ├── salary_model.pkl
│ ├── tfidf_vectorizer.pkl
│ └── medians.pkl
│
├── notebooks/
│ ├── 1_data_understanding.ipynb
│ ├── 2_exploratory_data_analysis.ipynb
│ ├── 3_feature_engineering.ipynb
│ ├── 4_salary_prediction_model.ipynb
│ └── 5_job_recommendation_engine.ipynb
│
├── app.py
├── requirements.txt
└── README.md

```
---

## 📊 Key Features

### 1️⃣ Job Market Intelligence

- Top hiring cities
- Industry-wise demand
- Role distribution analysis
- Experience demand trends
- Salary distribution insights

---

### 2️⃣ Salary Prediction Model

- Log-transformed salary regression
- Feature engineered experience buckets
- TF-IDF skill-based salary influence
- Handles missing salary values
- Production-ready serialized model

**Output:**  
Predicted fair annual salary for user profile.

---

### 3️⃣ Hybrid Job Recommendation Engine

Uses a weighted scoring system:

| Component        | Weight |
|------------------|--------|
| Skill Similarity | 50%    |
| Salary Match     | 20%    |
| Experience Match | 15%    |
| City Match       | 15%    |

### Recommendation Logic:

- Cosine similarity on TF-IDF skill vectors
- Salary closeness scoring (exponential decay)
- Experience gap normalization
- Location preference matching
- Final hybrid ranking score

---

### 4️⃣ Explainable AI

For each recommended job:

- Skill Match %
- Salary Match %
- Experience Match %
- Skill Gap Analysis (missing skills to learn)

This makes the system transparent and recruiter-friendly.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-Learn**
- **TF-IDF (NLP)**
- **Cosine Similarity**
- **Streamlit**
- **Joblib**

---

## 📈 Machine Learning Details

### Salary Model

- Target: `log(salary_avg)`
- Features:
  - Experience
  - Experience Squared
  - Experience Bucket
  - TF-IDF Skill Features

Why log transformation?

- Reduces skewness
- Handles salary outliers
- Improves regression stability

---

## 🎯 How to Run Locally

### 1️⃣ Clone the repository


git clone https://github.com/yourusername/AI-Job-Market-Intelligence.git

cd AI-Job-Market-Intelligence


### 2️⃣ Install dependencies


pip install -r requirements.txt


### 3️⃣ Run Streamlit app


streamlit run app.py


---

## 💡 Example User Flow

1. User enters:
   - Skills: `python sql machine learning`
   - Experience: `3 years`
   - Preferred City: `Bangalore`

2. System:
   - Predicts suitable salary
   - Calculates skill similarity
   - Computes hybrid ranking
   - Displays Top 10 jobs
   - Suggests missing skills

---

## 📊 Potential Business Impact

This system can be used by:

- Job portals (like Naukri/Indeed)
- EdTech companies
- HR analytics platforms
- Workforce planning firms

### Value Delivered:

- Personalized job recommendations
- Data-driven salary insights
- Career path guidance
- Skill gap identification
- Market trend intelligence

---

## 🔥 What Makes This Project Interview-Ready?

✔ End-to-End Pipeline  
✔ NLP + ML Integration  
✔ Hybrid Recommendation System  
✔ Explainable AI  
✔ Production Deployment  
✔ Real Business Framing  
✔ Clean Project Structure  

This is not just a notebook project — it is a deployable product simulation.

---

## 📌 Future Improvements

- Deep Learning Embeddings (BERT-based skill matching)
- Collaborative Filtering
- Resume Parsing Integration
- Real-time API deployment (FastAPI)
- Dockerization
- Cloud deployment (AWS/GCP/Azure)

---

## 👨‍💻 Author

**Your Name**  
Data Scientist | Machine Learning Engineer 
