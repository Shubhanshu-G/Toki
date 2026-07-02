# TOKI — AI Prompt Evaluation & Optimization

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://toki-prototype.streamlit.app/)

## Overview
TOKI is a machine learning-powered platform that evaluates and optimizes prompts before they are sent to Large Language Models (LLMs). Instead of using expensive LLM-as-a-Judge systems, TOKI leverages an XGBoost-based model to predict prompt quality, making evaluation faster, more transparent, and cost-efficient.

The platform analyzes prompt characteristics such as clarity, specificity, structure, constraints, and task type to generate a quality score, grade, SHAP-based explanations, and actionable improvement suggestions.

---

## Why TOKI?
Unlike traditional prompt evaluators that rely on another LLM, TOKI uses a lightweight machine learning model to assess prompt quality. This eliminates evaluator-model inference, reduces latency and operational costs, and provides transparent, explainable feedback to help users write better prompts.

---

## Key Features
- **ML-based prompt evaluation** (No evaluator LLM)
- **Prompt Quality Score & Grade Prediction**
- **SHAP Explainable AI (XAI)**
- **Prompt optimization recommendations**
- **Working Backward (Outcome-first)** prompt suggestions
- **Interactive Streamlit dashboard** (with human feedback collection support)
- **Fast, lightweight, and scalable architecture**

---

## Tech Stack
**Python** • **XGBoost** • **Scikit-learn** • **SHAP** • **Streamlit** • **Pandas** • **NumPy** • **Plotly**

---

## Dataset
Trained on a balanced synthetic dataset of 20,000 prompts across multiple domains, labeled using prompt engineering principles such as clarity, specificity, structure, constraints, and overall quality.

---

## Project Structure
```
TOKI/
│
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md
│
├── models/
│   ├── label_encoder.pkl
│   ├── min_max_scaler.pkl
│   ├── min_max_scaler_m2.pkl
│   ├── mlb.pkl
│   ├── xgb_cls_model.pkl
│   ├── xgb_reg_model.pkl
│   └── xgb_multi_output.pkl
│
├── EDA/
│   ├── app1.py
│   ├── eda_utils.py
│   ├── EDA.ipynb
│   ├── Cleaned_dataset.csv
│   ├── prompt_evaluation_dataset.csv
│   ├── prompt_evaluation_training_data.csv
│   └── README.md
│
├── Model1.ipynb
├── Model2.ipynb
└── Cleaned_dataset.csv
```

---

## Installation & Running

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/TOKI.git
   cd TOKI
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```
