# TOKI — AI Prompt Evaluation & Optimization

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://toki-prototype.streamlit.app/)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-TOKI-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/repository/docker/dropper135/toki/general)

## Overview

TOKI is a machine learning-powered platform that evaluates and optimizes prompts before they are sent to Large Language Models (LLMs). Instead of relying on expensive LLM-as-a-Judge systems, TOKI leverages XGBoost-based machine learning models to predict prompt quality, making evaluation faster, more transparent, explainable, and cost-efficient.

The platform analyzes prompt characteristics such as clarity, specificity, structure, constraints, and task type to generate a prompt quality score, grade prediction, SHAP-based explanations, and actionable improvement recommendations.

**Live Application:** https://toki-prototype.streamlit.app/

**Docker Hub:** https://hub.docker.com/repository/docker/dropper135/toki/general

---

## Why TOKI?

Traditional prompt evaluation often depends on another LLM acting as a judge, increasing inference cost and latency. TOKI replaces the evaluator LLM with a lightweight machine learning model that provides fast, consistent, and explainable prompt assessment while significantly reducing computational cost.

---

## Key Features

- Machine Learning-based prompt evaluation (No evaluator LLM)
- Prompt Quality Score Prediction
- Prompt Grade Prediction
- SHAP Explainable AI (XAI)
- Prompt optimization recommendations
- Working Backward (Outcome-first) prompt suggestions
- Interactive Streamlit dashboard
- Human feedback collection
- Fast, lightweight, and scalable architecture

---

## Tech Stack

- Python
- Streamlit
- XGBoost
- Scikit-learn
- SHAP
- Pandas
- NumPy
- Plotly

---

## Dataset

The models are trained on a balanced synthetic dataset containing **20,000 prompts** spanning multiple prompt engineering tasks. Each prompt is evaluated using prompt engineering principles including:

- Clarity
- Specificity
- Structure
- Constraints
- Task Type
- Overall Prompt Quality

---

## Project Structure

```text
TOKI/
│
├── .streamlit/
├── EDA/
│   ├── app1.py
│   ├── eda_utils.py
│   ├── EDA.ipynb
│   ├── Cleaned_dataset.csv
│   ├── prompt_evaluation_dataset.csv
│   ├── prompt_evaluation_training_data.csv
│   └── README.md
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
├── app.py
├── feedback_logger.py
├── feedback.csv
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
├── Model1.ipynb
├── Model2.ipynb
└── Cleaned_dataset.csv
```

---

# Running Locally

## Clone the Repository

```bash
git clone https://github.com/Shubhanshu-G/Toki.git
cd Toki
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Launch the Application

```bash
streamlit run app.py
```

The application will be available at:

```
http://localhost:8501
```

---

# Running with Docker

## Pull the Docker Image

```bash
docker pull dropper135/toki:latest
```

## Run the Container

```bash
docker run -p 8502:8502 dropper135/toki:latest
```

Open your browser and visit:

```
http://localhost:8502
```

---

# Build Docker Image Locally

Clone the repository and build the Docker image yourself.

```bash
git clone https://github.com/Shubhanshu-G/Toki.git
cd Toki

docker build -t toki .
```

Run the image:

```bash
docker run -p 8502:8502 toki
```

---

# Docker Support

This repository includes:

- Dockerfile for containerized deployment
- .dockerignore for optimized image builds
- Lightweight Python 3.12 Slim base image
- Streamlit server configured for Docker networking
- Optimized dependency installation using `pip --no-cache-dir`

---

## Repository Links

**GitHub Repository**

https://github.com/Shubhanshu-G/Toki

**Docker Hub Repository**

https://hub.docker.com/repository/docker/dropper135/toki/general

---

## License

This project is released for educational and research purposes.