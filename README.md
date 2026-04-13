# Bank Customer Churn Prediction

Comparing five classification algorithms to predict customer churn in a bank, with exploratory data analysis, feature importance, and ROC-AUC evaluation.

## Overview

This project builds and evaluates multiple machine learning models on the [Bank Customer Churn](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction) dataset. The goal is to identify customers likely to leave the bank based on demographic and account features.

- Dataset: 10,000 customers, 11 features after preprocessing
- Target: `Exited` (1 = churned, 0 = retained)
- Class distribution: ~20% churned (imbalanced)

## Project Structure

```
├── notebooks/
│   └── identify_customers_churn.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| K-Nearest Neighbors | 0.8235 | 0.5935 | 0.3232 | 0.4185 | 0.7359 |
| Decision Tree | 0.7870 | 0.4617 | 0.5064 | 0.4830 | 0.6810 |
| **Random Forest** | **0.8655** | **0.7627** | **0.4580** | **0.5723** | **0.8575** |
| SVM | 0.8510 | 0.8467 | 0.2952 | 0.4377 | 0.8096 |
| Naive Bayes | 0.8335 | 0.6351 | 0.3588 | 0.4585 | 0.8044 |

Random Forest performs best across all metrics. Low Recall across models is expected due to class imbalance — ~80% of customers did not churn, causing models to be biased toward the majority class.

## Key Findings

- **Age** is the strongest predictor of churn; customers aged 40-60 are at highest risk
- **Inactive members** are approximately 2x more likely to churn than active ones
- **Germany** has the highest churn rate (~32%) compared to France and Spain
- Customers with **3-4 products** show a counterintuitively high churn rate

## Setup

```bash
pip install -r requirements.txt
```

```bash
jupyter notebook notebooks/identify_customers_churn.ipynb
```

The dataset is downloaded automatically via `kagglehub` on first run.

## Tech Stack

Python, pandas, numpy, scikit-learn, seaborn, matplotlib, kagglehub
