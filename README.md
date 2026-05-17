# Home Credit Default Risk — End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.786-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A production-grade machine learning pipeline for predicting loan default risk, built on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) Kaggle dataset. The project covers the full ML lifecycle — data ingestion, preprocessing, feature engineering across 6 data sources, model training, SHAP-based explainability, and deployment as an interactive web application.

**Live Demo:** [Hugging Face Spaces](#) *(link after deployment)*

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Approach](#approach)
- [Results](#results)
- [SHAP Explainability](#shap-explainability)
- [Installation](#installation)
- [Usage](#usage)
- [Key Design Decisions](#key-design-decisions)
- [Future Improvements](#future-improvements)

---

## Problem Statement

Many people struggle to get loans due to insufficient or absent credit histories. Home Credit uses alternative data to predict whether an applicant is capable of repayment. The goal is to predict the probability that an applicant will default on a loan — a binary classification problem evaluated using ROC-AUC.

This is a real-world imbalanced classification problem: only ~8% of applicants default, making accuracy a misleading metric and requiring careful handling of class imbalance.

---

## Dataset

Data sourced from the [Home Credit Default Risk Kaggle Competition](https://www.kaggle.com/competitions/home-credit-default-risk/data).

| File | Rows | Description |
|---|---|---|
| `application_train.csv` | 307,511 | Main application data with target |
| `bureau.csv` | 1,716,428 | External credit history |
| `previous_application.csv` | 1,670,214 | Previous Home Credit applications |
| `installments_payments.csv` | 13,605,401 | Installment payment history |
| `POS_CASH_balance.csv` | 10,001,358 | POS and cash loan monthly balance |
| `credit_card_balance.csv` | 3,840,312 | Credit card monthly balance |

**Note:** Dataset files are not included in this repository due to size. Download them from Kaggle and place them in the project root.

---

## Project Structure

```
home-credit-default-risk/
├── configs/
│   └── config.yaml          ← all hyperparameters and paths
├── models/
│   ├── credit_risk_model_final.pkl
│   └── feature_columns_final.json
├── src/
│   ├── __init__.py
│   ├── data_loader.py       ← loads all 6 data sources
│   ├── preprocessor.py      ← cleaning, imputation, encoding
│   ├── feature_engineer.py  ← aggregations and derived features
│   ├── trainer.py           ← model training and saving
│   └── evaluator.py         ← metrics, SHAP, plots
├── app.py                   ← Gradio web application
├── main.py                  ← end-to-end pipeline entry point
├── requirements.txt
└── README.md
```

---

## Approach

### 1. Data Cleaning
- Dropped columns with more than 60% missing values (17 columns removed)
- Filled remaining numerical nulls with median, categorical nulls with mode
- Capped outliers at 99th percentile for `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`
- Detected and handled `DAYS_EMPLOYED` anomaly (365243 placeholder for unemployed)

### 2. Feature Engineering

**Main application features:**
- `CREDIT_INCOME_RATIO` — credit amount relative to income
- `ANNUITY_INCOME_RATIO` — monthly burden relative to income
- `CREDIT_TERM` — effective loan term in months
- `DAYS_EMPLOYED_RATIO` — employment duration relative to age
- `AGE_INCOME_RATIO` — age relative to income

**Bureau aggregations:**
- Total, active, and closed credit counts
- Total and maximum overdue days
- Debt-to-credit ratio

**Previous application aggregations:**
- Approval, refusal, cancellation counts
- Credit-to-application ratio
- Recency of last application

**Installment payment aggregations:**
- Late payment ratio (fraction of payments made late)
- Mean payment lateness in days
- Mean underpayment amount

**POS Cash aggregations:**
- Days past due statistics
- Overdue ratio across all months

**Credit card aggregations:**
- Credit utilization ratio (balance / limit)
- Days past due statistics
- Payment behavior features

### 3. Encoding
- Label encoding for high-cardinality categoricals (ORGANIZATION_TYPE: 58 unique)
- One-hot encoding for low-cardinality categoricals

### 4. Model Training
- Baseline: Logistic Regression (ROC-AUC: 0.748)
- Final model: XGBoost with `scale_pos_weight=11.38` to handle class imbalance
- Hyperparameter tuning with Optuna (20 trials)

### 5. Feature Selection
- Used SHAP values to identify 51 zero-importance features
- Dropped them — reduced from 199 to 147 features
- ROC-AUC maintained at 0.786

---

## Results

| Model | ROC-AUC |
|---|---|
| Logistic Regression (baseline) | 0.748 |
| XGBoost (default) | 0.752 |
| XGBoost (tuned) | 0.769 |
| XGBoost + bureau data | 0.774 |
| XGBoost + previous applications | 0.778 |
| XGBoost + installments | 0.781 |
| XGBoost + POS Cash | 0.782 |
| XGBoost + credit card | 0.787 |
| XGBoost + SHAP feature selection | **0.786** |

Kaggle leaderboard best: ~0.806

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) is used to explain both global feature importance and individual predictions.

**Top predictors globally:**
1. `EXT_SOURCE_2` — external credit score (strongest predictor)
2. `EXT_SOURCE_3` — external credit score
3. `EXT_SOURCE_1` — external credit score
4. `CODE_GENDER`
5. `CREDIT_TERM` — engineered feature
6. `LATE_PAYMENT_RATIO` — engineered feature
7. `DEBT_TO_CREDIT_RATIO` — engineered feature

The Gradio app shows a per-prediction SHAP waterfall chart explaining exactly which features drove that specific applicant's risk score up or down.

---

## Installation

### Prerequisites
- Python 3.10+
- Kaggle account (to download data)

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/Piyush-code-lab/home-credit-default-risk.git
cd home-credit-default-risk
```

**2. Create and activate virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the dataset**

Go to [Kaggle competition page](https://www.kaggle.com/competitions/home-credit-default-risk/data), download all files, and place them in the project root directory.

---

## Usage

### Run the full pipeline (train from scratch)
```bash
python main.py
```

This will:
- Load and preprocess all 6 data sources
- Engineer features
- Train XGBoost with optimal hyperparameters
- Run SHAP feature selection
- Retrain on clean feature set
- Save model and feature columns to `models/`

### Run the web application
```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser. Enter applicant details and get:
- Default probability (%)
- Risk level (HIGH / LOW)
- SHAP explanation chart showing top 10 factors

### Configuration

All settings are in `configs/config.yaml`. You can change:
- Data file paths
- Missing value threshold
- Outlier cap percentile
- Model hyperparameters
- Train/test split ratio

---

## Key Design Decisions

**Why XGBoost over neural networks?**
Tabular data with mixed feature types consistently favours gradient boosted trees over deep learning. Every major Kaggle tabular competition winner uses XGBoost or LightGBM. ANNs were considered but deprioritised given the marginal expected gain versus significant tuning cost.

**Why skip `bureau_balance.csv`?**
At 27 million rows with only 1.2% actual delinquency records, the signal-to-noise ratio is low and the two-step merge (bureau_balance → bureau → application) adds significant engineering complexity. The architecture supports adding it — see `data_loader.py` — as a future improvement.

**Why 99th percentile capping instead of IQR?**
Financial data is naturally right-skewed. IQR capping is too aggressive on income distributions and would remove legitimate high-income applicants. 99th percentile capping removes only the truly extreme values while preserving real variance.

**Why SHAP for feature selection?**
Rather than arbitrary correlation thresholds, SHAP provides model-native importance scores. 51 features had exactly zero importance — removing them reduces inference time and model complexity without touching performance.

---

## Future Improvements

- Integrate `bureau_balance.csv` for monthly delinquency history features
- Add LightGBM and CatBoost for ensemble stacking
- Implement proper cross-validation (stratified k-fold) instead of single train/test split
- Add model monitoring and drift detection
- Build a proper CI/CD pipeline with automated retraining

---

## Tech Stack

- **Data:** pandas, numpy
- **ML:** scikit-learn, XGBoost, Optuna
- **Explainability:** SHAP
- **Deployment:** Gradio, Hugging Face Spaces
- **Experiment tracking:** manual logging (MLflow integration planned)

---

## Author

**Piyush-code-lab** —  IIT Kharagpur

---

## License

MIT License