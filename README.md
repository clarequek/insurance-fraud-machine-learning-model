# Healthcare Insurance Fraud Detection Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning solution for detecting fraudulent healthcare providers using ensemble methods and advanced tree-based algorithms. This project implements a sophisticated multi-model approach with Extra Trees achieving **94.61% ROC-AUC** on validation data.

**Course:** BT4012 - Fraud Analytics  
**Institution:** National University of Singapore  
**Date:** November 2025

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Usage](#-usage)
- [Results](#-results)
- [Contributors](#-contributors)

---

## 🎯 Project Overview

This project addresses the critical challenge of healthcare provider fraud detection using advanced machine learning techniques. By analyzing historical claims data, provider information, and beneficiary details, we build predictive models to identify fraudulent healthcare providers.

**Business Problem:**
Healthcare fraud costs billions annually. This solution helps insurance companies:
- Identify high-risk providers for investigation
- Reduce fraudulent claim payments
- Allocate audit resources efficiently
- Improve claim approval processes

**Technical Solution:**
- 6 diverse machine learning models with comprehensive hyperparameter tuning
- Advanced feature engineering (138 features from raw data)
- Proper handling of class imbalance (38% fraud ratio)
- Stratified cross-validation to prevent data leakage
- Hyperparameter optimization using Optuna (390 total trials)

---

## ✨ Key Features

### Data Processing & Engineering
- **Comprehensive EDA:** Automated exploratory data analysis with statistical summaries, null rate analysis, and visualization
- **Advanced Feature Engineering:** 138 engineered features including:
  - Provider-level aggregations (mean reimbursement, claim counts)
  - Temporal features (claim patterns, date mismatches)
  - Chronic condition indicators
  - Diagnosis and procedure diversity metrics
  - Claim-to-deductible ratios
  - Arcsinh and log transformations for skewed distributions

### Machine Learning Pipeline
- **6 Optimized Models (Validation Set Performance):**
  - **Extra Trees (Winner): ROC-AUC: 94.61%, PR-AUC: 91.58%**
  - HistGradientBoosting: ROC-AUC: 91.21%, PR-AUC: 86.35%
  - LightGBM: ROC-AUC: 90.74%, PR-AUC: 86.28%
  - CatBoost: ROC-AUC: 90.40%, PR-AUC: 84.68%
  - XGBoost: ROC-AUC: 90.37%, PR-AUC: 85.18%
  - Random Forest: ROC-AUC: 85.64%, PR-AUC: 81.47%

- **Model Selection:**
  - Extra Trees selected as production model
  - Largest improvement from tuning: +22.05% ROC-AUC gain
  - Most efficient: Only 150 trees needed

- **Hyperparameter Optimization:**
  - Optuna-based Bayesian optimization
  - TPE sampler with median pruner
  - 100+ trials per model

### Model Interpretation
- Feature importance analysis across all models
- Aggregated feature rankings
- SHAP-ready predictions for explainability

---

## 📊 Dataset

**Source:** Healthcare provider fraud detection dataset from Kaggle

**Data Components:**
- **Training Set:** 558,211 samples with fraud labels (390,309 for training, 167,902 for validation in 70/30 split)
- **Test Set:** 135,392 samples for prediction
- **Fraud Ratio:** 38.12% (moderately imbalanced)

**Data Files:**
```
data/
├── Train_Beneficiarydata-1542865627584.csv
├── Train_Inpatientdata-1542865627584.csv
├── Train_Outpatientdata-1542865627584.csv
├── Train-1542865627584.csv
├── Test_Beneficiarydata-1542969243754.csv
├── Test_Inpatientdata-1542969243754.csv
├── Test_Outpatientdata-1542969243754.csv
└── Test-1542969243754.csv
```

**Key Variables:**
- **Provider Information:** Provider ID, state, county, specialty
- **Claim Details:** Claim amounts, reimbursement, deductibles, dates
- **Beneficiary Data:** Age, gender, chronic conditions, coverage months
- **Diagnoses & Procedures:** ICD codes, procedure codes, admission details

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 8GB+ RAM recommended

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/clarequek/insurance-fraud-machine-learning-model.git
cd insurance-fraud-machine-learning-model
```

2. **Create a virtual environment (recommended):**
```bash
# Using conda
conda create -n fraud-detection python=3.8
conda activate fraud-detection

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Required Packages
```
numpy
pandas
scikit-learn
xgboost
lightgbm
catboost
matplotlib
seaborn
scipy
tqdm
joblib
openpyxl
optuna
```

---

## 📁 Project Structure

```
insurance-fraud-machine-learning-model/
│
├── Group_10_Notebook.ipynb          # Main analysis notebook
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
│
├── data/                            # Raw data files
│   ├── Train_*.csv
│   └── Test_*.csv
│
├── curated/                         # Cleaned and merged data
│
├── features/                        # Engineered features
│
├── eda_report/                      # EDA outputs
│   ├── schema_summary.csv
│   ├── null_rates.csv
│   ├── linkage_coverage.csv
│   └── plots/
│
├── models/                          # Saved model artifacts
│   ├── model_performance_metrics_*.json
│   ├── model_hyperparameters_*.json
│   ├── selected_features_*.json
│   └── test_predictions_*.csv
│
├── artifacts/                       # Analysis artifacts
│   ├── best_hyperparameters.csv
│   ├── feature_importance_aggregated.csv
│   ├── meta_learner_hyperparameters.csv
│   └── selected_features_for_stacking.csv
│
└── results/                         # Model results
    └── detailed_baseline_results.csv
```

---

## 🔬 Methodology

### 1. Data Preprocessing
- **Cleaning:** Standardized column names, handled missing values, removed duplicates
- **Merging:** Combined inpatient/outpatient claims with beneficiary and provider data
- **Validation:** Checked data linkage coverage and key consistency

### 2. Exploratory Data Analysis
- Schema analysis across all tables
- Null rate computation
- Statistical summaries (numeric and categorical)
- Target variable distribution analysis
- Correlation analysis

### 3. Feature Engineering
Engineered 138 features across multiple categories:

**Provider Aggregations:**
- Mean/median reimbursement amounts
- Total claim counts
- Deductible ratios
- Refund indicators

**Temporal Features:**
- Claim month, quarter, year, day of week
- Date mismatches
- Admission duration (length of stay)

**Medical Features:**
- Chronic condition counts and indicators
- Diagnosis diversity and counts
- Procedure diversity and counts
- Specific diagnosis/procedure flags (top ICD codes)

**Transformations:**
- Arcsinh transformations for skewed distributions
- Log1p transformations
- Binary indicators (e.g., max deductible flags)

### 4. Model Development

**Base Learners:**
- **Boosting Models:** XGBoost, LightGBM, CatBoost, HistGradientBoosting
- **Bagging Models:** Random Forest, Extra Trees

**Why This Combination?**
- Boosting: Sequential learning, handles complexity well
- Bagging: Parallel learning, reduces variance
- Diversity: Different algorithms capture different patterns

**Cross-Validation Strategy:**
- 5-fold Stratified K-Fold
- Group-based splitting by provider to prevent leakage
- Out-of-fold predictions for stacking

**Hyperparameter Tuning:**
- Framework: Optuna with TPE sampler
- Objective: ROC-AUC maximization
- Trials: 390 total trials (varies per model)
- Pruning: Median pruner for efficiency
- Evaluation: 5-fold stratified cross-validation on 70% training data

### 5. Model Selection

**Final Model:** Extra Trees (standalone model)

**Why Extra Trees Won:**
- Highest validation performance: 94.61% ROC-AUC, 91.58% PR-AUC
- Largest improvement from tuning: +22.05% absolute gain
- Most efficient: Only 150 trees vs 450-950 for other models
- Strong generalization: Robust performance on validation set
- Stacking ensemble did not improve upon Extra Trees performance

---

## 📈 Model Performance

### Final Validation Set Results (30% Holdout)

| Rank | Model | ROC-AUC | PR-AUC | Improvement from Baseline |
|------|-------|---------|---------|---------------------------|
| 🥇 | **Extra Trees** | **94.61%** | **91.58%** | **+22.05%** |
| 🥈 | HistGradientBoosting | 91.21% | 86.35% | +0.42% |
| 🥉 | LightGBM | 90.74% | 86.28% | +0.21% |
| 4 | CatBoost | 90.40% | 84.68% | -0.78% |
| 5 | XGBoost | 90.37% | 85.18% | -0.40% |
| 6 | Random Forest | 85.64% | 81.47% | +2.68% |

### Baseline vs Tuned Performance

| Model | Baseline ROC-AUC | Tuned ROC-AUC | Absolute Gain |
|-------|------------------|---------------|---------------|
| **Extra Trees** | **72.55%** | **94.61%** | **+22.05%** |
| Random Forest | 82.96% | 85.64% | +2.68% |
| HistGradientBoosting | 90.80% | 91.21% | +0.42% |
| LightGBM | 90.53% | 90.74% | +0.21% |
| XGBoost | 90.76% | 90.37% | -0.40% |
| CatBoost | 91.18% | 90.40% | -0.78% |

### Top 10 Most Important Features

1. `provider_mean_reimbursed` - Average reimbursement per provider
2. `state` - Provider state
3. `county` - Provider county
4. `opannualdeductibleamt` - Outpatient annual deductible
5. `diagnosisgroupcode` - Diagnosis group classification
6. `admit_los_days` - Length of stay for admissions
7. `deductibleamtpaid` - Deductible amount paid
8. `noofmonths_partacov` - Months of Part A coverage
9. `gender` - Beneficiary gender
10. `race` - Beneficiary race

---

## 💻 Usage

### Running the Full Pipeline

Open and execute the Jupyter notebook:

```bash
jupyter notebook Group_10_Notebook.ipynb
```

The notebook includes:
1. **Data Loading & Cleaning** - Cells 1-20
2. **EDA** - Cells 21-40
3. **Feature Engineering** - Cells 41-60
4. **Model Training** - Cells 61-80
5. **Hyperparameter Tuning** - Cells 81-100
6. **Stacking Ensemble** - Cells 101-120
7. **Prediction & Evaluation** - Cells 121-140

### Making Predictions

```python
# Load trained models and make predictions
from joblib import load

# Load the stacking ensemble
meta_learner = load('models/meta_learner.pkl')
base_models = load('models/base_models.pkl')

# Make predictions on new data
predictions = meta_learner.predict_proba(base_predictions)[:, 1]
```

### Viewing Results

Results are automatically saved to:
- `models/test_predictions_*.csv` - Test set predictions
- `models/model_performance_metrics_*.json` - Performance metrics
- `artifacts/feature_importance_aggregated.csv` - Feature rankings

---

## 📊 Results

### Key Findings

1. **Extra Trees Dominates:** Extra Trees significantly outperforms all other models with 94.61% ROC-AUC, a 3.4% margin over second place

2. **Hyperparameter Tuning Impact:** Extra Trees showed the largest improvement (+22.05%) from systematic optimization using Optuna

3. **Provider Behavior is Critical:** Provider-level aggregations (mean reimbursement, location) are the strongest fraud indicators

4. **Geographic Patterns:** State and county show significant predictive power, suggesting regional fraud patterns

5. **Efficiency Matters:** Extra Trees achieves best performance with only 150 trees, making it both accurate and computationally efficient

6. **Ensemble Not Always Better:** Stacking ensemble did not improve upon Extra Trees, demonstrating that simpler models can outperform complex ensembles

### Production Model

- **Selected Model:** Extra Trees
- **Performance:** 94.61% ROC-AUC, 91.58% PR-AUC on validation set
- **Architecture:** 150 trees, max_depth=30, min_samples_split=8
- **Total Test Samples:** 135,392
- **Results saved to:** `models/test_predictions_detailed_20251121_200023.csv`

---

## 👥 Contributors

- **Quek Ying Han Clare**
- **Eunice Gong Shi Min** 
- **[Ow Zheng Wei]** 

**Course:** BT4012 - Fraud Analytics  
**Institution:** National University of Singapore  
**Academic Year:** 2024/2025

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

We would like to express our sincere gratitude to:

- **Professor Zhao Yiliang** - For his invaluable guidance and expertise throughout this project
- **Professor Zhao Rui** - For his constant support and insightful feedback
- **TA Sun Yichen** - For his dedication and assistance in helping us navigate challenges

Their mentorship has been instrumental in the success of this fraud detection project.

We also acknowledge:
- Dataset provided by Kaggle
- Open-source ML libraries: scikit-learn, XGBoost, LightGBM, CatBoost, Optuna

---

## 📧 Contact

For questions or collaborations, please contact:
- Clare Quek: [e1156061@u.nus.edu](mailto:e1156061@u.nus.edu)
- Eunice Gong: [e1155847@u.nus.edu](mailto:e1155847@u.nus.edu)
- Ow Zheng We: [e1122416@u.nus.edu]((mailto:e1122416@u.nus.edu))

---
