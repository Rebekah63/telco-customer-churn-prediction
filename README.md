# telco-customer-churn-prediction
End-to-end customer churn prediction using Random Forest from EDA and class imbalance handling to hyperparameter tuning and retention strategy recommendations.

> **Predicting customer churn using machine learning to enable proactive retention strategies**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

##  Problem Statement

In the telecommunications industry, customer churn — the rate at which customers stop doing business with a company — is one of the most critical metrics to manage. Acquiring a new customer costs **5–10x more** than retaining an existing one.

This project builds a machine learning pipeline to **predict which customers are likely to churn**, enabling the business to intervene with targeted retention strategies before it's too late.

---

##  Dataset

**Source:** [ Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Attribute | Detail |
|-----------|--------|
| Records | 7,043 customers |
| Features | 21 (demographics, services, account, billing) |
| Target | `Churn` (Yes / No) |
| Class Distribution | 73.5% No Churn / 26.5% Churn |

**Feature Categories:**
- **Demographics:** Gender, SeniorCitizen, Partner, Dependents
- **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account Info:** Tenure, Contract, PaperlessBilling, PaymentMethod
- **Financials:** MonthlyCharges, TotalCharges

---

##  Key Findings (EDA)

- **26.5% churn rate** — significant class imbalance requiring special handling
- **Tenure is right-skewed** — most customers are newer, and newer customers churn most
- **Month-to-month contract** holders churn at ~3x the rate of two-year contract holders
- **No Online Security / No Tech Support** customers churn significantly more
- **Electronic check** users have the highest churn rate among all payment methods
- **Fiber optic** internet customers churn more — possibly due to unmet value expectations
- Strong correlations: Tenure ↔ TotalCharges (0.83), MonthlyCharges ↔ TotalCharges (0.65)

---

##  Methodology

```
Raw Data → Cleaning → EDA → Preprocessing → SMOTE → Model Training → Tuning → Evaluation → Deployment
```

### Data Preparation
- Dropped `CustomerID` (non-predictive identifier)
- Converted `TotalCharges` from object → float; imputed 11 missing values with `0` (new customers)
- Label-encoded target variable (`Churn`: Yes=1, No=0)

### Preprocessing Pipeline
- **StandardScaler** on numerical features (SeniorCitizen, tenure, MonthlyCharges, TotalCharges)
- **OneHotEncoder** on 15 categorical features → expanded to 45 features
- **SMOTE** applied to training data to address class imbalance (balanced to 4,138 per class)

### Models Compared (5-Fold Cross-Validation)

| Model | CV Accuracy | Std Dev |
|-------|-------------|---------|
| Decision Tree | 78.69% | ±5.56% |
| **Random Forest** | **85.19%** | **±6.61%** |
| XGBoost | 83.60% | ±8.60% |

→ **Random Forest selected** as best performer.

### Hyperparameter Tuning
GridSearchCV with `scoring='recall'` across 96 parameter combinations (288 total fits):

```python
Best Parameters: {
  'criterion': 'gini',
  'max_depth': 10,
  'max_features': 'log2',
  'min_samples_leaf': 1,
  'min_samples_split': 2,
  'n_estimators': 200
}
```

---

## Results

### Tuned Random Forest — Test Set Performance

| Metric | No Churn | Churn |
|--------|----------|-------|
| Precision | 89% | 57% |
| Recall | 81% | **72%** |
| F1-Score | 85% | 64% |
| **Overall Accuracy** | — | **78%** |

### Confusion Matrix Comparison

|  | Baseline RF | Tuned RF | Change |
|--|-------------|----------|--------|
| True Positives (Churners caught) | 213 | **269** | +56 ↑ |
| False Negatives (Churners missed) | 160 | **104** | -56 ↓ |
| Churn Recall | 57% | **72%** | +15% ↑ |

> **Optimising for recall** was the deliberate choice — in retention scenarios, missing a churner (false negative) costs far more than a mistaken intervention (false positive).

---

## 🏆 Top 10 Feature Importances

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | TotalCharges | 0.1097 |
| 2 | Tenure | 0.1088 |
| 3 | MonthlyCharges | 0.0995 |
| 4 | Contract: Month-to-Month | 0.0703 |
| 5 | OnlineSecurity: No | 0.0571 |
| 6 | PaymentMethod: Electronic Check | 0.0498 |
| 7 | Contract: Two Year | 0.0385 |
| 8 | TechSupport: No | 0.0361 |
| 9 | InternetService: Fiber Optic | 0.0251 |
| 10 | PaperlessBilling: Yes | 0.0224 |

---

## 💡 Retention Strategy Recommendations

Based on feature importance, the following interventions are recommended:

| Driver | Strategy |
|--------|----------|
| High charges | Loyalty discounts, tiered pricing, bill shock alerts |
| Short tenure | Enhanced onboarding, 90-day check-ins, milestone rewards |
| Monthly contract | Contract upgrade incentives (15-25% discount) |
| No security/support | Bundle promotions, free 30-day trials |
| Electronic cheque | Incentivise auto-pay switching ($2-5/month discount) |

---

## 🗂️ Project Structure

```
telco-churn-prediction/
│
├── notebooks/
│   └── Telco_customer_churn_prediction.ipynb   # Full analysis notebook
│
├── models/
│   └── customer_churn_model.pkl                # Saved tuned Random Forest model
│
├── data/
│   └── README.md                               # Dataset download instructions
│
├── reports/
│   └── Telco_Churn_Prediction_Report.docx      # Full project report
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/telco-churn-prediction.git
cd telco-churn-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the `/data` folder.

**4. Run the notebook**
```bash
jupyter notebook notebooks/Telco_customer_churn_prediction.ipynb
```

**5. Make a prediction using the saved model**
```python
import pickle
import pandas as pd

with open('models/customer_churn_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

loaded_model = model_data['model']

# Example: New customer input
input_data = {
    'gender': ['Female'], 'SeniorCitizen': [0], 'Partner': ['Yes'],
    'Dependents': ['No'], 'tenure': [1], 'PhoneService': ['No'],
    'MultipleLines': ['No phone service'], 'InternetService': ['DSL'],
    'OnlineSecurity': ['No'], 'OnlineBackup': ['Yes'],
    'DeviceProtection': ['No'], 'TechSupport': ['No'],
    'StreamingTV': ['No'], 'StreamingMovies': ['No'],
    'Contract': ['Month-to-month'], 'PaperlessBilling': ['Yes'],
    'PaymentMethod': ['Electronic check'],
    'MonthlyCharges': [29.85], 'TotalCharges': [29.85]
}

# → Predicted: Churn = YES | Probability: 59.61%
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
jupyter
```

---

## 🔮 Next Steps

- [ ] Deploy model as a REST API (Flask / FastAPI)
- [ ] Build interactive  dashboard for real-time churn scoring
- [ ] Enrich features with support ticket history, NPS scores, app usage
- [ ] Implement MLOps pipeline for automated retraining (MLflow / SageMaker)
- [ ] Integrate Customer Lifetime Value (CLV) modelling for prioritisation
- [ ] Experiment with LightGBM, CatBoost, and neural network architectures

---

## 👤 Author

**[Rebecca Namuyanja]**
Data Scientist | Predictive Analytics

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/rebecca-namuyanja)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Rebekah63)

---

## 📄 License

This project is licensed under the MIT License.

---

*⭐ If you found this project useful, please consider starring the repository!*
