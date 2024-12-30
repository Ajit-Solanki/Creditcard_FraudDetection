# Creditcard_FraudDetection

## Objective
This notebook demonstrates a supervised approach to detecting fraudulent transactions using the Kaggle Credit Card Fraud Detection dataset. It implements data preprocessing, model training, evaluation.

---

## Steps
### 1. Data Preprocessing
- Load the dataset.
- Handle missing values (if any).
- Apply **SMOTE** to address class imbalance.
- Scale features using **StandardScaler**.

### 2. Model Training
- Train a **Logistic Regression** model as a baseline.
- Train an **XGBoost** model as the primary supervised model.

### 4. Evaluation
- Compute metrics: **Precision**, **Recall**, **F1-score**, and **Accuracy**.
- Generate **Confusion Matrices**.
- Visualize **ROC-AUC** and **Precision-Recall Curves**.

### 5. Analysis
- Display feature importance from the XGBoost model.
- Use **SHAP** to interpret individual predictions.

## Requirements
- Python 3.8 or 3.9
- Libraries:
  - `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `xgboost`, `shap`, `matplotlib`, `joblib`
