# Telco Customer Churn — Classification

Predicting which telecom customers will leave the company, using the IBM Telco Customer Churn dataset (7,043 customers).

## Goal
Build a model that catches as many churning customers as possible, and find the factors that drive churn.

## Workflow
1. **Data cleaning** — dropped IDs, location columns and leakage columns (`Churn Label`, `Churn Score`, `CLTV`, `Churn Reason`); fixed `Total Charges`.
2. **EDA** — distributions, categorical features vs target, correlation.
3. **Encoding** — one-hot encoding with `drop_first=True`.
4. **Train / test split** — 80/20, stratified.
5. **Scaling** — `StandardScaler` on numeric columns, inside a Pipeline.
6. **Imbalance** — two approaches compared:
   - **SMOTENC** (handles encoded categorical columns correctly)
   - **Class weighting** (`class_weight`, `scale_pos_weight`, `auto_class_weights`)
7. **Models** — Logistic Regression, KNN, SVM, Decision Tree, Bagging, Random Forest, AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost.
8. **Hyperparameter tuning** — `GridSearchCV` (5-fold, `scoring='f1'`) with all preprocessing inside an `imblearn` Pipeline to avoid data leakage.
9. **Model selection** — by CV F1, not by test results.
10. **Voting** — soft and hard voting with the top 3 models.
11. **Final evaluation, feature importance, and prediction for a new customer.**

## Results

| Model | F1 Train | CV F1 |
|---|---|---|
| Random Forest (class_weight) | 0.747 | 0.654 |
| XGBoost (class_weight) | 0.672 | 0.653 |
| Random Forest (SMOTENC) | 0.695 | 0.653 |
| Soft Voting | – | 0.654 |
| Hard Voting | – | 0.656 |

The top results were within ~0.003 of each other, so **XGBoost with class weights** was chosen as the simplest model with the least overfitting.

**Test set (final model):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Stay (0) | 0.92 | 0.73 | 0.81 |
| Leave (1) | 0.52 | 0.81 | 0.63 |

The model catches **81%** of churning customers (304 of 374).

**Most important features:** Contract type, Internet Service (Fiber optic), Payment Method (Electronic check), Dependents, Tenure Months.

## Files
- `Telco_Churn_GridSearch_SMOTENC_vs_ClassWeight.ipynb` — full notebook with outputs
- `Telco_customer_churn.csv` — dataset

## Tools
Python, pandas, scikit-learn, imbalanced-learn, XGBoost, LightGBM, CatBoost, matplotlib, seaborn

## Author
Amr Ahmed Bakr

## Kaggle Notebook
https://www.kaggle.com/code/amrbakr95/telco-churn-smotenc-vs-class-weighting-with-grids
