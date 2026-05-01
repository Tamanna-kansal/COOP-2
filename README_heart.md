Name -Tamanna Kansal 
Roll No- 2210992436
# Comparative Analysis of Machine Learning Algorithms for Heart Disease Prediction 

## Project Overview

This project builds and evaluates multiple machine learning models to predict the presence of heart disease in patients. It addresses class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)** and compares model performance before and after balancing. The best-performing model is **XGBoost**, achieving the highest accuracy and ROC-AUC score.

---

## Dataset

- **File:** `heart.csv`
- **Target Column:** `target` (1 = Heart Disease, 0 = No Disease)
- The dataset contains clinical features such as age, chest pain type, resting blood pressure, cholesterol, and more.

---

## Requirements

Install the required dependencies before running the notebook:

```bash
pip install imbalanced-learn xgboost scikit-learn pandas numpy matplotlib seaborn
```

Or run directly in the notebook:

```python
!pip install imbalanced-learn xgboost
```

---

## Project Structure

```
├── heart.csv                        # Input dataset
├── final__collab.ipynb              # Main Jupyter Notebook (Google Colab)
├── before_smote_metrics.png         # Performance chart before SMOTE
├── before_smote_prf.png             # Precision/Recall/F1 chart before SMOTE
├── feature_importance.png           # XGBoost feature importance plot
├── xgboost_smote_comparison.png     # Confusion matrix comparison (before vs after SMOTE)
├── roc_curve.png                    # ROC Curve for XGBoost
└── README.md
```

---

## How to Run

1. **Open in Google Colab** (recommended):
   - Upload `final__collab.ipynb` to [Google Colab](https://colab.research.google.com)
   - Mount Google Drive and place `heart.csv` at `/content/heart.csv` (or adjust the path in the notebook)

2. **Run all cells sequentially** from top to bottom.

3. Alternatively, run locally using Jupyter Notebook:
   ```bash
   jupyter notebook final__collab.ipynb
   ```

---

## Workflow

### 1. Data Loading & Exploration
- Load `heart.csv` and inspect shape, data types, missing values, and class distribution.

### 2. Preprocessing
- Separate features (`X`) and target (`y`)
- 80/20 Train-Test split with stratification
- Feature scaling using `StandardScaler`

### 3. Model Training — Before SMOTE
Five classifiers are trained and evaluated using **5-Fold Stratified Cross-Validation**:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

Metrics evaluated: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Overfitting Gap.

### 4. Handling Class Imbalance — SMOTE
- SMOTE is applied **only on the training data** to generate synthetic minority class samples.
- Models are retrained on the balanced dataset.

### 5. Model Training — After SMOTE
Same five models are retrained with regularization on SMOTE-balanced data and re-evaluated on the original test set.

### 6. Evaluation & Visualization
- Performance comparison tables (before vs after SMOTE)
- Bar charts: CV Accuracy, Test Accuracy, F1-Score
- Horizontal bar charts ranking models by key metrics
- Confusion matrices (XGBoost — before vs after SMOTE)
- XGBoost Feature Importance plot
- ROC Curve with AUC score

---

## Results Summary

| Model               | Test Accuracy (After SMOTE) |
|---------------------|-----------------------------|
| Logistic Regression | ~81%                        |
| Decision Tree       | ~87%                        |
| Random Forest       | ~92%                        |
| SVM                 | ~93%                        |
| **XGBoost**         | **Highest**                 |
0
### Key Findings
1. **XGBoost** achieved the highest accuracy after SMOTE.
2. **Random Forest** closely matched XGBoost performance.
3. SMOTE improved most models but slightly hurt SVM (-3.41%).
4. **Logistic Regression** achieved the best recall — suitable for screening use cases.
5. All models achieved **AUC above 0.92**.

---

## Models & Libraries Used

| Library              | Purpose                              |
|----------------------|--------------------------------------|
| `scikit-learn`       | ML models, preprocessing, evaluation |
| `xgboost`            | XGBoost classifier                   |
| `imbalanced-learn`   | SMOTE for class balancing            |
| `pandas` / `numpy`   | Data manipulation                    |
| `matplotlib` / `seaborn` | Visualization                    |

---

## Authors

BE-CSE, Batch 2022 — 8th Semester 
Chitkara University Institute of Engineering & Technology  
Course: COOP-2 (22CS421)
