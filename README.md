# Hospital Readmission Predictor

## Problem
Predicting whether a diabetic patient will be readmitted to hospital within 30 days, using the UCI Diabetes 130-US Hospitals dataset (100k patients).

## Why It Matters
Hospital readmissions cost the US healthcare system $26 billion annually. Identifying high-risk patients allows hospitals to provide targeted post-discharge care and reduce preventable readmissions.

## Dataset
- 101,766 patient records from 130 US hospitals (1999-2008)
- 50 features including demographics, diagnoses, medications and lab results
- Target: binary classification (readmitted within 30 days or not)
- Source: UCI ML Repository - Diabetes 130-US Hospitals

## Approach
1. Data Cleaning - handled missing values, '?' placeholders, duplicate patients
2. Feature Engineering - ICD-9 code grouping, medication encoding, age ordinalization
3. Class Imbalance - handled 9:1 imbalance using scale_pos_weight in XGBoost
4. Modeling - compared Logistic Regression, Random Forest, and XGBoost
5. Threshold Tuning - optimized classification threshold for medical context

## Results
| Model | ROC-AUC | Readmitted Recall |
|-------|---------|-------------------|
| Logistic Regression | 0.6508 | 0.54 |
| Random Forest | 0.6463 | 0.00 |
| XGBoost (default threshold) | 0.6834 | 0.61 |
| XGBoost (threshold=0.3) | 0.6834 | 0.97 |

## Key Findings
- Past inpatient visits is the strongest predictor of readmission
- Discharge disposition and number of diagnoses are also highly predictive
- Lowering classification threshold from 0.5 to 0.3 increased recall from 61% to 97%
- Catching 2208 out of 2271 at-risk patients vs 1392 at default threshold

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, SMOTE, Matplotlib, Seaborn

## How to Run
1. Download dataset from UCI ML Repository: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
2. Place it in the data/ folder as diabetic_data.csv
3. Run notebooks/readmission_model.ipynb
