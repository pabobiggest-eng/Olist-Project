
# Olist Final Expert ML

## Objective
Predict One-Time vs Repeat Customers.

## Target
repeat_customer = 1 when a customer has more than one DISTINCT order.

## Feature audit
Candidate features: 4
Applied features: 4
Skipped features: 17

See:
- 02_feature_audit.csv
- 12_final_feature_decisions.csv
- 13_features_applied.csv

## Imbalance
Training ratio BEFORE:
14.68:1

Balancing:
SMOTE (k_neighbors=5)

Training ratio AFTER:
1.00:1

Test set balanced: NO

## Leakage protection
Customer-level StratifiedGroupKFold.
Customer overlap: 0

## Diagnostics
- Skewness
- Kurtosis
- Correlation
- VIF

## Models
- Logistic Regression
- Random Forest
- Gradient Boosting

## Evaluation
Primary: PR-AUC
Secondary: F1
Also: Accuracy, Precision, Recall, ROC-AUC

## Best model
Logistic Regression

## Metrics
Accuracy: 0.9099
Precision: 0.3193
Recall: 0.3641
F1: 0.3402
ROC-AUC: 0.6845
PR-AUC: 0.4103

## Folders
tables/
plots/
