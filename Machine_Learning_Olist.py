# ============================================================
# OLIST - MACHINE LEARNING ANALYSIS
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "olist_cohort_analysis_dataset.csv"
OUTPUT_DIR = "ML_Results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("OLIST MACHINE LEARNING ANALYSIS")
print("=" * 70)

print("\n[1/10] Loading dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 2. CREATE TARGET
# ============================================================

print("\n[2/10] Creating repeat-customer target...")

customer_col = None

for col in [
    "customer_unique_id",
    "customer_id",
    "customer_unique",
    "customer_identifier"
]:
    if col in df.columns:
        customer_col = col
        break

if customer_col is None:
    raise ValueError(
        "Customer ID column not found."
    )

order_counts = df.groupby(customer_col).size()

df["order_count"] = df[customer_col].map(order_counts)

df["repeat_customer"] = (
    df["order_count"] > 1
).astype(int)

print("Customer column:", customer_col)
print("\nTarget distribution:")
print(df["repeat_customer"].value_counts())


# ============================================================
# 3. SELECT FEATURES
# ============================================================

print("\n[3/10] Preparing machine-learning features...")

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

excluded = [
    "repeat_customer",
    "order_count"
]

features = [
    col for col in numeric_cols
    if col not in excluded
]

X = df[features].copy()
y = df["repeat_customer"].copy()

# Replace infinite values
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove completely empty columns
X = X.dropna(
    axis=1,
    how="all"
)

# Remove constant columns
X = X.loc[
    :,
    X.nunique(dropna=True) > 1
]

print("Features used:", len(X.columns))

for col in X.columns:
    print(" -", col)


# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

print("\n[4/10] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))


# ============================================================
# 5. VIF ANALYSIS
# ============================================================

print("\n[5/10] Performing multicollinearity analysis...")

X_vif = X_train.copy()

X_vif = X_vif.replace(
    [np.inf, -np.inf],
    np.nan
)

X_vif = X_vif.fillna(
    X_vif.median()
)

X_vif = X_vif.fillna(0)

vif_results = []

for i, col in enumerate(X_vif.columns):

    try:
        vif = variance_inflation_factor(
            X_vif.values,
            i
        )
    except:
        vif = np.inf

    vif_results.append(
        {
            "Feature": col,
            "VIF": vif
        }
    )

vif_df = pd.DataFrame(vif_results)

vif_df = vif_df.sort_values(
    "VIF",
    ascending=False
)

vif_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "vif_analysis.csv"
    ),
    index=False
)

print("\nVIF results:")
print(vif_df.to_string(index=False))


# ============================================================
# 6. MACHINE LEARNING MODELS
# ============================================================

print("\n[6/10] Training machine-learning models...")

models = {

    "Logistic Regression": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]),

    "Random Forest": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]),

    "Gradient Boosting": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE
            )
        )
    ])
}


# ============================================================
# 7. TRAIN + COMPARE
# ============================================================

print("\n[7/10] Comparing models...")

results = []
trained_models = {}

for name, model in models.items():

    print("Training:", name)

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    probability = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    precision = precision_score(
        y_test,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probability
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    })

    trained_models[name] = model


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "ROC-AUC",
    ascending=False
)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)

print("\nMODEL COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))


# ============================================================
# 8. BEST MODEL
# ============================================================

print("\n[8/10] Selecting best model...")

best_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_name
]

y_pred = best_model.predict(
    X_test
)

y_prob = best_model.predict_proba(
    X_test
)[:, 1]

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    y_prob
)

print("\nBEST MODEL:", best_name)
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(auc, 4))


# ============================================================
# 9. SAVE PERFORMANCE
# ============================================================

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]
})

metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_model_metrics.csv"
    ),
    index=False
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

print("\n[9/10] Calculating feature importance...")

classifier = best_model.named_steps["model"]

if hasattr(
    classifier,
    "feature_importances_"
):

    importance = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

elif hasattr(
    classifier,
    "coef_"
):

    importance = np.abs(
        classifier.coef_[0]
    )

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

else:

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": 0
    })


importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_importance.csv"
    ),
    index=False
)


# ============================================================
# 11. PLOTS
# ============================================================

print("\n[10/10] Creating professional plots...")


# Model comparison

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["ROC-AUC"]
)

plt.title(
    "Machine Learning Model Comparison"
)

plt.ylabel(
    "ROC-AUC"
)

plt.xlabel(
    "Model"
)

plt.ylim(
    0,
    1
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_model_comparison.png"
    ),
    dpi=300
)

plt.close()


# Confusion matrix

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title(
    "Confusion Matrix - " + best_name
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.xticks(
    [0, 1],
    ["One-time", "Repeat"]
)

plt.yticks(
    [0, 1],
    ["One-time", "Repeat"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ROC curve

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.title(
    "ROC Curve - " + best_name
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_roc_curve.png"
    ),
    dpi=300
)

plt.close()


# Feature importance

top_features = (
    importance_df
    .head(15)
    .sort_values(
        "Importance"
    )
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title(
    "Top 15 Feature Importance - " + best_name
)

plt.xlabel(
    "Importance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_feature_importance.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PREDICTIONS
# ============================================================

predictions = X_test.copy()

predictions["Actual"] = y_test.values
predictions["Predicted"] = y_pred
predictions["Repeat_Probability"] = y_prob

predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "predictions.csv"
    ),
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Item": [
        "Dataset Rows",
        "Dataset Columns",
        "Number of Features",
        "Best Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Value": [
        len(df),
        len(df.columns),
        len(X.columns),
        best_name,
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]
})

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ML_project_summary.csv"
    ),
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("MACHINE LEARNING ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nBest Model:", best_name)
print("ROC-AUC:", round(auc, 4))
print("F1 Score:", round(f1, 4))
print("Accuracy:", round(accuracy, 4))

print("\nAll results saved in:")
print("ML_Results")

print("\nFiles created:")

for file in sorted(
    os.listdir(OUTPUT_DIR)
):
    print(" -", file)

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)