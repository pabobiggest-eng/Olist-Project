
# ============================================================
# OLIST FINAL EXPERT ML
# ============================================================

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================

DATA_FILENAME = "olist_cohort_analysis_dataset.csv"
RANDOM_STATE = 42
N_SPLITS = 5
CORR_THRESHOLD = 0.90
VIF_THRESHOLD = 10.0
MISSING_THRESHOLD = 0.50

SCRIPT_DIR = Path(__file__).resolve().parent
CURRENT_DIR = Path.cwd()


# ============================================================
# HELPERS
# ============================================================

def find_dataset():
    places = [
        CURRENT_DIR / DATA_FILENAME,
        SCRIPT_DIR / DATA_FILENAME,
        SCRIPT_DIR.parent / DATA_FILENAME
    ]

    for path in places:
        if path.is_file():
            return path

    for root in [SCRIPT_DIR, SCRIPT_DIR.parent]:
        if root.exists():
            for path in root.rglob(DATA_FILENAME):
                if path.is_file():
                    return path

    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILENAME}"
    )


def find_column(df, choices):
    for column in choices:
        if column in df.columns:
            return column
    return None


def save_csv(df, filename):
    df.to_csv(
        TABLE_DIR / filename,
        index=False
    )


def save_plot(filename):
    plt.savefig(
        PLOT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def ratio(one_time, repeat):
    if repeat == 0:
        return "Undefined / Infinite"
    return f"{one_time / repeat:.2f}:1"


def class_table(y, stage):
    y = pd.Series(y)
    counts = y.value_counts().reindex(
        [0, 1],
        fill_value=0
    )
    total = len(y)

    return pd.DataFrame({
        "Stage": [stage, stage],
        "Class": ["One-Time (0)", "Repeat (1)"],
        "Count": [
            int(counts.loc[0]),
            int(counts.loc[1])
        ],
        "Percentage": [
            counts.loc[0] / total * 100,
            counts.loc[1] / total * 100
        ]
    })


def diagnostic_impute(df):
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    values = SimpleImputer(
        strategy="median"
    ).fit_transform(df)

    return pd.DataFrame(
        values,
        columns=df.columns,
        index=df.index
    )


def high_corr_pairs(df, threshold):
    if df.shape[1] < 2:
        return pd.DataFrame(
            columns=[
                "Feature_1",
                "Feature_2",
                "Correlation",
                "Absolute_Correlation"
            ]
        )

    corr = df.corr()
    rows = []

    for i, first in enumerate(corr.columns):
        for j in range(i + 1, len(corr.columns)):
            second = corr.columns[j]
            value = corr.iloc[i, j]

            if pd.notna(value) and abs(value) >= threshold:
                rows.append({
                    "Feature_1": first,
                    "Feature_2": second,
                    "Correlation": value,
                    "Absolute_Correlation": abs(value)
                })

    if not rows:
        return pd.DataFrame(
            columns=[
                "Feature_1",
                "Feature_2",
                "Correlation",
                "Absolute_Correlation"
            ]
        )

    return pd.DataFrame(rows).sort_values(
        "Absolute_Correlation",
        ascending=False
    )


def vif_table(df):
    if df.shape[1] < 2:
        return pd.DataFrame(
            columns=["Feature", "VIF", "Status"]
        )

    clean = diagnostic_impute(df)

    constants = [
        c for c in clean.columns
        if clean[c].nunique() <= 1
    ]

    clean = clean.drop(
        columns=constants,
        errors="ignore"
    )

    if clean.shape[1] < 2:
        return pd.DataFrame(
            columns=["Feature", "VIF", "Status"]
        )

    rows = []

    for i, feature in enumerate(clean.columns):
        try:
            value = variance_inflation_factor(
                clean.values,
                i
            )
        except Exception:
            value = np.inf

        if np.isinf(value):
            status = "Severe / Infinite"
        elif value >= VIF_THRESHOLD:
            status = "High"
        elif value >= 5:
            status = "Moderate"
        else:
            status = "Acceptable"

        rows.append({
            "Feature": feature,
            "VIF": value,
            "Status": status
        })

    return pd.DataFrame(rows).sort_values(
        "VIF",
        ascending=False
    )


def is_id(column):
    name = str(column).lower().strip()

    exact = {
        "customer_id", "customer_unique_id",
        "customer_identifier", "order_id",
        "order_identifier", "order_number",
        "seller_id", "product_id", "review_id",
        "payment_id", "postal_code", "zip_code"
    }

    if name in exact:
        return True

    return (
        name.endswith("_id")
        or name.startswith("id_")
        or "identifier" in name
        or "postal" in name
        or "zip_code" in name
    )


def is_leakage(column):
    name = str(column).lower().strip()

    exact = {
        "repeat_customer", "order_count",
        "total_orders", "customer_order_count",
        "lifetime_orders", "future_orders",
        "future_purchases", "next_order",
        "next_purchase", "target"
    }

    if name in exact:
        return True

    return any(
        token in name
        for token in [
            "future_",
            "next_order",
            "next_purchase",
            "repeat_customer"
        ]
    )


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 85)
print("OLIST FINAL EXPERT MACHINE LEARNING")
print("=" * 85)

DATA_FILE = find_dataset()

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

OUTPUT_DIR = DATA_FILE.parent / "ML_Results"
TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("\nDataset loaded successfully.")
print("Dataset:", DATA_FILE)
print("Rows   :", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 2. PROJECT PROCEDURE / COMPONENTS
# ============================================================

procedure = pd.DataFrame({
    "Step": range(1, 13),
    "Procedure": [
        "Load data",
        "Detect customer/order columns",
        "Create repeat-customer target",
        "Feature audit",
        "Skewness and kurtosis",
        "Correlation and VIF",
        "Customer-level train/test split",
        "Measure class imbalance",
        "SMOTE on training data only",
        "Train and compare models",
        "Evaluate on untouched test set",
        "Save all tables, plots and documentation"
    ],
    "Strategy": [
        "Use Olist cohort dataset",
        "Use IDs only for grouping/target construction",
        "Repeat = more than one distinct order",
        "Exclude IDs, leakage, non-numeric, constant and high-missing fields",
        "Understand distribution shape before modeling",
        "Reduce multicollinearity using training data only",
        "Keep customers separated between train and test",
        "Measure real class ratio first",
        "Balance only training data",
        "Compare three classifiers",
        "Do not rebalance test data",
        "Create GitHub-ready outputs"
    ]
})

components = pd.DataFrame({
    "Component": [
        "Python", "Pandas", "NumPy", "Matplotlib",
        "Scikit-learn", "Statsmodels", "Imbalanced-learn",
        "Target", "Split", "Balancing",
        "Models", "Primary Metric"
    ],
    "Purpose": [
        "Programming",
        "Data processing",
        "Numerical analysis",
        "Visualization",
        "Preprocessing, models and metrics",
        "VIF / multicollinearity",
        "SMOTE / oversampling",
        "Repeat vs One-Time classification",
        "Customer-level grouped split",
        "SMOTE on training only",
        "Logistic, Random Forest, Gradient Boosting",
        "PR-AUC for imbalanced classification"
    ]
})

save_csv(
    procedure,
    "00_project_procedure_strategy.csv"
)

save_csv(
    components,
    "00_components_equipment.csv"
)


# ============================================================
# 3. CUSTOMER + ORDER + TARGET
# ============================================================

customer_col = find_column(
    df,
    [
        "customer_unique_id",
        "customer_id",
        "customer_unique",
        "customer_identifier"
    ]
)

order_col = find_column(
    df,
    [
        "order_id",
        "order_identifier",
        "order_number"
    ]
)

if customer_col is None:
    raise ValueError(
        "Customer column not found."
    )

if order_col is None:
    raise ValueError(
        "Order column not found."
    )

# Distinct order count prevents item-level rows from falsely
# making one order look like repeated orders.
distinct_orders = (
    df.groupby(customer_col)[order_col]
    .nunique()
)

df["repeat_customer"] = (
    df[customer_col]
    .map(distinct_orders)
    .fillna(0)
    .gt(1)
    .astype(int)
)

TARGET = "repeat_customer"

full_counts = (
    df[TARGET]
    .value_counts()
    .reindex(
        [0, 1],
        fill_value=0
    )
)

one_time_full = int(full_counts.loc[0])
repeat_full = int(full_counts.loc[1])

print("\nTARGET")
print("One-Time:", one_time_full)
print("Repeat  :", repeat_full)
print(
    "One-Time : Repeat:",
    ratio(
        one_time_full,
        repeat_full
    )
)

save_csv(
    class_table(
        df[TARGET],
        "FULL DATASET"
    ),
    "01_full_target_distribution.csv"
)


# ============================================================
# 4. COMPLETE FEATURE AUDIT
# ============================================================

audit_rows = []

for column in df.columns:

    decision = "USE"
    reason = "Eligible numeric predictor"

    if column == TARGET:
        decision = "SKIP"
        reason = "Target variable"

    elif is_id(column):
        decision = "SKIP"
        reason = "Identifier / key column"

    elif is_leakage(column):
        decision = "SKIP"
        reason = "Target-derived or future-information leakage risk"

    elif not pd.api.types.is_numeric_dtype(
        df[column]
    ):
        decision = "SKIP"
        reason = "Non-numeric categorical/text column"

    elif df[column].nunique(
        dropna=True
    ) <= 1:
        decision = "SKIP"
        reason = "Constant / single-value column"

    elif df[column].isna().mean() >= MISSING_THRESHOLD:
        decision = "SKIP"
        reason = "50% or more values missing"

    audit_rows.append({
        "Column": column,
        "Data_Type": str(df[column].dtype),
        "Missing_Count": int(
            df[column].isna().sum()
        ),
        "Missing_Percentage": round(
            df[column].isna().mean() * 100,
            4
        ),
        "Unique_Count": int(
            df[column].nunique(dropna=True)
        ),
        "Decision": decision,
        "Reason": reason
    })

audit_df = pd.DataFrame(
    audit_rows
)

candidate_features = audit_df.loc[
    audit_df["Decision"] == "USE",
    "Column"
].tolist()

save_csv(
    audit_df,
    "02_feature_audit.csv"
)

print("\nFEATURE AUDIT")
print("Total columns:", len(df.columns))
print("Candidate features:", len(candidate_features))

print("\nUSED INITIALLY:")
for feature in candidate_features:
    print(" +", feature)

print("\nSKIPPED + REASON:")
for _, row in audit_df[
    audit_df["Decision"] == "SKIP"
].iterrows():
    print(
        f" - {row['Column']} -> {row['Reason']}"
    )


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_candidate = df[
    candidate_features
].copy()

y = df[TARGET].copy()

groups = df[
    customer_col
].astype(str)

splitter = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

try:
    train_idx, test_idx = next(
        splitter.split(
            X_candidate,
            y,
            groups
        )
    )
except ValueError as exc:
    raise ValueError(
        "Customer-level stratified split failed. "
        "There may be too few repeat-customer groups."
    ) from exc

X_train_candidate = X_candidate.iloc[
    train_idx
].copy()

X_test_candidate = X_candidate.iloc[
    test_idx
].copy()

y_train = y.iloc[
    train_idx
].copy()

y_test = y.iloc[
    test_idx
].copy()

train_customer_ids = groups.iloc[
    train_idx
].unique()

test_customer_ids = groups.iloc[
    test_idx
].unique()

overlap = set(
    train_customer_ids
).intersection(
    set(test_customer_ids)
)

if overlap:
    raise RuntimeError(
        "Customer leakage detected."
    )

print("\nSPLIT")
print("Training rows:", len(X_train_candidate))
print("Testing rows :", len(X_test_candidate))
print("Customer overlap:", len(overlap))

save_csv(
    pd.concat(
        [
            class_table(
                y_train,
                "TRAIN BEFORE BALANCING"
            ),
            class_table(
                y_test,
                "TEST ORIGINAL / UNTOUCHED"
            )
        ],
        ignore_index=True
    ),
    "03_train_test_distribution.csv"
)


# ============================================================
# 6. SKEWNESS / KURTOSIS
# ============================================================

skew_rows = []

for feature in candidate_features:

    values = pd.to_numeric(
        X_train_candidate[feature],
        errors="coerce"
    ).dropna()

    skew_rows.append({
        "Feature": feature,
        "Skewness": values.skew() if len(values) else np.nan,
        "Kurtosis": values.kurtosis() if len(values) else np.nan
    })

skew_df = pd.DataFrame(
    skew_rows
)

save_csv(
    skew_df.sort_values(
        "Skewness",
        key=lambda s: s.abs(),
        ascending=False
    ),
    "04_skewness_kurtosis.csv"
)

if not skew_df.empty:

    plot = skew_df.dropna(
        subset=["Skewness"]
    ).sort_values("Skewness")

    plt.figure(figsize=(10, 6))
    plt.barh(
        plot["Feature"],
        plot["Skewness"]
    )
    plt.axvline(0)
    plt.title("Feature Skewness")
    plt.xlabel("Skewness")
    plt.ylabel("Feature")

    save_plot(
        "08_skewness.png"
    )


# ============================================================
# 7. TRAINING-ONLY CORRELATION
# ============================================================

X_train_diag = diagnostic_impute(
    X_train_candidate
)

train_corr = X_train_diag.corr()

save_csv(
    train_corr.reset_index().rename(
        columns={"index": "Feature"}
    ),
    "05_training_correlation_matrix.csv"
)

corr_pairs = high_corr_pairs(
    X_train_diag,
    CORR_THRESHOLD
)

save_csv(
    corr_pairs,
    "06_training_high_correlation_pairs.csv"
)

# Correlation plot
plt.figure(figsize=(13, 10))

plt.imshow(
    train_corr.values,
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(train_corr.columns)),
    train_corr.columns,
    rotation=90
)

plt.yticks(
    range(len(train_corr.columns)),
    train_corr.columns
)

plt.title(
    "Training Feature Correlation"
)

plt.tight_layout()

save_plot(
    "09_correlation_heatmap.png"
)


# ============================================================
# 8. TRAINING-ONLY VIF
# ============================================================

initial_vif = vif_table(
    X_train_candidate
)

save_csv(
    initial_vif,
    "07_vif_initial.csv"
)

# Plot initial VIF
if not initial_vif.empty:

    vif_plot = (
        initial_vif
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["VIF"])
        .head(20)
        .sort_values("VIF")
    )

    if not vif_plot.empty:

        plt.figure(figsize=(10, 7))

        plt.barh(
            vif_plot["Feature"],
            vif_plot["VIF"]
        )

        plt.axvline(
            VIF_THRESHOLD,
            linestyle="--"
        )

        plt.title(
            "Initial VIF Analysis"
        )

        plt.xlabel("VIF")
        plt.ylabel("Feature")

        save_plot(
            "10_vif_analysis.png"
        )


# ============================================================
# 9. FINAL FEATURE SELECTION
# ============================================================

features_to_drop = set()

# Remove one feature from highly correlated pairs.
for _, row in corr_pairs.iterrows():

    f1 = row["Feature_1"]
    f2 = row["Feature_2"]

    missing_f1 = (
        X_train_candidate[f1].isna().mean()
    )

    missing_f2 = (
        X_train_candidate[f2].isna().mean()
    )

    if missing_f1 > missing_f2:
        features_to_drop.add(f1)
    elif missing_f2 > missing_f1:
        features_to_drop.add(f2)
    else:
        features_to_drop.add(
            max(f1, f2)
        )

final_features = [
    feature
    for feature in candidate_features
    if feature not in features_to_drop
]

# Remove highest VIF repeatedly.
while len(final_features) >= 2:

    current_vif = vif_table(
        X_train_candidate[
            final_features
        ]
    )

    if current_vif.empty:
        break

    highest = current_vif.iloc[0]

    if pd.isna(
        highest["VIF"]
    ):
        break

    if highest["VIF"] <= VIF_THRESHOLD:
        break

    feature_to_remove = (
        highest["Feature"]
    )

    final_features.remove(
        feature_to_remove
    )

    features_to_drop.add(
        feature_to_remove
    )

if not final_features:
    raise ValueError(
        "No features remain after correlation/VIF screening."
    )

final_vif = vif_table(
    X_train_candidate[
        final_features
    ]
)

save_csv(
    final_vif,
    "11_vif_final_features.csv"
)

final_decisions = []

for column in df.columns:

    if column == TARGET:
        decision = "SKIP"
        reason = "Target variable"

    elif column in final_features:
        decision = "APPLY"
        reason = "Final model predictor"

    elif column in features_to_drop:
        decision = "SKIP"
        reason = (
            "Removed by training-only high correlation "
            "and/or high VIF"
        )

    else:

        row = audit_df[
            audit_df["Column"] == column
        ]

        decision = "SKIP"

        if not row.empty:
            reason = row["Reason"].iloc[0]
        else:
            reason = "Not selected"

    final_decisions.append({
        "Column": column,
        "Decision": decision,
        "Reason": reason
    })

final_decisions_df = pd.DataFrame(
    final_decisions
)

save_csv(
    final_decisions_df,
    "12_final_feature_decisions.csv"
)

save_csv(
    pd.DataFrame({
        "Feature_Number": range(
            1,
            len(final_features) + 1
        ),
        "Feature": final_features
    }),
    "13_features_applied.csv"
)

print("\nFINAL FEATURES APPLIED:", len(final_features))

for feature in final_features:
    print(" +", feature)

print("\nFINAL SKIPPED:")
for _, row in final_decisions_df[
    final_decisions_df["Decision"] == "SKIP"
].iterrows():
    print(
        f" - {row['Column']} -> {row['Reason']}"
    )


# ============================================================
# 10. IMBALANCE BEFORE / AFTER
# ============================================================

train_counts = (
    y_train
    .value_counts()
    .reindex(
        [0, 1],
        fill_value=0
    )
)

before_one_time = int(
    train_counts.loc[0]
)

before_repeat = int(
    train_counts.loc[1]
)

print("\n" + "=" * 85)
print("CLASS IMBALANCE")
print("=" * 85)

print(
    "TRAIN BEFORE:",
    before_one_time,
    "One-Time /",
    before_repeat,
    "Repeat"
)

print(
    "RATIO BEFORE:",
    ratio(
        before_one_time,
        before_repeat
    )
)

if before_repeat >= 2:

    smote_k = min(
        5,
        before_repeat - 1
    )

    method_name = (
        f"SMOTE (k_neighbors={smote_k})"
    )

else:

    smote_k = None

    method_name = (
        "RandomOverSampler fallback"
    )

# Diagnostic class balancing only for reporting.
balance_input = diagnostic_impute(
    X_train_candidate[
        final_features
    ]
)

if before_repeat >= 2:
    sampler = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=smote_k
    )
else:
    sampler = RandomOverSampler(
        random_state=RANDOM_STATE
    )

_, y_after = sampler.fit_resample(
    balance_input,
    y_train
)

after_counts = (
    pd.Series(y_after)
    .value_counts()
    .reindex(
        [0, 1],
        fill_value=0
    )
)

after_one_time = int(
    after_counts.loc[0]
)

after_repeat = int(
    after_counts.loc[1]
)

print(
    "METHOD:",
    method_name
)

print(
    "TRAIN AFTER:",
    after_one_time,
    "One-Time /",
    after_repeat,
    "Repeat"
)

print(
    "RATIO AFTER:",
    ratio(
        after_one_time,
        after_repeat
    )
)

print(
    "TEST SET CHANGED? NO"
)

save_csv(
    pd.DataFrame({
        "Metric": [
            "One-Time Before",
            "Repeat Before",
            "One-Time % Before",
            "Repeat % Before",
            "Ratio Before",
            "Balancing Method",
            "One-Time After",
            "Repeat After",
            "One-Time % After",
            "Repeat % After",
            "Ratio After",
            "Test Set Changed"
        ],
        "Value": [
            before_one_time,
            before_repeat,
            before_one_time / len(y_train) * 100,
            before_repeat / len(y_train) * 100,
            ratio(
                before_one_time,
                before_repeat
            ),
            method_name,
            after_one_time,
            after_repeat,
            after_one_time / len(y_after) * 100,
            after_repeat / len(y_after) * 100,
            ratio(
                after_one_time,
                after_repeat
            ),
            "NO"
        ]
    }),
    "14_imbalance_before_after.csv"
)

positions = np.arange(2)
width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(
    positions - width / 2,
    [before_one_time, after_one_time],
    width,
    label="One-Time"
)

plt.bar(
    positions + width / 2,
    [before_repeat, after_repeat],
    width,
    label="Repeat"
)

plt.xticks(
    positions,
    ["Before", "After"]
)

plt.title(
    "Training Class Distribution Before vs After Balancing"
)

plt.xlabel("Stage")
plt.ylabel("Samples")
plt.legend()

save_plot(
    "01_imbalance_before_after.png"
)


# ============================================================
# 11. MODELS
# ============================================================

def new_sampler():
    if before_repeat >= 2:
        return SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=smote_k
        )
    return RandomOverSampler(
        random_state=RANDOM_STATE
    )


models = {

    "Logistic Regression": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "sampler",
            new_sampler()
        ),
        (
            "power",
            PowerTransformer(
                method="yeo-johnson",
                standardize=False
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
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
            "sampler",
            new_sampler()
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
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
            "sampler",
            new_sampler()
        ),
        (
            "model",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE
            )
        )
    ])
}


# ============================================================
# 12. TRAIN + COMPARE
# ============================================================

X_train = X_train_candidate[
    final_features
].copy()

X_test = X_test_candidate[
    final_features
].copy()

results = []
trained_models = {}
model_outputs = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    prob = model.predict_proba(
        X_test
    )[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(
            y_test,
            pred
        ),
        "Precision": precision_score(
            y_test,
            pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            pred,
            zero_division=0
        ),
        "F1_Score": f1_score(
            y_test,
            pred,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_test,
            prob
        ),
        "PR_AUC": average_precision_score(
            y_test,
            prob
        )
    })

    trained_models[name] = model

    model_outputs[name] = {
        "pred": pred,
        "prob": prob
    }

comparison = (
    pd.DataFrame(results)
    .sort_values(
        [
            "PR_AUC",
            "F1_Score",
            "ROC_AUC"
        ],
        ascending=False
    )
    .reset_index(drop=True)
)

save_csv(
    comparison,
    "15_model_comparison.csv"
)

print("\nMODEL COMPARISON")
print(
    comparison.to_string(
        index=False
    )
)

plt.figure(figsize=(10, 6))

plt.bar(
    comparison["Model"],
    comparison["PR_AUC"]
)

plt.title(
    "Model Comparison - PR-AUC"
)

plt.xlabel("Model")
plt.ylabel("PR-AUC")
plt.ylim(0, 1)

plt.xticks(
    rotation=15
)

save_plot(
    "05_model_comparison.png"
)


# ============================================================
# 13. BEST MODEL
# ============================================================

best_name = comparison.iloc[
    0
]["Model"]

best_model = trained_models[
    best_name
]

y_pred = model_outputs[
    best_name
]["pred"]

y_prob = model_outputs[
    best_name
]["prob"
]

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

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)

print("\n" + "=" * 85)
print("BEST MODEL")
print("=" * 85)

print("Model    :", best_name)
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))
print("PR-AUC   :", round(pr_auc, 4))


# ============================================================
# 14. METRICS
# ============================================================

save_csv(
    pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "ROC_AUC",
            "PR_AUC"
        ],
        "Score": [
            accuracy,
            precision,
            recall,
            f1,
            roc_auc,
            pr_auc
        ]
    }),
    "16_final_metrics.csv"
)

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "One-Time",
        "Repeat"
    ],
    output_dict=True,
    zero_division=0
)

save_csv(
    pd.DataFrame(report)
    .T
    .reset_index()
    .rename(
        columns={"index": "Class"}
    ),
    "17_classification_report.csv"
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title(
    f"Confusion Matrix - {best_name}"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["One-Time", "Repeat"]
)

plt.yticks(
    [0, 1],
    ["One-Time", "Repeat"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

save_plot(
    "02_confusion_matrix.png"
)


# ============================================================
# 16. ROC
# ============================================================

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.title(
    f"ROC Curve - {best_name}"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

save_plot(
    "03_roc_curve.png"
)


# ============================================================
# 17. PRECISION-RECALL
# ============================================================

p, r, _ = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    r,
    p,
    label=f"PR-AUC = {pr_auc:.3f}"
)

plt.title(
    f"Precision-Recall Curve - {best_name}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

save_plot(
    "04_precision_recall_curve.png"
)


# ============================================================
# 18. FEATURE IMPORTANCE
# ============================================================

classifier = best_model.named_steps[
    "model"
]

if hasattr(
    classifier,
    "feature_importances_"
):

    importance_values = (
        classifier.feature_importances_
    )

elif hasattr(
    classifier,
    "coef_"
):

    importance_values = np.abs(
        classifier.coef_[0]
    )

else:

    importance_values = np.zeros(
        len(final_features)
    )

importance = pd.DataFrame({
    "Feature": final_features,
    "Importance": importance_values
}).sort_values(
    "Importance",
    ascending=False
)

save_csv(
    importance,
    "18_feature_importance.csv"
)

top = (
    importance
    .head(15)
    .sort_values("Importance")
)

plt.figure(figsize=(10, 7))

plt.barh(
    top["Feature"],
    top["Importance"]
)

plt.title(
    f"Feature Importance - {best_name}"
)

plt.xlabel("Importance")

save_plot(
    "06_feature_importance.png"
)


# ============================================================
# 19. PROBABILITY DISTRIBUTION
# ============================================================

actual = (
    y_test
    .reset_index(drop=True)
    .to_numpy()
)

plt.figure(figsize=(9, 6))

plt.hist(
    y_prob[actual == 0],
    bins=30,
    alpha=0.7,
    label="Actual One-Time"
)

plt.hist(
    y_prob[actual == 1],
    bins=30,
    alpha=0.7,
    label="Actual Repeat"
)

plt.title(
    "Predicted Repeat-Customer Probability"
)

plt.xlabel(
    "Repeat Probability"
)

plt.ylabel(
    "Frequency"
)

plt.legend()

save_plot(
    "07_probability_distribution.png"
)


# ============================================================
# 20. PREDICTIONS
# ============================================================

prediction_output = (
    X_test
    .reset_index(drop=True)
    .copy()
)

prediction_output["Actual"] = (
    y_test
    .reset_index(drop=True)
)

prediction_output["Predicted"] = (
    y_pred
)

prediction_output[
    "Repeat_Probability"
] = y_prob

save_csv(
    prediction_output,
    "19_predictions.csv"
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

skipped = int(
    (
        final_decisions_df["Decision"] == "SKIP"
    ).sum()
)

summary = pd.DataFrame({
    "Item": [

        "Dataset Rows",
        "Dataset Columns",

        "Customer Column",
        "Order Column",
        "Target",

        "Candidate Features",
        "Applied Features",
        "Skipped Features",

        "Training Rows",
        "Testing Rows",

        "Ratio Before",
        "Ratio After",

        "Balancing Method",

        "Customer Leakage",
        "Test Set Balanced",

        "Best Model",

        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC"
    ],

    "Value": [

        len(df),
        len(df.columns),

        customer_col,
        order_col,
        TARGET,

        len(candidate_features),
        len(final_features),
        skipped,

        len(X_train),
        len(X_test),

        ratio(
            before_one_time,
            before_repeat
        ),

        ratio(
            after_one_time,
            after_repeat
        ),

        method_name,

        "NO",
        "NO",

        best_name,

        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc
    ]
})

save_csv(
    summary,
    "20_ML_project_summary.csv"
)


# ============================================================
# 22. README
# ============================================================

readme = f"""
# Olist Final Expert ML

## Objective
Predict One-Time vs Repeat Customers.

## Target
repeat_customer = 1 when a customer has more than one DISTINCT order.

## Feature audit
Candidate features: {len(candidate_features)}
Applied features: {len(final_features)}
Skipped features: {skipped}

See:
- 02_feature_audit.csv
- 12_final_feature_decisions.csv
- 13_features_applied.csv

## Imbalance
Training ratio BEFORE:
{ratio(before_one_time, before_repeat)}

Balancing:
{method_name}

Training ratio AFTER:
{ratio(after_one_time, after_repeat)}

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
{best_name}

## Metrics
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1: {f1:.4f}
ROC-AUC: {roc_auc:.4f}
PR-AUC: {pr_auc:.4f}

## Folders
tables/
plots/
"""

(OUTPUT_DIR / "README.md").write_text(
    readme,
    encoding="utf-8"
)


# ============================================================
# 23. FINAL CONSOLE OUTPUT
# ============================================================

print("\n" + "=" * 90)
print("FINAL ML REPORT")
print("=" * 90)

print("\nFEATURES")
print("Candidate:", len(candidate_features))
print("Applied  :", len(final_features))
print("Skipped  :", skipped)

print("\nIMBALANCE")
print(
    "BEFORE:",
    ratio(
        before_one_time,
        before_repeat
    )
)
print(
    "METHOD:",
    method_name
)
print(
    "AFTER :",
    ratio(
        after_one_time,
        after_repeat
    )
)
print("TEST CHANGED: NO")

print("\nBEST MODEL")
print("Model   :", best_name)
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall  :", round(recall, 4))
print("F1      :", round(f1, 4))
print("ROC-AUC :", round(roc_auc, 4))
print("PR-AUC  :", round(pr_auc, 4))

print("\nOUTPUT")
print("Tables:", TABLE_DIR)
print("Plots :", PLOT_DIR)
print("README:", OUTPUT_DIR / "README.md")

print("\n" + "=" * 90)
print("MACHINE LEARNING ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 90)
