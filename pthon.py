import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("olist_cohort_analysis_dataset.csv")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv("olist_cohort_analysis_dataset.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# ==============================
# 2. BASIC INFORMATION
# ==============================
print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- COLUMNS ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- DATASET INFO ---")
df.info()

# ==============================
# 3. MISSING VALUES
# ==============================
missing = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Missing Percentage": (
        df.isnull().sum() / len(df) * 100
    ).round(2)
})

print("\n--- MISSING VALUES ---")
print(missing)

missing.to_csv("missing_value_analysis.csv")

# ==============================
# 4. DUPLICATES
# ==============================
duplicates = df.duplicated().sum()

print("\nDuplicate rows:", duplicates)

df = df.drop_duplicates()

# ==============================
# 5. UNIQUE VALUES
# ==============================
unique = pd.DataFrame({
    "Column": df.columns,
    "Unique Values": [
        df[col].nunique() for col in df.columns
    ]
})

print("\n--- UNIQUE VALUES ---")
print(unique)

unique.to_csv("unique_value_analysis.csv", index=False)

# ==============================
# 6. STATISTICS
# ==============================
print("\n--- STATISTICAL SUMMARY ---")
print(df.describe(include="all"))

# ==============================
# 7. NUMERICAL COLUMNS
# ==============================
numeric = df.select_dtypes(
    include=np.number
)

print("\n--- NUMERICAL COLUMNS ---")
print(numeric.columns.tolist())

# ==============================
# 8. CORRELATION ANALYSIS
# ==============================
if numeric.shape[1] >= 2:

    correlation = numeric.corr()

    print("\n--- CORRELATION MATRIX ---")
    print(correlation)

    # High correlations
    pairs = []

    for i in range(len(correlation.columns)):
        for j in range(i + 1, len(correlation.columns)):

            value = correlation.iloc[i, j]

            if abs(value) >= 0.70:
                pairs.append({
                    "Feature 1": correlation.columns[i],
                    "Feature 2": correlation.columns[j],
                    "Correlation": round(value, 3)
                })

    high_corr = pd.DataFrame(pairs)

    high_corr.to_csv(
        "high_correlation_pairs.csv",
        index=False
    )

    # Heatmap
    plt.figure(figsize=(12, 8))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()

    plt.savefig("correlation_heatmap.png")
    plt.close()

# ==============================
# 9. OUTLIER ANALYSIS
# ==============================
outlier_results = []

for col in numeric.columns:

    Q1 = numeric[col].quantile(0.25)
    Q3 = numeric[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = (
        (numeric[col] < lower) |
        (numeric[col] > upper)
    ).sum()

    outlier_results.append({
        "Column": col,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Outlier Count": count
    })

outlier_df = pd.DataFrame(outlier_results)

print("\n--- OUTLIER ANALYSIS ---")
print(outlier_df)

outlier_df.to_csv(
    "outlier_analysis.csv",
    index=False
)

# ==============================
# 10. DISTRIBUTIONS
# ==============================
for col in numeric.columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[col].dropna(),
        kde=True
    )

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        f"distribution_{col}.png"
    )

    plt.close()

# ==============================
# 11. CATEGORICAL ANALYSIS
# ==============================
categorical = df.select_dtypes(
    include="object"
).columns

for col in categorical:

    print(f"\n--- TOP VALUES: {col} ---")
    print(df[col].value_counts().head(10))

# ==============================
# 12. SAVE CLEAN DATA
# ==============================
df.to_csv(
    "cleaned_olist_cohort_dataset.csv",
    index=False
)

# ==============================
# FINAL
# ==============================
print("\n" + "=" * 60)
print("TASK ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("Final shape:", df.shape)

print("\nOutput files created:")
print("- missing_value_analysis.csv")
print("- unique_value_analysis.csv")
print("- high_correlation_pairs.csv")
print("- outlier_analysis.csv")
print("- correlation_heatmap.png")
print("- distribution plots")
print("- cleaned_olist_cohort_dataset.csv")
