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
# ============================================================
# ADVANCED / BUSINESS EDA
# ============================================================

print("\n" + "=" * 60)
print("ADVANCED BUSINESS EDA")
print("=" * 60)

# ---------- DATE ANALYSIS ----------
date_candidates = [
    "order_purchase_timestamp",
    "order_date",
    "purchase_date",
    "date"
]

date_col = None

for col in date_candidates:
    if col in df.columns:
        date_col = col
        break

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df["purchase_year"] = df[date_col].dt.year
    df["purchase_month"] = df[date_col].dt.month
    df["purchase_month_name"] = df[date_col].dt.strftime("%b")
    df["purchase_year_month"] = df[date_col].dt.to_period("M").astype(str)

    monthly = df.groupby("purchase_year_month").agg(
        Orders=("order_id", "nunique"),
        Customers=("customer_id", "nunique")
    ).reset_index()

    if "order_item_value" in df.columns:
        revenue_monthly = df.groupby("purchase_year_month")[
            "order_item_value"
        ].sum().reset_index(name="Revenue")

        monthly = monthly.merge(
            revenue_monthly,
            on="purchase_year_month",
            how="left"
        )

    monthly.to_csv("advanced_monthly_business_analysis.csv", index=False)

    # Monthly orders
    plt.figure(figsize=(12, 5))
    plt.plot(monthly["purchase_year_month"], monthly["Orders"], marker="o")
    plt.title("Monthly Orders Trend")
    plt.xlabel("Month")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("monthly_orders_trend.png")
    plt.close()

    # Monthly revenue
    if "Revenue" in monthly.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(
            monthly["purchase_year_month"],
            monthly["Revenue"],
            marker="o"
        )
        plt.title("Monthly Revenue Trend")
        plt.xlabel("Month")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("monthly_revenue_trend.png")
        plt.close()


# ---------- CUSTOMER ANALYSIS ----------

if "customer_id" in df.columns and "order_id" in df.columns:

    customer_orders = df.groupby("customer_id").agg(
        Total_Orders=("order_id", "nunique")
    ).reset_index()

    customer_orders["Customer_Type"] = np.where(
        customer_orders["Total_Orders"] > 1,
        "Repeat Customer",
        "One-Time Customer"
    )

    customer_summary = customer_orders["Customer_Type"].value_counts(
    ).reset_index()

    customer_summary.columns = [
        "Customer_Type",
        "Customer_Count"
    ]

    customer_summary.to_csv(
        "advanced_customer_type_analysis.csv",
        index=False
    )

    print("\n--- CUSTOMER TYPE ANALYSIS ---")
    print(customer_summary)

    plt.figure(figsize=(7, 5))
    plt.bar(
        customer_summary["Customer_Type"],
        customer_summary["Customer_Count"]
    )
    plt.title("Repeat vs One-Time Customers")
    plt.xlabel("Customer Type")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig("advanced_repeat_customer_analysis.png")
    plt.close()


# ---------- ORDER VALUE ANALYSIS ----------

if "order_item_value" in df.columns:

    order_value = df.groupby("order_id").agg(
        Order_Value=("order_item_value", "sum")
    ).reset_index()

    order_value["Order_Value_Category"] = pd.cut(
        order_value["Order_Value"],
        bins=[-np.inf, 100, 250, 500, 1000, np.inf],
        labels=[
            "Under 100",
            "100-250",
            "250-500",
            "500-1000",
            "1000+"
        ]
    )

    order_value.to_csv(
        "order_value_analysis.csv",
        index=False
    )

    print("\n--- ORDER VALUE SUMMARY ---")
    print(order_value["Order_Value"].describe())

    plt.figure(figsize=(9, 5))
    plt.hist(
        order_value["Order_Value"].dropna(),
        bins=40
    )
    plt.title("Order Value Distribution")
    plt.xlabel("Order Value")
    plt.ylabel("Number of Orders")
    plt.tight_layout()
    plt.savefig("order_value_distribution.png")
    plt.close()


# ---------- FREIGHT ANALYSIS ----------

if "total_freight_value" in df.columns:

    freight_summary = df["total_freight_value"].describe()

    freight_summary.to_csv(
        "freight_value_summary.csv"
    )

    print("\n--- FREIGHT VALUE SUMMARY ---")
    print(freight_summary)


# ---------- COHORT ANALYSIS ----------

if date_col and "customer_id" in df.columns:

    cohort_data = df[[
        "customer_id",
        date_col
    ]].dropna().copy()

    first_purchase = cohort_data.groupby(
        "customer_id"
    )[date_col].min().reset_index()

    first_purchase.columns = [
        "customer_id",
        "first_purchase_date"
    ]

    cohort_data = cohort_data.merge(
        first_purchase,
        on="customer_id",
        how="left"
    )

    cohort_data["cohort_month"] = (
        cohort_data["first_purchase_date"]
        .dt.to_period("M")
    )

    cohort_data["purchase_month"] = (
        cohort_data[date_col]
        .dt.to_period("M")
    )

    cohort_data["cohort_index"] = (
        (cohort_data["purchase_month"].dt.year -
         cohort_data["cohort_month"].dt.year) * 12
        +
        (cohort_data["purchase_month"].dt.month -
         cohort_data["cohort_month"].dt.month)
        + 1
    )

    cohort_table = cohort_data.groupby(
        ["cohort_month", "cohort_index"]
    )["customer_id"].nunique().reset_index()

    cohort_table.columns = [
        "Cohort_Month",
        "Cohort_Index",
        "Customers"
    ]

    cohort_table.to_csv(
        "advanced_cohort_analysis.csv",
        index=False
    )

    print("\n--- COHORT ANALYSIS ---")
    print(cohort_table.head(20))


# ---------- FINAL BUSINESS INSIGHTS ----------

print("\n" + "=" * 60)
print("BUSINESS EDA SUMMARY")
print("=" * 60)

print("\nTotal Rows:", len(df))
print("Total Columns:", len(df.columns))

if "order_id" in df.columns:
    print("Unique Orders:", df["order_id"].nunique())

if "customer_id" in df.columns:
    print("Unique Customers:", df["customer_id"].nunique())

if "order_item_value" in df.columns:
    print(
        "Total Revenue:",
        round(df["order_item_value"].sum(), 2)
    )
    print(
        "Average Item Value:",
        round(df["order_item_value"].mean(), 2)
    )

if "total_freight_value" in df.columns:
    print(
        "Total Freight:",
        round(df["total_freight_value"].sum(), 2)
    )

print("\nADVANCED EDA COMPLETED SUCCESSFULLY!")

print("\nNew files created:")
print("- advanced_monthly_business_analysis.csv")
print("- monthly_orders_trend.png")
print("- monthly_revenue_trend.png")
print("- advanced_customer_type_analysis.csv")
print("- advanced_repeat_customer_analysis.png")
print("- order_value_analysis.csv")
print("- order_value_distribution.png")
print("- freight_value_summary.csv")
print("- advanced_cohort_analysis.csv")