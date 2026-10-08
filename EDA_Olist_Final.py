
# ============================================================
# OLIST E-COMMERCE PROJECT - FINAL EXPERT EDA
# ============================================================
# Purpose:
#   1) Perform complete EDA
#   2) Analyze missing values, duplicates, uniqueness
#   3) Analyze skewness, kurtosis and outliers
#   4) Analyze correlations and dependence
#   5) Analyze monthly business trends
#   6) Analyze one-time vs repeat customers
#   7) Perform customer cohort retention analysis
#   8) Save ALL tables and plots in EDA_Results
#
# Input:
#   olist_cohort_analysis_dataset.csv
#
# Run:
#   python EDA_Olist_Final.py
# ============================================================

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -----------------------------
# SETTINGS
# -----------------------------
DATA_FILE = "olist_cohort_analysis_dataset.csv"
OUTPUT_DIR = Path("EDA_Results")
TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"
DATA_DIR = OUTPUT_DIR / "data"

for folder in [TABLE_DIR, PLOT_DIR, DATA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)
RANDOM_STATE = 42


# -----------------------------
# HELPERS
# -----------------------------
def safe_filename(name):
    """Convert a column name into a Windows-safe filename."""
    name = re.sub(r"[^\w.-]+", "_", str(name).strip())
    return name[:120]


def save_csv(df_obj, filename, folder=TABLE_DIR, index=False):
    path = folder / filename
    df_obj.to_csv(path, index=index)
    return path


def save_text(text, filename, folder=OUTPUT_DIR):
    path = folder / filename
    path.write_text(text, encoding="utf-8")
    return path


def find_first_column(frame, candidates):
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def print_section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def safe_numeric_series(series):
    return pd.to_numeric(series, errors="coerce")


# -----------------------------
# 1. LOAD DATA
# -----------------------------
print_section("OLIST FINAL EXPERT EDA")

print("[1/12] Loading dataset...")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Place the CSV in the same folder as this Python file."
    )

df_raw = pd.read_csv(DATA_FILE, low_memory=False)
df = df_raw.copy()

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# -----------------------------
# 2. BASIC DATA AUDIT
# -----------------------------
print_section("[2/12] BASIC DATA AUDIT")

basic_summary = pd.DataFrame({
    "Metric": [
        "Rows",
        "Columns",
        "Duplicate Rows",
        "Numeric Columns",
        "Categorical Columns",
        "Date-like Columns"
    ],
    "Value": [
        len(df),
        len(df.columns),
        int(df.duplicated().sum()),
        len(df.select_dtypes(include=np.number).columns),
        len(df.select_dtypes(include=["object", "category"]).columns),
        0
    ]
})

print("\nShape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

save_csv(basic_summary, "01_basic_summary.csv")
save_csv(
    pd.DataFrame({"Column": df.columns, "Data_Type": df.dtypes.astype(str).values}),
    "02_column_data_types.csv"
)

# Duplicate audit
duplicate_count = int(df.duplicated().sum())
duplicate_rate = (duplicate_count / len(df) * 100) if len(df) else 0

duplicate_summary = pd.DataFrame({
    "Metric": ["Duplicate Rows", "Duplicate Percentage"],
    "Value": [duplicate_count, round(duplicate_rate, 4)]
})
save_csv(duplicate_summary, "03_duplicate_analysis.csv")

if duplicate_count > 0:
    print(f"\nDuplicate rows found: {duplicate_count}")
    df = df.drop_duplicates().reset_index(drop=True)
    print("Duplicates removed.")
else:
    print("\nDuplicate rows: 0")


# -----------------------------
# 3. MISSING + UNIQUE VALUES
# -----------------------------
print_section("[3/12] MISSING VALUES + UNIQUE VALUES")

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": df.isna().sum().values,
    "Missing_Percentage": (df.isna().mean() * 100).round(4).values
}).sort_values(["Missing_Percentage", "Missing_Count"], ascending=False)

print("\nMissing value analysis:")
print(missing_df.to_string(index=False))
save_csv(missing_df, "04_missing_value_analysis.csv")

unique_df = pd.DataFrame({
    "Column": df.columns,
    "Unique_Count": [df[col].nunique(dropna=True) for col in df.columns],
    "Unique_Percentage": [
        round(df[col].nunique(dropna=True) / len(df) * 100, 4) if len(df) else 0
        for col in df.columns
    ]
}).sort_values("Unique_Count", ascending=False)

save_csv(unique_df, "05_unique_value_analysis.csv")


# -----------------------------
# 4. DATA TYPES / DATE DETECTION
# -----------------------------
print_section("[4/12] DATE AND COLUMN TYPE DETECTION")

date_candidates = [
    "order_purchase_timestamp",
    "order_date",
    "purchase_date",
    "date",
    "purchase_datetime"
]

date_col = find_first_column(df, date_candidates)

if date_col is not None:
    parsed_date = pd.to_datetime(df[date_col], errors="coerce")
    valid_date_rate = parsed_date.notna().mean() * 100
    if valid_date_rate >= 50:
        df[date_col] = parsed_date
        date_status = "Detected and parsed"
        date_valid_rate = valid_date_rate
    else:
        date_status = "Candidate found but parsing quality < 50%"
        date_valid_rate = valid_date_rate
        date_col = None
else:
    date_status = "No expected date column found"
    date_valid_rate = 0

date_audit = pd.DataFrame({
    "Item": ["Date Column", "Date Status", "Valid Date Percentage"],
    "Value": [str(date_col), date_status, round(date_valid_rate, 2)]
})
save_csv(date_audit, "06_date_detection.csv")

print("Date column:", date_col)
print("Status:", date_status)


# -----------------------------
# 5. STATISTICAL SUMMARY
# -----------------------------
print_section("[5/12] STATISTICAL SUMMARY")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

if numeric_cols:
    numeric_description = df[numeric_cols].describe().T.reset_index()
    numeric_description = numeric_description.rename(columns={"index": "Column"})
    save_csv(numeric_description, "07_numeric_descriptive_statistics.csv")

if categorical_cols:
    categorical_summary = []
    for col in categorical_cols:
        categorical_summary.append({
            "Column": col,
            "Unique_Values": df[col].nunique(dropna=True),
            "Missing_Count": df[col].isna().sum(),
            "Top_Value": df[col].mode(dropna=True).iloc[0] if not df[col].mode(dropna=True).empty else np.nan,
            "Top_Value_Count": df[col].value_counts(dropna=True).iloc[0] if df[col].value_counts(dropna=True).size else 0
        })
    categorical_summary = pd.DataFrame(categorical_summary)
    save_csv(categorical_summary, "08_categorical_summary.csv")

print("Numeric columns:", len(numeric_cols))
print("Categorical columns:", len(categorical_cols))


# -----------------------------
# 6. SKEWNESS + KURTOSIS
# -----------------------------
print_section("[6/12] SKEWNESS + KURTOSIS")

skew_rows = []

for col in numeric_cols:
    series = safe_numeric_series(df[col]).dropna()

    if len(series) == 0:
        skew_value = np.nan
        kurt_value = np.nan
        skew_class = "No valid numeric observations"
    else:
        skew_value = series.skew()
        kurt_value = series.kurtosis()

        if pd.isna(skew_value):
            skew_class = "Undefined"
        elif skew_value > 1:
            skew_class = "Highly Right-Skewed"
        elif skew_value > 0.5:
            skew_class = "Moderately Right-Skewed"
        elif skew_value < -1:
            skew_class = "Highly Left-Skewed"
        elif skew_value < -0.5:
            skew_class = "Moderately Left-Skewed"
        else:
            skew_class = "Approximately Symmetric"

    skew_rows.append({
        "Column": col,
        "Skewness": skew_value,
        "Kurtosis": kurt_value,
        "Distribution_Assessment": skew_class
    })

skewness_df = pd.DataFrame(skew_rows)
if not skewness_df.empty:
    skewness_df = skewness_df.sort_values(
        "Skewness", key=lambda s: s.abs(), ascending=False
    )

print(skewness_df.to_string(index=False))
save_csv(skewness_df, "09_skewness_kurtosis_analysis.csv")

# Skewness plot
if not skewness_df.empty:
    plot_df = skewness_df.dropna(subset=["Skewness"]).copy()
    plt.figure(figsize=(11, 6))
    plt.barh(plot_df["Column"].astype(str), plot_df["Skewness"])
    plt.axvline(0, linewidth=1)
    plt.title("Skewness Analysis of Numerical Features")
    plt.xlabel("Skewness")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_skewness_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# 7. OUTLIER ANALYSIS
# -----------------------------
print_section("[7/12] OUTLIER ANALYSIS")

outlier_rows = []

for col in numeric_cols:
    series = safe_numeric_series(df[col])
    valid = series.dropna()

    if valid.empty:
        q1 = q3 = iqr = lower = upper = np.nan
        count = 0
        pct = 0
    else:
        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        pct = count / len(valid) * 100

    outlier_rows.append({
        "Column": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower_Bound": lower,
        "Upper_Bound": upper,
        "Outlier_Count": count,
        "Outlier_Percentage": pct
    })

outlier_df = pd.DataFrame(outlier_rows)
if not outlier_df.empty:
    outlier_df = outlier_df.sort_values("Outlier_Percentage", ascending=False)

print(outlier_df.to_string(index=False))
save_csv(outlier_df, "10_outlier_analysis.csv")

# Combined boxplot
if numeric_cols:
    plot_cols = numeric_cols[:30]
    plt.figure(figsize=(14, 8))
    plt.boxplot(
        [safe_numeric_series(df[c]).dropna().values for c in plot_cols],
        labels=[str(c)[:22] for c in plot_cols],
        vert=False
    )
    plt.title("Numerical Feature Boxplots")
    plt.xlabel("Value")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_numerical_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# 8. CORRELATION / DEPENDENCE
# -----------------------------
print_section("[8/12] CORRELATION + DEPENDENCE ANALYSIS")

if len(numeric_cols) >= 2:
    correlation = df[numeric_cols].corr(numeric_only=True)

    corr_out = correlation.reset_index().rename(columns={"index": "Feature"})
    save_csv(corr_out, "11_correlation_matrix.csv")

    high_corr_rows = []
    for i, col1 in enumerate(correlation.columns):
        for j in range(i + 1, len(correlation.columns)):
            col2 = correlation.columns[j]
            value = correlation.iloc[i, j]
            if pd.notna(value) and abs(value) >= 0.70:
                high_corr_rows.append({
                    "Feature_1": col1,
                    "Feature_2": col2,
                    "Correlation": value,
                    "Absolute_Correlation": abs(value),
                    "Interpretation": (
                        "Strong Positive Relationship" if value > 0
                        else "Strong Negative Relationship"
                    )
                })

    high_corr_df = pd.DataFrame(high_corr_rows)
    if not high_corr_df.empty:
        high_corr_df = high_corr_df.sort_values("Absolute_Correlation", ascending=False)

    save_csv(high_corr_df, "12_high_correlation_pairs.csv")

    # Heatmap using matplotlib only
    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(correlation.values, aspect="auto")
    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=90)
    ax.set_yticklabels(correlation.columns)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nHigh correlation pairs (|r| >= 0.70):")
    if high_corr_df.empty:
        print("No strong correlation pairs found.")
    else:
        print(high_corr_df.to_string(index=False))
else:
    save_csv(pd.DataFrame(), "12_high_correlation_pairs.csv")


# -----------------------------
# 9. DISTRIBUTION PLOTS
# -----------------------------
print_section("[9/12] DISTRIBUTION PLOTS")

for col in numeric_cols:
    values = safe_numeric_series(df[col]).dropna()
    if values.empty:
        continue

    # Histogram
    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=40)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / f"distribution_{safe_filename(col)}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Boxplot
    plt.figure(figsize=(9, 4))
    plt.boxplot(values, vert=False)
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / f"boxplot_{safe_filename(col)}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# -----------------------------
# 10. BUSINESS DATE ANALYSIS
# -----------------------------
print_section("[10/12] BUSINESS TREND ANALYSIS")

order_col = find_first_column(
    df,
    ["order_id", "order_identifier", "order_number"]
)

customer_col = find_first_column(
    df,
    ["customer_unique_id", "customer_id", "customer_unique", "customer_identifier"]
)

value_col = find_first_column(
    df,
    ["order_item_value", "order_value", "total_order_value", "payment_value"]
)

freight_col = find_first_column(
    df,
    ["total_freight_value", "freight_value", "order_freight_value"]
)

monthly = pd.DataFrame()

if date_col is not None:
    work = df.copy()
    work["purchase_period"] = work[date_col].dt.to_period("M").astype(str)

    order_expr = (order_col, "nunique") if order_col else (date_col, "size")
    customer_expr = (customer_col, "nunique") if customer_col else (date_col, "size")

    agg_dict = {
        "Orders": order_expr,
        "Customers": customer_expr
    }

    if value_col:
        agg_dict["Revenue"] = (value_col, "sum")

    if freight_col:
        agg_dict["Freight"] = (freight_col, "sum")

    monthly = (
        work.groupby("purchase_period")
        .agg(**agg_dict)
        .reset_index()
    )

    save_csv(monthly, "13_monthly_business_analysis.csv")

    # Orders trend
    plt.figure(figsize=(13, 5))
    plt.plot(monthly["purchase_period"], monthly["Orders"], marker="o")
    plt.title("Monthly Orders Trend")
    plt.xlabel("Month")
    plt.ylabel("Orders")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_monthly_orders_trend.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Revenue trend
    if "Revenue" in monthly.columns:
        plt.figure(figsize=(13, 5))
        plt.plot(monthly["purchase_period"], monthly["Revenue"], marker="o")
        plt.title("Monthly Revenue Trend")
        plt.xlabel("Month")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "05_monthly_revenue_trend.png", dpi=300, bbox_inches="tight")
        plt.close()
else:
    save_csv(pd.DataFrame(), "13_monthly_business_analysis.csv")


# -----------------------------
# 11. CUSTOMER / ORDER ANALYSIS
# -----------------------------
print_section("[11/12] CUSTOMER + ORDER ANALYSIS")

customer_summary = pd.DataFrame()

if customer_col and order_col:
    customer_orders = (
        df.groupby(customer_col)[order_col]
        .nunique()
        .reset_index(name="Total_Orders")
    )

    customer_orders["Customer_Type"] = np.where(
        customer_orders["Total_Orders"] > 1,
        "Repeat Customer",
        "One-Time Customer"
    )

    counts = customer_orders["Customer_Type"].value_counts()
    total_customers = counts.sum()

    repeat_count = int(counts.get("Repeat Customer", 0))
    one_time_count = int(counts.get("One-Time Customer", 0))

    repeat_pct = repeat_count / total_customers * 100 if total_customers else 0
    one_time_pct = one_time_count / total_customers * 100 if total_customers else 0

    customer_summary = pd.DataFrame({
        "Customer_Type": ["One-Time Customer", "Repeat Customer"],
        "Customer_Count": [one_time_count, repeat_count],
        "Percentage": [one_time_pct, repeat_pct]
    })

    save_csv(customer_summary, "14_repeat_vs_one_time_customers.csv")
    save_csv(customer_orders, "15_customer_order_frequency.csv")

    print("\nCUSTOMER TYPE DISTRIBUTION")
    print(customer_summary.to_string(index=False))

    ratio = (
        one_time_count / repeat_count
        if repeat_count > 0
        else np.inf
    )

    ratio_text = "Undefined (no repeat customers)" if not np.isfinite(ratio) else f"{ratio:.2f}:1"
    ratio_summary = pd.DataFrame({
        "Metric": [
            "One-Time Customers",
            "Repeat Customers",
            "One-Time Percentage",
            "Repeat Percentage",
            "One-Time : Repeat Ratio"
        ],
        "Value": [
            one_time_count,
            repeat_count,
            round(one_time_pct, 4),
            round(repeat_pct, 4),
            ratio_text
        ]
    })
    save_csv(ratio_summary, "16_customer_imbalance_summary.csv")

    plt.figure(figsize=(8, 5))
    plt.bar(customer_summary["Customer_Type"], customer_summary["Customer_Count"])
    plt.title("One-Time vs Repeat Customers")
    plt.xlabel("Customer Type")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "06_repeat_vs_one_time_customers.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Order value
    if value_col:
        order_value = (
            df.groupby(order_col)[value_col]
            .sum()
            .reset_index(name="Order_Value")
        )

        bins = [-np.inf, 100, 250, 500, 1000, np.inf]
        labels = ["Under 100", "100-250", "250-500", "500-1000", "1000+"]

        order_value["Order_Value_Category"] = pd.cut(
            order_value["Order_Value"],
            bins=bins,
            labels=labels
        )

        save_csv(order_value, "17_order_value_analysis.csv")

        print("\nOrder value statistics:")
        print(order_value["Order_Value"].describe())

        plt.figure(figsize=(10, 5))
        plt.hist(order_value["Order_Value"].dropna(), bins=40)
        plt.title("Order Value Distribution")
        plt.xlabel("Order Value")
        plt.ylabel("Number of Orders")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "07_order_value_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

if freight_col:
    freight_summary = df[freight_col].describe().to_frame(name="Value").reset_index()
    freight_summary = freight_summary.rename(columns={"index": "Statistic"})
    save_csv(freight_summary, "18_freight_value_summary.csv")


# -----------------------------
# 12. COHORT RETENTION
# -----------------------------
print_section("[12/12] CUSTOMER COHORT RETENTION ANALYSIS")

cohort_count = pd.DataFrame()
cohort_retention = pd.DataFrame()

if date_col and customer_col:
    cohort_data = df[[customer_col, date_col]].dropna().copy()

    # First purchase month for each customer
    first_purchase = (
        cohort_data.groupby(customer_col)[date_col]
        .min()
        .reset_index(name="first_purchase_date")
    )

    cohort_data = cohort_data.merge(first_purchase, on=customer_col, how="left")

    cohort_data["cohort_month"] = cohort_data["first_purchase_date"].dt.to_period("M")
    cohort_data["purchase_month"] = cohort_data[date_col].dt.to_period("M")

    cohort_data["cohort_index"] = (
        (cohort_data["purchase_month"].dt.year - cohort_data["cohort_month"].dt.year) * 12
        + (cohort_data["purchase_month"].dt.month - cohort_data["cohort_month"].dt.month)
        + 1
    )

    # Distinct customers in each cohort/month
    cohort_count = (
        cohort_data.groupby(["cohort_month", "cohort_index"])[customer_col]
        .nunique()
        .reset_index(name="Customers")
    )

    cohort_count["Cohort_Month"] = cohort_count["cohort_month"].astype(str)

    # Cohort size
    cohort_size = (
        cohort_data.groupby("cohort_month")[customer_col]
        .nunique()
        .rename("Cohort_Size")
    )

    cohort_count = cohort_count.merge(
        cohort_size,
        left_on="cohort_month",
        right_index=True,
        how="left"
    )

    cohort_count["Retention_Percentage"] = (
        cohort_count["Customers"] / cohort_count["Cohort_Size"] * 100
    )

    save_csv(
        cohort_count[
            [
                "Cohort_Month",
                "cohort_index",
                "Customers",
                "Cohort_Size",
                "Retention_Percentage"
            ]
        ],
        "19_cohort_retention_long_format.csv"
    )

    retention_matrix = cohort_count.pivot_table(
        index="Cohort_Month",
        columns="cohort_index",
        values="Retention_Percentage",
        aggfunc="first"
    )

    cohort_retention = retention_matrix.reset_index()
    save_csv(cohort_retention, "20_cohort_retention_matrix.csv")

    # Cohort retention heatmap
    if not retention_matrix.empty:
        fig, ax = plt.subplots(
            figsize=(
                max(12, retention_matrix.shape[1] * 0.65),
                max(7, retention_matrix.shape[0] * 0.45)
            )
        )

        matrix_values = retention_matrix.values.astype(float)
        im = ax.imshow(matrix_values, aspect="auto", vmin=0, vmax=100)

        ax.set_xticks(range(len(retention_matrix.columns)))
        ax.set_xticklabels(retention_matrix.columns)
        ax.set_yticks(range(len(retention_matrix.index)))
        ax.set_yticklabels(retention_matrix.index)

        ax.set_xlabel("Cohort Month Index")
        ax.set_ylabel("Acquisition Cohort")
        ax.set_title("Customer Cohort Retention (%)")

        fig.colorbar(im, ax=ax, label="Retention %")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "08_cohort_retention_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Cohort counts
    first_month_customers = (
        cohort_data.groupby("cohort_month")[customer_col]
        .nunique()
        .reset_index(name="New_Customers")
    )
    first_month_customers["Cohort_Month"] = first_month_customers["cohort_month"].astype(str)
    save_csv(first_month_customers[["Cohort_Month", "New_Customers"]], "21_new_customer_cohorts.csv")

else:
    print("Cohort analysis skipped: required date/customer column not found.")


# -----------------------------
# FINAL SUMMARY / README
# -----------------------------
print_section("FINAL EDA SUMMARY")

summary_rows = [
    ("Original Rows", len(df_raw)),
    ("Final Rows After Duplicate Removal", len(df)),
    ("Original Columns", len(df_raw.columns)),
    ("Final Columns", len(df.columns)),
    ("Duplicate Rows Removed", len(df_raw) - len(df)),
    ("Numeric Columns", len(numeric_cols)),
    ("Categorical Columns", len(categorical_cols)),
    ("Date Column Used", str(date_col)),
    ("Customer Column Used", str(customer_col)),
    ("Order Column Used", str(order_col)),
    ("Value Column Used", str(value_col)),
    ("Freight Column Used", str(freight_col))
]

if not customer_summary.empty:
    repeat_row = customer_summary.loc[
        customer_summary["Customer_Type"] == "Repeat Customer"
    ]
    one_time_row = customer_summary.loc[
        customer_summary["Customer_Type"] == "One-Time Customer"
    ]
    if not repeat_row.empty:
        summary_rows.append(("Repeat Customers", int(repeat_row["Customer_Count"].iloc[0])))
        summary_rows.append(("Repeat Customer %", float(repeat_row["Percentage"].iloc[0])))
    if not one_time_row.empty:
        summary_rows.append(("One-Time Customers", int(one_time_row["Customer_Count"].iloc[0])))
        summary_rows.append(("One-Time Customer %", float(one_time_row["Percentage"].iloc[0])))

summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
save_csv(summary_df, "22_EDA_project_summary.csv")

readme = f"""# Olist EDA Results

## Dataset
- Input file: `{DATA_FILE}`
- Original rows: {len(df_raw)}
- Final rows after duplicate removal: {len(df)}
- Original columns: {len(df_raw.columns)}
- Final columns: {len(df.columns)}

## Analyses performed
- Data types and structure
- Missing values
- Duplicates
- Unique-value analysis
- Descriptive statistics
- Skewness and kurtosis
- Outlier detection using IQR
- Correlation and high-correlation pairs
- Numerical distribution and boxplots
- Monthly orders/revenue trends
- One-time vs repeat customer analysis
- Customer order-frequency analysis
- Order-value analysis
- Freight-value analysis
- Customer cohort retention analysis

## Important methodology notes
- Duplicate rows are removed only after the duplicate audit is saved.
- Missing values are reported; they are not blindly deleted.
- Outliers are identified using the IQR rule; they are not automatically deleted.
- Repeat customer status is based on distinct orders per customer, using the best available customer identifier.
- Cohort retention is calculated from distinct customers by acquisition month.
- All plots are saved under `EDA_Results/plots/`.
- All tables are saved under `EDA_Results/tables/`.
- Data outputs are saved under `EDA_Results/data/` when applicable.

## Output folders
- `EDA_Results/plots/`
- `EDA_Results/tables/`
- `EDA_Results/data/`
"""

save_text(readme, "README.md")

# Save final EDA dataset to data folder
df.to_csv(DATA_DIR / "cleaned_olist_cohort_analysis_dataset.csv", index=False)

print("\nEDA COMPLETED SUCCESSFULLY.")
print("All outputs saved in:", OUTPUT_DIR)
print("Plots folder:", PLOT_DIR)
print("Tables folder:", TABLE_DIR)
print("Data folder :", DATA_DIR)
