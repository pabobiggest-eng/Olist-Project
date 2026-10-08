# ============================================================
# OLIST E-COMMERCE PROJECT - FINAL EDA
# ============================================================

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# 1. SETTINGS + DATASET LOCATION
# ============================================================

DATA_FILENAME = "olist_cohort_analysis_dataset.csv"

SCRIPT_DIR = Path(__file__).resolve().parent
CURRENT_DIR = Path.cwd()


def find_dataset():
    """
    Search for the dataset automatically.
    This avoids errors when Python is run from another folder.
    """

    possible_locations = [
        CURRENT_DIR / DATA_FILENAME,
        SCRIPT_DIR / DATA_FILENAME,
        SCRIPT_DIR.parent / DATA_FILENAME
    ]

    for path in possible_locations:
        if path.is_file():
            return path

    # Recursive search
    search_roots = [
        SCRIPT_DIR,
        SCRIPT_DIR.parent
    ]

    checked = set()

    for root in search_roots:
        if not root.exists():
            continue

        for path in root.rglob(DATA_FILENAME):
            key = str(path.resolve()).lower()

            if key not in checked and path.is_file():
                checked.add(key)
                return path

    raise FileNotFoundError(
        f"\nDataset '{DATA_FILENAME}' was not found.\n"
        "Make sure the CSV exists inside your Olist project folder."
    )


DATA_FILE = find_dataset()

# Results will be saved beside the dataset
OUTPUT_DIR = DATA_FILE.parent / "EDA_Results"

TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"
DATA_DIR = OUTPUT_DIR / "data"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_filename(name):
    name = re.sub(r"[^\w.-]+", "_", str(name))
    return name[:120]


def save_csv(dataframe, filename, folder=TABLE_DIR):
    path = folder / filename
    dataframe.to_csv(path, index=False)
    return path


def save_text(text, filename, folder=OUTPUT_DIR):
    path = folder / filename
    path.write_text(text, encoding="utf-8")
    return path


def find_column(dataframe, candidates):

    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


def print_section(title):

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# 2. LOAD DATA
# ============================================================

print_section("OLIST FINAL EDA")

print("[1/12] Loading dataset...")

df_raw = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

df = df_raw.copy()

print("\nDataset loaded successfully.")
print("Dataset path:", DATA_FILE)
print("Results folder:", OUTPUT_DIR)
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 3. BASIC DATA INFORMATION
# ============================================================

print_section("[2/12] BASIC DATA INFORMATION")

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

basic_summary = pd.DataFrame({
    "Metric": [
        "Rows",
        "Columns",
        "Duplicate Rows",
        "Numeric Columns",
        "Categorical Columns"
    ],
    "Value": [
        len(df),
        len(df.columns),
        int(df.duplicated().sum()),
        len(df.select_dtypes(include=np.number).columns),
        len(df.select_dtypes(include="object").columns)
    ]
})

save_csv(
    basic_summary,
    "01_basic_summary.csv"
)


# ============================================================
# 4. DUPLICATE ANALYSIS
# ============================================================

print_section("[3/12] DUPLICATE ANALYSIS")

duplicate_count = int(df.duplicated().sum())

print("Duplicate rows:", duplicate_count)

duplicate_percentage = (
    duplicate_count / len(df) * 100
    if len(df) > 0
    else 0
)

duplicate_summary = pd.DataFrame({
    "Metric": [
        "Duplicate Rows",
        "Duplicate Percentage"
    ],
    "Value": [
        duplicate_count,
        round(duplicate_percentage, 4)
    ]
})

save_csv(
    duplicate_summary,
    "02_duplicate_analysis.csv"
)

if duplicate_count > 0:

    df = df.drop_duplicates().reset_index(drop=True)

    print("Duplicates removed.")

else:

    print("No duplicates found.")


# ============================================================
# 5. MISSING VALUES
# ============================================================

print_section("[4/12] MISSING VALUE ANALYSIS")

missing = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": df.isna().sum().values,
    "Missing_Percentage": (
        df.isna().mean() * 100
    ).round(4).values
})

missing = missing.sort_values(
    "Missing_Percentage",
    ascending=False
)

print(missing.to_string(index=False))

save_csv(
    missing,
    "03_missing_value_analysis.csv"
)


# ============================================================
# 6. UNIQUE VALUE ANALYSIS
# ============================================================

print_section("[5/12] UNIQUE VALUE ANALYSIS")

unique = pd.DataFrame({
    "Column": df.columns,
    "Unique_Count": [
        df[column].nunique(dropna=True)
        for column in df.columns
    ]
})

unique["Unique_Percentage"] = (
    unique["Unique_Count"] / len(df) * 100
).round(4)

unique = unique.sort_values(
    "Unique_Count",
    ascending=False
)

print(unique.to_string(index=False))

save_csv(
    unique,
    "04_unique_value_analysis.csv"
)


# ============================================================
# 7. DATE DETECTION
# ============================================================

print_section("[6/12] DATE ANALYSIS")

date_candidates = [
    "order_purchase_timestamp",
    "order_date",
    "purchase_date",
    "date",
    "purchase_datetime"
]

date_col = find_column(
    df,
    date_candidates
)

if date_col is not None:

    converted_date = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    valid_percentage = (
        converted_date.notna().mean() * 100
    )

    if valid_percentage >= 50:

        df[date_col] = converted_date

        print("Date column:", date_col)
        print(
            "Valid dates:",
            round(valid_percentage, 2),
            "%"
        )

    else:

        print(
            "Date column found, but parsing quality is too low."
        )

        date_col = None

else:

    print("No date column detected.")


date_audit = pd.DataFrame({
    "Item": [
        "Date Column",
        "Date Status"
    ],
    "Value": [
        str(date_col),
        "Detected and parsed"
        if date_col
        else "Not detected"
    ]
})

save_csv(
    date_audit,
    "05_date_detection.csv"
)


# ============================================================
# 8. STATISTICAL SUMMARY
# ============================================================

print_section("[7/12] STATISTICAL SUMMARY")

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    include="object"
).columns.tolist()

print("\nNumeric columns:")
print(numeric_columns)

if len(numeric_columns) > 0:

    numerical_stats = (
        df[numeric_columns]
        .describe()
        .T
        .reset_index()
    )

    numerical_stats.rename(
        columns={"index": "Column"},
        inplace=True
    )

    save_csv(
        numerical_stats,
        "06_numeric_descriptive_statistics.csv"
    )


# ============================================================
# 9. SKEWNESS + KURTOSIS
# ============================================================

print_section("[8/12] SKEWNESS + KURTOSIS")

skew_rows = []

for column in numeric_columns:

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if len(values) == 0:

        skew_value = np.nan
        kurtosis_value = np.nan
        assessment = "No valid data"

    else:

        skew_value = values.skew()
        kurtosis_value = values.kurtosis()

        if skew_value > 1:

            assessment = "Highly Right-Skewed"

        elif skew_value > 0.5:

            assessment = "Moderately Right-Skewed"

        elif skew_value < -1:

            assessment = "Highly Left-Skewed"

        elif skew_value < -0.5:

            assessment = "Moderately Left-Skewed"

        else:

            assessment = "Approximately Symmetric"

    skew_rows.append({
        "Column": column,
        "Skewness": skew_value,
        "Kurtosis": kurtosis_value,
        "Distribution_Assessment": assessment
    })

skewness_df = pd.DataFrame(skew_rows)

if not skewness_df.empty:

    skewness_df = skewness_df.sort_values(
        "Skewness",
        key=lambda x: x.abs(),
        ascending=False
    )

print(skewness_df.to_string(index=False))

save_csv(
    skewness_df,
    "07_skewness_kurtosis_analysis.csv"
)


# -----------------------------
# Skewness plot
# -----------------------------

if not skewness_df.empty:

    plot_df = skewness_df.dropna(
        subset=["Skewness"]
    )

    plt.figure(figsize=(11, 6))

    plt.barh(
        plot_df["Column"].astype(str),
        plot_df["Skewness"]
    )

    plt.axvline(0)

    plt.title(
        "Skewness Analysis of Numerical Features"
    )

    plt.xlabel("Skewness")
    plt.ylabel("Feature")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "01_skewness_analysis.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 10. OUTLIER ANALYSIS
# ============================================================

print_section("[9/12] OUTLIER ANALYSIS")

outlier_rows = []

for column in numeric_columns:

    series = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if len(series) == 0:

        continue

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outlier_count = int(
        (
            (series < lower_limit) |
            (series > upper_limit)
        ).sum()
    )

    outlier_percentage = (
        outlier_count / len(series) * 100
    )

    outlier_rows.append({
        "Column": column,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower_Bound": lower_limit,
        "Upper_Bound": upper_limit,
        "Outlier_Count": outlier_count,
        "Outlier_Percentage": outlier_percentage
    })

outlier_df = pd.DataFrame(outlier_rows)

if not outlier_df.empty:

    outlier_df = outlier_df.sort_values(
        "Outlier_Percentage",
        ascending=False
    )

print(outlier_df.to_string(index=False))

save_csv(
    outlier_df,
    "08_outlier_analysis.csv"
)


# ============================================================
# 11. CORRELATION ANALYSIS
# ============================================================

print_section("[10/12] CORRELATION + DEPENDENCE ANALYSIS")

if len(numeric_columns) >= 2:

    correlation = df[
        numeric_columns
    ].corr()

    correlation_output = (
        correlation
        .reset_index()
        .rename(columns={"index": "Feature"})
    )

    save_csv(
        correlation_output,
        "09_correlation_matrix.csv"
    )

    # High correlations
    pairs = []

    for i in range(len(correlation.columns)):

        for j in range(
            i + 1,
            len(correlation.columns)
        ):

            value = correlation.iloc[i, j]

            if pd.notna(value) and abs(value) >= 0.70:

                pairs.append({
                    "Feature_1":
                        correlation.columns[i],

                    "Feature_2":
                        correlation.columns[j],

                    "Correlation":
                        round(value, 4),

                    "Absolute_Correlation":
                        round(abs(value), 4)
                })

    high_corr_df = pd.DataFrame(pairs)

    if not high_corr_df.empty:

        high_corr_df = high_corr_df.sort_values(
            "Absolute_Correlation",
            ascending=False
        )

    print("\nHigh correlation pairs |r| >= 0.70:")

    if high_corr_df.empty:

        print("None found.")

    else:

        print(
            high_corr_df.to_string(index=False)
        )

    save_csv(
        high_corr_df,
        "10_high_correlation_pairs.csv"
    )

    # -----------------------------
    # Correlation heatmap
    # -----------------------------

    plt.figure(figsize=(13, 10))

    plt.imshow(
        correlation.values,
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "02_correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 12. NUMERICAL DISTRIBUTIONS
# ============================================================

print_section("[11/12] NUMERICAL DISTRIBUTION PLOTS")

for column in numeric_columns:

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        continue

    # Histogram
    plt.figure(figsize=(9, 5))

    plt.hist(
        values,
        bins=40
    )

    plt.title(
        f"Distribution of {column}"
    )

    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR /
        f"distribution_{safe_filename(column)}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Individual boxplot
    plt.figure(figsize=(9, 4))

    plt.boxplot(
        values,
        vert=False
    )

    plt.title(
        f"Boxplot of {column}"
    )

    plt.xlabel(column)

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR /
        f"boxplot_{safe_filename(column)}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 13. BUSINESS TREND ANALYSIS
# ============================================================

print_section("[12/12] BUSINESS + CUSTOMER + COHORT ANALYSIS")

order_col = find_column(
    df,
    [
        "order_id",
        "order_identifier",
        "order_number"
    ]
)

customer_col = find_column(
    df,
    [
        "customer_unique_id",
        "customer_id",
        "customer_unique",
        "customer_identifier"
    ]
)

value_col = find_column(
    df,
    [
        "order_item_value",
        "order_value",
        "total_order_value",
        "payment_value"
    ]
)

freight_col = find_column(
    df,
    [
        "total_freight_value",
        "freight_value",
        "order_freight_value"
    ]
)

print("\nDetected columns:")
print("Order column   :", order_col)
print("Customer column:", customer_col)
print("Value column   :", value_col)
print("Freight column :", freight_col)


# ============================================================
# MONTHLY BUSINESS ANALYSIS
# ============================================================

if date_col is not None:

    temp = df.copy()

    temp["purchase_period"] = (
        temp[date_col]
        .dt.to_period("M")
        .astype(str)
    )

    if order_col:

        monthly = (
            temp.groupby("purchase_period")
            .agg(
                Orders=(order_col, "nunique"),
                Customers=(customer_col, "nunique")
                if customer_col
                else (date_col, "size")
            )
            .reset_index()
        )

    else:

        monthly = (
            temp.groupby("purchase_period")
            .size()
            .reset_index(name="Orders")
        )

    if value_col:

        revenue = (
            temp.groupby("purchase_period")[value_col]
            .sum()
            .reset_index(name="Revenue")
        )

        monthly = monthly.merge(
            revenue,
            on="purchase_period",
            how="left"
        )

    save_csv(
        monthly,
        "11_monthly_business_analysis.csv"
    )

    # Orders trend
    plt.figure(figsize=(13, 5))

    plt.plot(
        monthly["purchase_period"],
        monthly["Orders"],
        marker="o"
    )

    plt.title(
        "Monthly Orders Trend"
    )

    plt.xlabel("Month")
    plt.ylabel("Orders")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "03_monthly_orders_trend.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Revenue trend
    if "Revenue" in monthly.columns:

        plt.figure(figsize=(13, 5))

        plt.plot(
            monthly["purchase_period"],
            monthly["Revenue"],
            marker="o"
        )

        plt.title(
            "Monthly Revenue Trend"
        )

        plt.xlabel("Month")
        plt.ylabel("Revenue")

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(
            PLOT_DIR / "04_monthly_revenue_trend.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


# ============================================================
# CUSTOMER TYPE ANALYSIS
# ============================================================

if customer_col and order_col:

    customer_orders = (
        df.groupby(customer_col)[order_col]
        .nunique()
        .reset_index(
            name="Total_Orders"
        )
    )

    customer_orders["Customer_Type"] = np.where(
        customer_orders["Total_Orders"] > 1,
        "Repeat Customer",
        "One-Time Customer"
    )

    counts = (
        customer_orders["Customer_Type"]
        .value_counts()
    )

    one_time = int(
        counts.get(
            "One-Time Customer",
            0
        )
    )

    repeat = int(
        counts.get(
            "Repeat Customer",
            0
        )
    )

    total_customers = one_time + repeat

    one_time_percentage = (
        one_time / total_customers * 100
        if total_customers > 0
        else 0
    )

    repeat_percentage = (
        repeat / total_customers * 100
        if total_customers > 0
        else 0
    )

    customer_summary = pd.DataFrame({
        "Customer_Type": [
            "One-Time Customer",
            "Repeat Customer"
        ],

        "Customer_Count": [
            one_time,
            repeat
        ],

        "Percentage": [
            one_time_percentage,
            repeat_percentage
        ]
    })

    print("\nCustomer type:")
    print(
        customer_summary.to_string(
            index=False
        )
    )

    save_csv(
        customer_summary,
        "12_repeat_vs_one_time_customers.csv"
    )

    save_csv(
        customer_orders,
        "13_customer_order_frequency.csv"
    )

    # Ratio
    if repeat > 0:

        ratio = one_time / repeat

        ratio_text = (
            f"{ratio:.2f}:1 "
            "(One-Time : Repeat)"
        )

    else:

        ratio_text = (
            "Undefined - no repeat customers"
        )

    imbalance_summary = pd.DataFrame({
        "Metric": [
            "One-Time Customers",
            "Repeat Customers",
            "One-Time Percentage",
            "Repeat Percentage",
            "One-Time : Repeat Ratio"
        ],

        "Value": [
            one_time,
            repeat,
            round(one_time_percentage, 4),
            round(repeat_percentage, 4),
            ratio_text
        ]
    })

    save_csv(
        imbalance_summary,
        "14_customer_class_distribution.csv"
    )

    # Plot
    plt.figure(figsize=(8, 5))

    plt.bar(
        customer_summary["Customer_Type"],
        customer_summary["Customer_Count"]
    )

    plt.title(
        "One-Time vs Repeat Customers"
    )

    plt.xlabel("Customer Type")
    plt.ylabel("Number of Customers")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR /
        "05_repeat_vs_one_time_customers.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# ORDER VALUE ANALYSIS
# ============================================================

if value_col and order_col:

    order_value = (
        df.groupby(order_col)[value_col]
        .sum()
        .reset_index(
            name="Order_Value"
        )
    )

    bins = [
        -np.inf,
        100,
        250,
        500,
        1000,
        np.inf
    ]

    labels = [
        "Under 100",
        "100-250",
        "250-500",
        "500-1000",
        "1000+"
    ]

    order_value[
        "Order_Value_Category"
    ] = pd.cut(
        order_value["Order_Value"],
        bins=bins,
        labels=labels
    )

    print("\nOrder value statistics:")
    print(
        order_value["Order_Value"]
        .describe()
    )

    save_csv(
        order_value,
        "15_order_value_analysis.csv"
    )

    plt.figure(figsize=(10, 5))

    plt.hist(
        order_value["Order_Value"].dropna(),
        bins=40
    )

    plt.title(
        "Order Value Distribution"
    )

    plt.xlabel("Order Value")
    plt.ylabel("Number of Orders")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR /
        "06_order_value_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# FREIGHT ANALYSIS
# ============================================================

if freight_col:

    freight_summary = (
        df[freight_col]
        .describe()
        .reset_index()
    )

    freight_summary.rename(
        columns={"index": "Statistic"},
        inplace=True
    )

    save_csv(
        freight_summary,
        "16_freight_value_summary.csv"
    )

    print("\nFreight summary:")
    print(freight_summary.to_string(index=False))


# ============================================================
# COHORT RETENTION ANALYSIS
# ============================================================

if date_col and customer_col:

    print("\nRunning cohort retention analysis...")

    cohort_data = df[
        [customer_col, date_col]
    ].dropna().copy()

    # First purchase date
    first_purchase = (
        cohort_data
        .groupby(customer_col)[date_col]
        .min()
        .reset_index(
            name="first_purchase_date"
        )
    )

    cohort_data = cohort_data.merge(
        first_purchase,
        on=customer_col,
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

    # Month index
    cohort_data["cohort_index"] = (
        (
            cohort_data["purchase_month"].dt.year
            -
            cohort_data["cohort_month"].dt.year
        ) * 12
        +
        (
            cohort_data["purchase_month"].dt.month
            -
            cohort_data["cohort_month"].dt.month
        )
        + 1
    )

    cohort_counts = (
        cohort_data
        .groupby(
            [
                "cohort_month",
                "cohort_index"
            ]
        )[customer_col]
        .nunique()
        .reset_index(
            name="Customers"
        )
    )

    cohort_size = (
        cohort_data
        .groupby("cohort_month")[customer_col]
        .nunique()
        .rename("Cohort_Size")
    )

    cohort_counts = cohort_counts.merge(
        cohort_size,
        left_on="cohort_month",
        right_index=True,
        how="left"
    )

    cohort_counts[
        "Retention_Percentage"
    ] = (
        cohort_counts["Customers"]
        /
        cohort_counts["Cohort_Size"]
        * 100
    )

    cohort_counts["Cohort_Month"] = (
        cohort_counts["cohort_month"]
        .astype(str)
    )

    cohort_output = cohort_counts[
        [
            "Cohort_Month",
            "cohort_index",
            "Customers",
            "Cohort_Size",
            "Retention_Percentage"
        ]
    ]

    save_csv(
        cohort_output,
        "17_cohort_retention_long_format.csv"
    )

    # Retention matrix
    retention_matrix = (
        cohort_counts
        .pivot_table(
            index="Cohort_Month",
            columns="cohort_index",
            values="Retention_Percentage"
        )
    )

    save_csv(
        retention_matrix.reset_index(),
        "18_cohort_retention_matrix.csv"
    )

    # Heatmap
    if not retention_matrix.empty:

        plt.figure(
            figsize=(
                max(
                    12,
                    retention_matrix.shape[1] * 0.65
                ),
                max(
                    7,
                    retention_matrix.shape[0] * 0.45
                )
            )
        )

        plt.imshow(
            retention_matrix.values,
            aspect="auto",
            vmin=0,
            vmax=100
        )

        plt.colorbar(
            label="Retention %"
        )

        plt.xticks(
            range(
                len(retention_matrix.columns)
            ),
            retention_matrix.columns
        )

        plt.yticks(
            range(
                len(retention_matrix.index)
            ),
            retention_matrix.index
        )

        plt.xlabel(
            "Cohort Month Index"
        )

        plt.ylabel(
            "Acquisition Cohort"
        )

        plt.title(
            "Customer Cohort Retention (%)"
        )

        plt.tight_layout()

        plt.savefig(
            PLOT_DIR /
            "07_cohort_retention_heatmap.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

else:

    print(
        "Cohort analysis skipped because "
        "date/customer column was not detected."
    )


# ============================================================
# FINAL PROJECT SUMMARY
# ============================================================

print_section("FINAL EDA SUMMARY")

summary_rows = [
    ("Original Rows", len(df_raw)),
    ("Final Rows", len(df)),
    ("Original Columns", len(df_raw.columns)),
    ("Final Columns", len(df.columns)),
    ("Duplicate Rows Removed", len(df_raw) - len(df)),
    ("Numeric Columns", len(numeric_columns)),
    ("Categorical Columns", len(categorical_columns)),
    ("Date Column", str(date_col)),
    ("Customer Column", str(customer_col)),
    ("Order Column", str(order_col)),
    ("Value Column", str(value_col)),
    ("Freight Column", str(freight_col))
]

if customer_col and order_col:

    summary_rows.append(
        ("Repeat Customers", repeat)
    )

    summary_rows.append(
        ("Repeat Customer Percentage",
         repeat_percentage)
    )

    summary_rows.append(
        ("One-Time Customers", one_time)
    )

    summary_rows.append(
        ("One-Time Customer Percentage",
         one_time_percentage)
    )


summary_df = pd.DataFrame(
    summary_rows,
    columns=[
        "Metric",
        "Value"
    ]
)

save_csv(
    summary_df,
    "19_EDA_project_summary.csv"
)


# ============================================================
# SAVE CLEANED DATA
# ============================================================

df.to_csv(
    DATA_DIR /
    "cleaned_olist_cohort_analysis_dataset.csv",
    index=False
)


# ============================================================
# README
# ============================================================

readme = f"""
# Olist Final EDA

## Dataset
Input:
{DATA_FILE}

## Main analyses
- Dataset structure
- Data types
- Missing values
- Duplicate rows
- Unique values
- Descriptive statistics
- Skewness
- Kurtosis
- Outlier analysis
- Correlation analysis
- High correlation pairs
- Numerical distributions
- Monthly business trends
- Repeat vs one-time customers
- Customer order frequency
- Order value analysis
- Freight analysis
- Cohort retention analysis

## Important methodology
- Duplicates are audited before removal.
- Missing values are reported.
- Outliers are detected using the IQR method.
- Outliers are not automatically deleted.
- Repeat customers are identified using distinct orders per customer.
- Cohort retention is based on distinct customers by acquisition month.

## Output folders

EDA_Results/
    tables/
    plots/
    data/
"""

save_text(
    readme,
    "README.md"
)


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 80)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 80)

print("\nDataset:")
print(DATA_FILE)

print("\nResults:")
print(OUTPUT_DIR)

print("\nTables:")
print(TABLE_DIR)

print("\nPlots:")
print(PLOT_DIR)

print("\nCleaned Data:")
print(DATA_DIR)

print("\nAll EDA analysis completed.")
print("DONE!")