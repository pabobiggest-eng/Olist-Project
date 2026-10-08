
# Olist Final EDA

## Dataset
Input:
C:\Users\National\Downloads\olist_3_data_vscode (1)\olist_3_data_vscode\olist_cohort_analysis_dataset.csv

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
