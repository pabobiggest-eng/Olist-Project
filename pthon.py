import pandas as pd

# ==============================
# 1. Load the 3 datasets
# ==============================

orders = pd.read_csv("01_olist_orders_dataset.csv")
customers = pd.read_csv("02_olist_customers_dataset.csv")
items = pd.read_csv("03_olist_order_items_summary.csv")


# ==============================
# 2. Merge Orders + Customers
# ==============================

merged_data = orders.merge(
    customers,
    on="customer_id",
    how="left"
)


# ==============================
# 3. Merge with Order Items
# ==============================

merged_data = merged_data.merge(
    items,
    on="order_id",
    how="left"
)


# ==============================
# 4. Save final merged dataset
# ==============================

merged_data.to_csv(
    "final_olist_merged_dataset.csv",
    index=False
)


# ==============================
# 5. Show results
# ==============================

print("====================================")
print("MERGE COMPLETED SUCCESSFULLY!")
print("====================================")

print("Orders rows:", len(orders))
print("Customers rows:", len(customers))
print("Items rows:", len(items))

print("Final merged rows:", len(merged_data))
print("Final merged columns:", len(merged_data.columns))

print("\nFirst 5 rows:")
print(merged_data.head())

print("\nFinal dataset saved as:")
print("final_olist_merged_dataset.csv")