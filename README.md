# Olist — 3 Data Files + Merged Dataset

1. `01_olist_orders_dataset.csv` — order-level data
2. `02_olist_customers_dataset.csv` — customer-level data
3. `03_olist_order_items_summary.csv` — item/value summary by order
4. `04_olist_cohort_merged_dataset.csv` — merged master dataset

Merge keys:
- Orders ↔ Customers: `customer_id`
- Orders ↔ Items summary: `order_id`

Note: the uploaded ZIP contained an already-merged dataset, not the original
raw item-level Olist CSV. Therefore file 03 is derived from the available data
and is a summary, not the original raw item-level file.
