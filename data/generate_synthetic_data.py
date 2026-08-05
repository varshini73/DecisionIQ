"""
generate_synthetic_data.py
---------------------------
Generates a realistic multi-store, multi-SKU retail transaction dataset.

WHY SYNTHETIC DATA:
Real retail POS data with SKU-level daily sales is rarely public because of
commercial sensitivity. Kaggle's "Store Item Demand Forecasting Challenge"
and "Rossmann Store Sales" are the closest public analogues (see README for
links), but they don't include cost/price/lead-time fields needed for an
inventory-optimization layer. This script generates data with the same
statistical properties (trend, weekly seasonality, promo spikes, noise) PLUS
the extra business fields (unit cost, lead time, holding cost %) so the full
pipeline -- forecast -> reorder point -> safety stock -> ABC -> dashboard --
can be demonstrated end to end.

You can swap this out for the real Kaggle CSVs later; the preprocessing
module expects the same schema documented in README.md.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_STORES = 10
N_PRODUCTS = 60
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2024, 12, 31)

CATEGORIES = ["Grocery", "Electronics", "Apparel", "Home & Kitchen", "Beauty", "Toys"]


def generate_products(n_products=N_PRODUCTS):
    rows = []
    for i in range(1, n_products + 1):
        category = np.random.choice(CATEGORIES)
        base_price = {
            "Grocery": np.random.uniform(2, 25),
            "Electronics": np.random.uniform(50, 900),
            "Apparel": np.random.uniform(15, 120),
            "Home & Kitchen": np.random.uniform(10, 250),
            "Beauty": np.random.uniform(5, 80),
            "Toys": np.random.uniform(8, 100),
        }[category]
        unit_cost = base_price * np.random.uniform(0.45, 0.65)
        lead_time_days = np.random.choice([3, 5, 7, 10, 14], p=[0.15, 0.3, 0.3, 0.15, 0.1])
        rows.append({
            "product_id": f"P{i:04d}",
            "product_name": f"{category} Item {i}",
            "category": category,
            "unit_price": round(base_price, 2),
            "unit_cost": round(unit_cost, 2),
            "supplier_lead_time_days": lead_time_days,
            "holding_cost_pct": round(np.random.uniform(0.15, 0.30), 3),  # annual % of unit cost
            "ordering_cost": round(np.random.uniform(20, 100), 2),  # fixed cost per PO
        })
    return pd.DataFrame(rows)


def generate_stores(n_stores=N_STORES):
    rows = []
    regions = ["North", "South", "East", "West"]
    for i in range(1, n_stores + 1):
        rows.append({
            "store_id": f"S{i:03d}",
            "region": np.random.choice(regions),
            "store_size_sqft": int(np.random.uniform(3000, 20000)),
        })
    return pd.DataFrame(rows)


def simulate_daily_demand(base_level, trend_per_day, day_index, day_of_week,
                           is_promo, is_holiday, noise_scale):
    weekly_mult = {0: 0.9, 1: 0.85, 2: 0.9, 3: 0.95, 4: 1.15, 5: 1.35, 6: 1.1}[day_of_week]
    trend = base_level + trend_per_day * day_index
    promo_mult = 1.6 if is_promo else 1.0
    holiday_mult = 1.8 if is_holiday else 1.0
    seasonal = 1 + 0.15 * np.sin(2 * np.pi * day_index / 365)  # annual seasonality
    demand = trend * weekly_mult * promo_mult * holiday_mult * seasonal
    demand += np.random.normal(0, noise_scale)
    return max(0, np.round(demand))


def generate_transactions(products, stores):
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    holidays = set(pd.to_datetime(["2022-12-25", "2023-12-25", "2024-12-25",
                                    "2022-11-25", "2023-11-24", "2024-11-29",  # Black Friday-ish
                                    "2022-01-01", "2023-01-01", "2024-01-01"]))
    records = []
    for _, prod in products.iterrows():
        base_level = np.random.uniform(3, 40)
        trend_per_day = np.random.uniform(-0.01, 0.02)
        noise_scale = base_level * 0.2
        promo_days = set(np.random.choice(len(dates), size=int(len(dates) * 0.05), replace=False))

        for _, store in stores.iterrows():
            store_factor = np.random.uniform(0.6, 1.4)
            for day_index, date in enumerate(dates):
                is_promo = day_index in promo_days
                is_holiday = date in holidays
                units = simulate_daily_demand(
                    base_level * store_factor, trend_per_day, day_index,
                    date.dayofweek, is_promo, is_holiday, noise_scale
                )
                if units > 0 or np.random.random() < 0.02:  # keep some zero-demand rows
                    records.append({
                        "date": date,
                        "store_id": store["store_id"],
                        "product_id": prod["product_id"],
                        "units_sold": int(units),
                        "is_promo": int(is_promo),
                        "is_holiday": int(is_holiday),
                    })
    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Generating products...")
    products = generate_products()
    print("Generating stores...")
    stores = generate_stores()
    print("Generating transactions (this can take a minute)...")
    transactions = generate_transactions(products, stores)

    products.to_csv("data/raw/products.csv", index=False)
    stores.to_csv("data/raw/stores.csv", index=False)
    transactions.to_csv("data/raw/transactions.csv", index=False)

    print(f"Products: {products.shape}, Stores: {stores.shape}, Transactions: {transactions.shape}")
    print("Saved to data/raw/")
