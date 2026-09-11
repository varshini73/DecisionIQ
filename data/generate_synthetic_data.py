"""Create a realistic *synthetic* daily retail panel for DecisionIQ.

Writes one canonical CSV, ``data/raw/retail_inventory_raw.csv``. It is for a
portfolio demonstration only; it is not retailer data.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "raw" / "retail_inventory_raw.csv"
RNG = np.random.default_rng(42)


def generate_data(n_stores=6, n_skus=30, start="2023-01-01", end="2024-12-31"):
    dates = pd.date_range(start, end, freq="D")
    categories = ["Grocery", "Beauty", "Apparel", "Home & Kitchen", "Toys"]
    holidays = set(pd.to_datetime([f"{y}-{m}-{d}" for y in (2023, 2024)
                                   for m, d in ((1, 1), (11, 24), (12, 25))]))
    weekly = np.array([.82, .86, .91, .96, 1.10, 1.28, 1.08]); rows = []
    for sku_num in range(1, n_skus + 1):
        category = categories[(sku_num - 1) % len(categories)]
        selling_price = float(RNG.uniform(5, 80)); unit_cost = round(selling_price * RNG.uniform(.45, .68), 2)
        lead_time = int(RNG.choice([3, 5, 7, 10, 14], p=[.15, .30, .30, .15, .10])); base = RNG.uniform(4, 28); trend = RNG.uniform(-.004, .012)
        for store_num in range(1, n_stores + 1):
            factor = RNG.uniform(.65, 1.35); stock = float(RNG.uniform(base * lead_time * 1.2, base * lead_time * 3)); arrival = None
            for day_idx, date in enumerate(dates):
                promotion, holiday = int(RNG.random() < .055), int(date in holidays)
                latent = max(0, (base + trend * day_idx) * factor * weekly[date.dayofweek] * (1 + .16*np.sin(2*np.pi*day_idx/365.25)) * (1.55 if promotion else 1) * (1.35 if holiday else 1) + RNG.normal(0, max(1, base*.16)))
                if RNG.random() < .025: latent *= RNG.uniform(1.8, 2.8)
                demand = int(round(latent))
                if arrival and day_idx == arrival[0]: stock += arrival[1]; arrival = None
                available = int(stock > 0); sales = min(demand, int(max(stock, 0))); stock = max(0, stock-sales)
                if stock <= base * lead_time * .55 and arrival is None: arrival = (day_idx + lead_time, max(base*lead_time*2.5, demand*lead_time))
                price = round(selling_price * (1-.12 if promotion else 1), 2)
                rows.append({"date":date,"store_id":f"Store_{store_num:02d}","sku_id":f"SKU_{sku_num:04d}","category":category,"sales":sales,"price":price,"promotion":promotion,"stock_available":available,"inventory_level":round(stock,1),"lead_time_days":lead_time,"unit_cost":unit_cost,"selling_price":price,"holiday":holiday})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True); frame = generate_data(); frame.to_csv(OUT, index=False)
    print(f"Wrote {len(frame):,} synthetic rows to {OUT}")
