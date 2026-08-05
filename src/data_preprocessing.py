"""
data_preprocessing.py
======================
Loads raw retail sales/inventory data and produces a clean, continuous
daily panel of (store, sku, date) -> features ready for downstream
feature engineering.

Expected raw columns (rename in COLUMN_MAP if your CSV differs):
    Date, Store ID, Product ID, Category, Units Sold, Units Ordered,
    Inventory Level, Price, Discount, Promotion, Region

Run:
    python src/data_preprocessing.py
"""

import pandas as pd
import numpy as np
from utils import DATA_RAW_DIR, DATA_PROCESSED_DIR, get_logger

logger = get_logger("data_preprocessing")

# Map your raw CSV's column names to the pipeline's standard names here.
COLUMN_MAP = {
    "Date": "date",
    "Store ID": "store_id",
    "Product ID": "sku_id",
    "Category": "category",
    "Units Sold": "units_sold",
    "Units Ordered": "units_ordered",
    "Inventory Level": "inventory_level",
    "Price": "price",
    "Discount": "discount",
    "Promotion": "promotion",
    "Region": "region",
}


def load_raw_data(filename: str = "retail_inventory_raw.csv") -> pd.DataFrame:
    path = DATA_RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expected raw dataset at {path}. Download the recommended "
            f"Kaggle dataset (see README section 5) and place the CSV there."
        )
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    logger.info(f"Loaded raw data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    for col in ["store_id", "sku_id", "category"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    numeric_cols = ["units_sold", "units_ordered", "inventory_level", "price", "discount"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "promotion" in df.columns:
        df["promotion"] = df["promotion"].fillna(0).astype(int)
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["store_id", "sku_id", "date"])
    logger.info(f"Dropped {before - len(df):,} duplicate rows")
    return df


def reindex_continuous_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Every (store, sku) series must have one row per calendar day, even
    days with zero sales. Missing days are NOT missing data -- they are
    true zero-demand days and matter for accurate demand distributions.
    """
    full_frames = []
    date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

    for (store, sku), grp in df.groupby(["store_id", "sku_id"], observed=True):
        grp = grp.set_index("date").reindex(date_range)
        grp["store_id"] = store
        grp["sku_id"] = sku
        grp.index.name = "date"
        full_frames.append(grp)

    out = pd.concat(full_frames).reset_index()

    # Fill: sales/orders default to 0 on unlisted days; inventory forward-filled
    for col in ["units_sold", "units_ordered", "discount", "promotion"]:
        if col in out.columns:
            out[col] = out[col].fillna(0)
    if "inventory_level" in out.columns:
        out["inventory_level"] = out.groupby(["store_id", "sku_id"], observed=True)[
            "inventory_level"
        ].ffill()
    for col in ["category", "region", "price"]:
        if col in out.columns:
            out[col] = out.groupby(["store_id", "sku_id"], observed=True)[col].ffill().bfill()

    logger.info(f"Reindexed to continuous calendar: {out.shape[0]:,} rows")
    return out


def cap_outliers_per_series(df: pd.DataFrame, col: str = "units_sold") -> pd.DataFrame:
    """
    IQR-based capping, computed PER (store, sku) series -- a global cap would
    wrongly clip high-volume SKUs' normal sales as 'outliers'.
    """
    def cap_group(g):
        q1, q3 = g[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        upper = q3 + 3 * iqr  # generous bound: only clip extreme spikes
        g[col] = g[col].clip(upper=max(upper, g[col].median() + 1))
        return g

    df = df.groupby(["store_id", "sku_id"], group_keys=False, observed=True).apply(cap_group)
    return df


def flag_low_history_series(df: pd.DataFrame, min_days: int = 60) -> pd.DataFrame:
    """Drop SKU-store series with too little history to model reliably."""
    counts = df.groupby(["store_id", "sku_id"], observed=True)["date"].transform("count")
    before = df["sku_id"].nunique()
    df = df[counts >= min_days].copy()
    logger.info(
        f"Filtered series with < {min_days} days of history "
        f"({before} -> {df['sku_id'].nunique()} unique SKUs remaining across stores)"
    )
    return df


def flag_stockout_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demand censoring flag: if inventory hit ~0 and sales also flatline,
    observed 'units_sold' likely understates true demand for that day.
    This flag is consumed later by feature_engineering.py to correct demand.
    """
    if "inventory_level" in df.columns:
        df["is_stockout_day"] = (df["inventory_level"] <= 0).astype(int)
    else:
        df["is_stockout_day"] = 0
    return df


def run_pipeline(raw_filename: str = "retail_inventory_raw.csv") -> pd.DataFrame:
    df = load_raw_data(raw_filename)
    df = enforce_schema(df)
    df = drop_duplicates(df)
    df = reindex_continuous_calendar(df)
    df = cap_outliers_per_series(df)
    df = flag_low_history_series(df)
    df = flag_stockout_days(df)

    out_path = DATA_PROCESSED_DIR / "clean_panel.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved cleaned panel to {out_path} ({df.shape[0]:,} rows)")
    return df


if __name__ == "__main__":
    run_pipeline()
