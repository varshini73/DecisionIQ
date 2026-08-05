"""
feature_engineering.py
========================
Turns the cleaned daily panel into a model-ready feature table:
lags, rolling stats, calendar features, price/promo signal, demand
un-censoring for stockout days, and ABC/XYZ classification.

Run:
    python src/feature_engineering.py
"""

import pandas as pd
import numpy as np
from utils import DATA_PROCESSED_DIR, CONFIG, get_logger

logger = get_logger("feature_engineering")


def uncensor_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    On stockout days, observed units_sold understates true demand.
    Replace those days' sales with the series' rolling average of
    non-stockout days, so the forecasting model learns true demand,
    not truncated demand.
    """
    df = df.sort_values(["store_id", "sku_id", "date"])

    def correct(g):
        clean_mean = g.loc[g["is_stockout_day"] == 0, "units_sold"].rolling(
            14, min_periods=3
        ).mean()
        g["demand_est"] = g["units_sold"]
        stockout_idx = g["is_stockout_day"] == 1
        # forward-fill the rolling clean mean into stockout days
        fill_series = clean_mean.reindex(g.index).ffill()
        g.loc[stockout_idx, "demand_est"] = fill_series[stockout_idx]
        g["demand_est"] = g["demand_est"].fillna(g["units_sold"])
        return g

    df = df.groupby(["store_id", "sku_id"], group_keys=False, observed=True).apply(correct)
    logger.info("Applied demand un-censoring correction for stockout days")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame, target: str = "demand_est") -> pd.DataFrame:
    df = df.sort_values(["store_id", "sku_id", "date"])
    grp = df.groupby(["store_id", "sku_id"], observed=True)[target]

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = grp.shift(lag)

    for window in [7, 14, 30]:
        shifted = grp.shift(1)  # avoid leakage: rolling stats exclude current day
        df[f"roll_mean_{window}"] = shifted.groupby(
            [df["store_id"], df["sku_id"]], observed=True
        ).transform(lambda s: s.rolling(window, min_periods=3).mean())
        df[f"roll_std_{window}"] = shifted.groupby(
            [df["store_id"], df["sku_id"]], observed=True
        ).transform(lambda s: s.rolling(window, min_periods=3).std())

    return df


def add_price_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    if "price" in df.columns:
        cat_avg_price = df.groupby(["category", "date"], observed=True)["price"].transform("mean")
        df["price_vs_category_avg"] = df["price"] / cat_avg_price.replace(0, np.nan)
    if "discount" in df.columns:
        df["discount_pct"] = df["discount"].fillna(0)
    if "promotion" not in df.columns:
        df["promotion"] = 0
    return df


def add_inventory_features(df: pd.DataFrame) -> pd.DataFrame:
    if "inventory_level" in df.columns:
        df = df.sort_values(["store_id", "sku_id", "date"])
        rolling_daily_demand = df.groupby(["store_id", "sku_id"], observed=True)[
            "demand_est"
        ].transform(lambda s: s.shift(1).rolling(14, min_periods=3).mean())
        df["days_of_supply"] = df["inventory_level"] / rolling_daily_demand.replace(0, np.nan)
        df["days_of_supply"] = df["days_of_supply"].clip(upper=365)
        df["recent_stockout_flag"] = (
            df.groupby(["store_id", "sku_id"], observed=True)["is_stockout_day"]
            .transform(lambda s: s.shift(1).rolling(14, min_periods=1).max())
            .fillna(0)
        )
    return df


def abc_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classic ABC analysis by cumulative revenue share, computed at the SKU
    level (aggregated across stores) so the class is stable and interpretable.
    """
    revenue = (df["demand_est"] * df.get("price", 1)).groupby(df["sku_id"], observed=True).sum()
    revenue = revenue.sort_values(ascending=False)
    cum_share = revenue.cumsum() / revenue.sum()

    thresholds = CONFIG["abc_thresholds"]
    abc_map = {}
    for sku, share in cum_share.items():
        if share <= thresholds["A"]:
            abc_map[sku] = "A"
        elif share <= thresholds["B"]:
            abc_map[sku] = "B"
        else:
            abc_map[sku] = "C"

    df["abc_class"] = df["sku_id"].map(abc_map)
    logger.info(
        f"ABC classification: "
        f"{sum(v=='A' for v in abc_map.values())} A-items, "
        f"{sum(v=='B' for v in abc_map.values())} B-items, "
        f"{sum(v=='C' for v in abc_map.values())} C-items"
    )
    return df


def xyz_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    XYZ analysis by demand variability (coefficient of variation).
    X = stable/predictable demand, Y = moderate variability, Z = erratic.
    Combined with ABC this drives review frequency and safety stock policy.
    """
    stats = df.groupby("sku_id", observed=True)["demand_est"].agg(["mean", "std"])
    stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)

    def classify(cv):
        if pd.isna(cv):
            return "Z"
        if cv <= 0.5:
            return "X"
        elif cv <= 1.0:
            return "Y"
        return "Z"

    xyz_map = stats["cv"].apply(classify).to_dict()
    df["xyz_class"] = df["sku_id"].map(xyz_map)
    return df


def run_pipeline() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "clean_panel.parquet")

    df = uncensor_demand(df)
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    df = add_price_promo_features(df)
    df = add_inventory_features(df)
    df = abc_classification(df)
    df = xyz_classification(df)

    out_path = DATA_PROCESSED_DIR / "features.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved feature table to {out_path} ({df.shape[0]:,} rows, {df.shape[1]} cols)")
    return df


if __name__ == "__main__":
    run_pipeline()
