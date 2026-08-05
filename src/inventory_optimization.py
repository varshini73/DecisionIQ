"""
inventory_optimization.py
============================
Converts demand forecasts into inventory policy numbers:
safety stock, reorder point, economic order quantity, and cost impact.

Run:
    python src/inventory_optimization.py
"""

import numpy as np
import pandas as pd
from utils import DATA_PROCESSED_DIR, CONFIG, get_logger

logger = get_logger("inventory_optimization")


def compute_forecast_error_std(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safety stock should be based on forecast ERROR variability, not raw
    demand variability -- a perfectly forecastable but highly seasonal SKU
    needs less safety stock than a poorly-forecasted but stable one.
    """
    df["forecast_error"] = df["demand_est"] - df["forecast"]
    err_std = df.groupby(["store_id", "sku_id"], observed=True)["forecast_error"].transform("std")
    df["forecast_error_std"] = err_std.fillna(df["demand_est"].std())
    return df


def compute_avg_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    avg_demand = df.groupby(["store_id", "sku_id"], observed=True)["forecast"].transform("mean")
    df["avg_daily_demand"] = avg_demand
    return df


def get_lead_time(row, lead_time_map: dict = None) -> float:
    """Lead time can vary by SKU/category in a real system; defaulted here."""
    if lead_time_map and row["sku_id"] in lead_time_map:
        return lead_time_map[row["sku_id"]]
    return CONFIG["default_lead_time_days"]


def compute_safety_stock(df: pd.DataFrame, lead_time_map: dict = None) -> pd.DataFrame:
    z = CONFIG["service_level_z"]
    df["lead_time_days"] = df.apply(lambda r: get_lead_time(r, lead_time_map), axis=1)
    df["safety_stock"] = z * df["forecast_error_std"] * np.sqrt(df["lead_time_days"])
    df["safety_stock"] = df["safety_stock"].round().clip(lower=0)
    return df


def compute_reorder_point(df: pd.DataFrame) -> pd.DataFrame:
    df["reorder_point"] = (df["avg_daily_demand"] * df["lead_time_days"]) + df["safety_stock"]
    df["reorder_point"] = df["reorder_point"].round()
    return df


def compute_eoq(df: pd.DataFrame, unit_cost_col: str = None) -> pd.DataFrame:
    """
    Economic Order Quantity: sqrt(2 * D * S / H)
    D = annual demand, S = ordering cost per order, H = annual holding cost/unit.
    If unit cost isn't available, use a flat assumed cost so EOQ still runs.
    """
    S = CONFIG["ordering_cost_per_order"]
    unit_cost = df[unit_cost_col] if unit_cost_col and unit_cost_col in df.columns else 10.0
    H = CONFIG["holding_cost_rate"] * unit_cost

    annual_demand = df["avg_daily_demand"] * 365
    df["eoq"] = np.sqrt((2 * annual_demand * S) / H.replace(0, np.nan) if hasattr(H, "replace") else (2 * annual_demand * S) / H)
    df["eoq"] = df["eoq"].round().clip(lower=1)
    return df


def compute_costs(df: pd.DataFrame, unit_cost_col: str = None) -> pd.DataFrame:
    unit_cost = df[unit_cost_col] if unit_cost_col and unit_cost_col in df.columns else 10.0
    holding_cost_per_unit_year = CONFIG["holding_cost_rate"] * unit_cost

    df["annual_holding_cost"] = df["safety_stock"] * holding_cost_per_unit_year
    df["days_until_stockout"] = (
        df["inventory_level"] / df["avg_daily_demand"].replace(0, np.nan)
    ).clip(lower=0)

    margin_loss_rate = CONFIG["stockout_margin_loss_rate"]
    shortfall = (df["avg_daily_demand"] * df["lead_time_days"] - df["inventory_level"]).clip(lower=0)
    df["estimated_stockout_cost"] = shortfall * unit_cost * margin_loss_rate
    return df


def run_pipeline(unit_cost_col: str = None, lead_time_map: dict = None) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "forecast_results.parquet")

    # Use the latest available date per store-sku as the "current" snapshot
    latest = df.sort_values("date").groupby(["store_id", "sku_id"], observed=True).tail(1).copy()
    full_hist = df.copy()

    full_hist = compute_forecast_error_std(full_hist)
    full_hist = compute_avg_daily_demand(full_hist)

    snapshot_stats = full_hist.groupby(["store_id", "sku_id"], observed=True).agg(
        forecast_error_std=("forecast_error_std", "last"),
        avg_daily_demand=("avg_daily_demand", "last"),
    ).reset_index()

    latest = latest.merge(snapshot_stats, on=["store_id", "sku_id"], suffixes=("", "_calc"))
    latest["forecast_error_std"] = latest["forecast_error_std_calc"]
    latest["avg_daily_demand"] = latest["avg_daily_demand_calc"]
    latest = latest.drop(columns=["forecast_error_std_calc", "avg_daily_demand_calc"])

    latest = compute_safety_stock(latest, lead_time_map)
    latest = compute_reorder_point(latest)
    latest = compute_eoq(latest, unit_cost_col)
    latest = compute_costs(latest, unit_cost_col)

    out_path = DATA_PROCESSED_DIR / "inventory_policy.parquet"
    latest.to_parquet(out_path, index=False)
    logger.info(f"Saved inventory policy for {len(latest):,} SKU-store combinations to {out_path}")
    return latest


if __name__ == "__main__":
    run_pipeline()
