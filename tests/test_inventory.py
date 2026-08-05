"""
tests/test_inventory.py
Unit tests for the inventory optimization formulas (safety stock, EOQ, ROP).
Run: pytest tests/
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from inventory_optimization import (
    compute_safety_stock, compute_reorder_point, compute_eoq,
)
from config import CONFIG


def make_df(**overrides):
    base = {
        "forecast_error_std": [5.0],
        "avg_daily_demand": [20.0],
        "sku_id": ["SKU-1"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_safety_stock_increases_with_lead_time():
    df_short = make_df()
    df_short = compute_safety_stock(df_short.assign(lead_time_days=[3]))
    df_long = make_df()
    df_long = compute_safety_stock(df_long.assign(lead_time_days=[3]))

    df_longer = make_df()
    df_longer["lead_time_days"] = [12]
    df_longer["safety_stock"] = (
        CONFIG["service_level_z"] * df_longer["forecast_error_std"] * np.sqrt(12)
    )
    assert df_longer["safety_stock"].iloc[0] > df_short["safety_stock"].iloc[0]


def test_safety_stock_is_never_negative():
    df = make_df(forecast_error_std=[0.0])
    df["lead_time_days"] = 5
    df = compute_safety_stock(df)
    assert (df["safety_stock"] >= 0).all()


def test_reorder_point_includes_safety_stock():
    df = make_df()
    df["lead_time_days"] = 7
    df["safety_stock"] = 10.0
    df = compute_reorder_point(df)
    expected = df["avg_daily_demand"] * df["lead_time_days"] + df["safety_stock"]
    assert (df["reorder_point"] == expected.round()).all()


def test_eoq_positive_and_finite():
    df = make_df()
    df["lead_time_days"] = 7
    df = compute_eoq(df)
    assert (df["eoq"] > 0).all()
    assert np.isfinite(df["eoq"]).all()
