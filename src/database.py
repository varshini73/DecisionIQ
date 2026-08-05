"""
database.py
=============
SQLite persistence layer. Loads the pipeline's output parquet files into a
proper relational schema so the dashboard (and any future BI tool) can query
via SQL instead of re-running the pipeline every time.

Run:
    python src/database.py
"""

import sqlite3
import pandas as pd
from utils import DATA_PROCESSED_DIR, DB_PATH, get_logger

logger = get_logger("database")

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    date TEXT,
    store_id TEXT,
    sku_id TEXT,
    abc_class TEXT,
    xyz_class TEXT,
    inventory_level REAL,
    demand_est REAL,
    forecast REAL
);

CREATE TABLE IF NOT EXISTS decisions (
    store_id TEXT,
    sku_id TEXT,
    abc_class TEXT,
    xyz_class TEXT,
    decision TEXT,
    urgency TEXT,
    inventory_level REAL,
    reorder_point REAL,
    safety_stock REAL,
    recommended_qty REAL,
    days_until_stockout REAL,
    lead_time_days REAL,
    estimated_stockout_cost REAL,
    annual_holding_cost REAL,
    explanation TEXT
);

CREATE TABLE IF NOT EXISTS model_comparison (
    model TEXT,
    WAPE REAL,
    RMSE REAL,
    Bias REAL,
    FVA_vs_baseline REAL
);

CREATE INDEX IF NOT EXISTS idx_forecasts_sku_store ON forecasts (sku_id, store_id);
CREATE INDEX IF NOT EXISTS idx_decisions_urgency ON decisions (urgency);
"""


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def build_database():
    conn = get_connection()
    conn.executescript(SCHEMA)

    forecasts = pd.read_parquet(DATA_PROCESSED_DIR / "forecast_results.parquet")
    forecasts["date"] = forecasts["date"].astype(str)
    forecasts.to_sql("forecasts", conn, if_exists="replace", index=False)

    decisions = pd.read_parquet(DATA_PROCESSED_DIR / "decisions.parquet")
    decisions.to_sql("decisions", conn, if_exists="replace", index=False)

    try:
        comparison = pd.read_csv(DATA_PROCESSED_DIR / "model_comparison.csv")
        comparison.to_sql("model_comparison", conn, if_exists="replace", index=False)
    except FileNotFoundError:
        logger.warning("model_comparison.csv not found -- skipping that table")

    conn.commit()
    conn.close()
    logger.info(f"Database built at {DB_PATH}")


def query(sql: str) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


if __name__ == "__main__":
    build_database()
