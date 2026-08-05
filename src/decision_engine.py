"""
decision_engine.py
=====================
Orchestrates the decision layer: pulls in inventory policy numbers, applies
the rules in business_rules.py, attaches SHAP-based explanations if
available, and outputs one structured recommendation per SKU-store, e.g.:

    Product: Coffee Powder
    Forecast: Demand expected to increase by 23%.
    Current Stock: 180 units
    Recommended Action: Reorder 320 units
    Reason: Weekend demand is expected to increase while supplier lead
            time is 5 days.
    Priority: HIGH

This module is intentionally thin -- it orchestrates and formats. The actual
decision *rules* live in business_rules.py so they can be reviewed and
unit-tested independently (see tests/test_decision_engine.py).

Run:
    python src/decision_engine.py
"""

import numpy as np
import pandas as pd
from utils import DATA_PROCESSED_DIR, get_logger
from business_rules import (
    classify_decision, classify_urgency, classify_status_light,
    recommended_quantity, priority_label, build_reason,
)

logger = get_logger("decision_engine")


def load_explanations() -> pd.DataFrame:
    """Optional join: top SHAP drivers per SKU-store, if explainability.py has run."""
    path = DATA_PROCESSED_DIR / "explanations.parquet"
    if path.exists():
        exp = pd.read_parquet(path)
        latest = exp.sort_values("date").groupby(["store_id", "sku_id"], observed=True).tail(1)
        return latest[["store_id", "sku_id", "top_drivers"]]
    logger.warning("explanations.parquet not found -- run explainability.py for SHAP drivers")
    return pd.DataFrame(columns=["store_id", "sku_id", "top_drivers"])


def pct_demand_change(row) -> float:
    """Forecast vs. trailing average, expressed as % change -- feeds the
    'Demand expected to increase by X%' line on the recommendation card."""
    baseline = row.get("roll_mean_14") or row.get("avg_daily_demand")
    if not baseline or baseline == 0 or pd.isna(baseline):
        return 0.0
    return 100 * (row["forecast"] - baseline) / baseline if "forecast" in row else 0.0


def run_pipeline() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "inventory_policy.parquet")
    explanations = load_explanations()

    df["decision"] = df.apply(
        lambda r: classify_decision(r["inventory_level"], r["reorder_point"],
                                     r["days_until_stockout"], r["lead_time_days"]),
        axis=1,
    )
    df["urgency"] = df.apply(lambda r: classify_urgency(r["decision"], r["abc_class"]), axis=1)
    df["priority"] = df["urgency"].apply(priority_label)
    df["status_light"] = df["days_until_stockout"].apply(classify_status_light)
    df["recommended_qty"] = df.apply(
        lambda r: recommended_quantity(r["decision"], r["reorder_point"],
                                        r["inventory_level"], r["eoq"]),
        axis=1,
    )

    df = df.merge(explanations, on=["store_id", "sku_id"], how="left")
    df["top_drivers"] = df["top_drivers"].fillna("")

    def compose_reason(row):
        demand_pct = None
        if "avg_daily_demand" in row and row.get("reorder_point", 0):
            demand_pct = None  # placeholder unless a trailing baseline is present
        driver_list = [d.split(" (")[0] for d in row["top_drivers"].split("; ") if d] \
            if row["top_drivers"] else []
        base_reason = build_reason(demand_pct, driver_list)
        if row["decision"] == "REORDER_NOW":
            return (
                f"{base_reason}. Stock covers only {row['days_until_stockout']:.1f} days "
                f"against a {row['lead_time_days']:.0f}-day supplier lead time."
            )
        elif row["decision"] == "REORDER_SOON":
            return f"{base_reason}. Inventory has fallen below the reorder point."
        elif row["decision"] == "OVERSTOCKED_REDUCE":
            return f"{base_reason}. Inventory is more than 2.5x the reorder point, tying up capital."
        elif row["decision"] == "MONITOR":
            return "Inventory trending toward excess; no action needed yet, recheck next cycle."
        return "Inventory is within the healthy operating range."

    df["explanation"] = df.apply(compose_reason, axis=1)

    urgency_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    df["urgency_rank"] = df["urgency"].map(urgency_order)
    df = df.sort_values(["urgency_rank", "estimated_stockout_cost"], ascending=[True, False])

    cols = [
        "store_id", "sku_id", "abc_class", "xyz_class", "decision", "urgency", "priority",
        "status_light", "inventory_level", "reorder_point", "safety_stock", "recommended_qty",
        "days_until_stockout", "lead_time_days", "estimated_stockout_cost",
        "annual_holding_cost", "top_drivers", "explanation",
    ]
    out = df[cols].copy()

    out_path = DATA_PROCESSED_DIR / "decisions.parquet"
    out.to_parquet(out_path, index=False)
    out.to_csv(DATA_PROCESSED_DIR / "decisions.csv", index=False)

    logger.info(f"Generated {len(out):,} decisions")
    logger.info(f"\n{out['decision'].value_counts().to_string()}")
    logger.info(f"Saved to {out_path}")
    return out


def print_recommendation_card(row: pd.Series) -> str:
    """Formats a single row as the human-readable card shown in the README/demo."""
    action = (
        f"Reorder {row['recommended_qty']:.0f} units" if row["recommended_qty"] > 0
        else (f"Reduce stock by {abs(row['recommended_qty']):.0f} units"
              if row["recommended_qty"] < 0 else "No action")
    )
    return (
        f"Product: {row['sku_id']}\n"
        f"Store: {row['store_id']}\n"
        f"Current Stock: {row['inventory_level']:.0f} units\n"
        f"Recommended Action: {action}\n"
        f"Reason: {row['explanation']}\n"
        f"Priority: {row['priority']}\n"
    )


if __name__ == "__main__":
    result = run_pipeline()
    print("\n--- Sample Recommendation Card ---\n")
    print(print_recommendation_card(result.iloc[0]))
