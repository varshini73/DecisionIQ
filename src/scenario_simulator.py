"""
scenario_simulator.py
========================
Reusable what-if simulation logic, extracted from the dashboard so it can
also be called from tests, notebooks, or a future API. Answers questions
like:
    "What if demand increases by 25%?"
    "What if supplier lead time doubles?"
    "What if we reduce safety stock?"

Each function takes the current decisions/policy table and a scenario
lever, and returns the recomputed policy + a delta summary.
"""

import numpy as np
import pandas as pd
from config import CONFIG
from business_rules import classify_decision, classify_urgency, recommended_quantity, priority_label

# Approximate z-scores for common service levels (used by the dashboard slider)
Z_LOOKUP = {80: 0.84, 85: 1.04, 90: 1.28, 95: 1.65, 97: 1.88, 99: 2.33}


def z_for_service_level(service_level_pct: int) -> float:
    return min(Z_LOOKUP.items(), key=lambda kv: abs(kv[0] - service_level_pct))[1]


def simulate(
    df: pd.DataFrame,
    demand_uplift_pct: float = 0.0,
    lead_time_multiplier: float = 1.0,
    service_level_pct: int = None,
    inventory_level: float = None,
) -> pd.DataFrame:
    """
    Recomputes safety stock, reorder point, and cost impact under a scenario.

    Parameters
    ----------
    df : DataFrame with columns avg_daily_demand, forecast_error_std,
         lead_time_days, safety_stock, inventory_level, reorder_point
    demand_uplift_pct : e.g. 25 for "demand increases by 25%"
    lead_time_multiplier : e.g. 2.0 for "lead time doubles"
    service_level_pct : overrides CONFIG's default service level if provided
    """
    sim = df.copy()
    if inventory_level is not None:
        if inventory_level < 0:
            raise ValueError("inventory_level must be non-negative")
        sim["inventory_level"] = float(inventory_level)
    z = z_for_service_level(service_level_pct) if service_level_pct else CONFIG["service_level_z"]
    uplift_factor = 1 + demand_uplift_pct / 100

    sim["sim_avg_daily_demand"] = sim["avg_daily_demand"] * uplift_factor
    sim["sim_lead_time_days"] = sim["lead_time_days"] * lead_time_multiplier
    sim["sim_safety_stock"] = (
        z * sim["forecast_error_std"] * uplift_factor * np.sqrt(sim["sim_lead_time_days"])
    ).round().clip(lower=0)
    sim["sim_reorder_point"] = (
        sim["sim_avg_daily_demand"] * sim["sim_lead_time_days"] + sim["sim_safety_stock"]
    ).round()

    sim["sim_days_until_stockout"] = (
        sim["inventory_level"] / sim["sim_avg_daily_demand"].replace(0, np.nan)
    ).clip(lower=0)

    # Cost deltas use the observed unit cost when available, not a UI constant.
    unit_cost = sim["unit_cost"] if "unit_cost" in sim.columns else 10.0
    holding_cost_per_unit_year = CONFIG["holding_cost_rate"] * unit_cost
    sim["sim_holding_cost_delta"] = (
        (sim["sim_safety_stock"] - sim["safety_stock"]) * holding_cost_per_unit_year
    )

    shortfall = (sim["sim_reorder_point"] - sim["inventory_level"]).clip(lower=0)
    sim["sim_stockout_cost"] = shortfall * unit_cost * CONFIG["stockout_margin_loss_rate"]
    sim["sim_eoq"] = sim.get("eoq", shortfall).clip(lower=0)
    sim["sim_decision"] = sim.apply(lambda r: classify_decision(
        r["inventory_level"], r["sim_reorder_point"], r["sim_days_until_stockout"], r["sim_lead_time_days"]
    ), axis=1)
    sim["sim_recommended_qty"] = sim.apply(lambda r: recommended_quantity(
        r["sim_decision"], r["sim_reorder_point"], r["inventory_level"], r["sim_eoq"]
    ), axis=1)
    sim["sim_priority"] = sim.apply(lambda r: priority_label(classify_urgency(r["sim_decision"], r["abc_class"])), axis=1)

    return sim


def scenario_summary(sim: pd.DataFrame) -> dict:
    return {
        "avg_safety_stock_before": round(sim["safety_stock"].mean(), 1),
        "avg_safety_stock_after": round(sim["sim_safety_stock"].mean(), 1),
        "total_holding_cost_delta": round(sim["sim_holding_cost_delta"].sum(), 2),
        "total_simulated_stockout_cost": round(sim["sim_stockout_cost"].sum(), 2),
        "skus_newly_at_risk": int(
            (sim["sim_days_until_stockout"] <= sim["sim_lead_time_days"]).sum()
        ),
    }


# Convenience presets matching common interview/demo questions
PRESET_SCENARIOS = {
    "demand_up_25": dict(demand_uplift_pct=25, lead_time_multiplier=1.0),
    "lead_time_doubles": dict(demand_uplift_pct=0, lead_time_multiplier=2.0),
    "safety_stock_90pct_service": dict(demand_uplift_pct=0, lead_time_multiplier=1.0,
                                        service_level_pct=90),
    "peak_season": dict(demand_uplift_pct=40, lead_time_multiplier=1.3),
}


if __name__ == "__main__":
    from utils import DATA_PROCESSED_DIR, get_logger

    logger = get_logger("scenario_simulator")
    df = pd.read_parquet(DATA_PROCESSED_DIR / "inventory_policy.parquet")

    for name, params in PRESET_SCENARIOS.items():
        sim = simulate(df, **params)
        summary = scenario_summary(sim)
        logger.info(f"Scenario '{name}': {summary}")
