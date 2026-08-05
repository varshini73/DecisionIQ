"""
dashboard/pages/What_If.py
=============================
Interactive what-if simulator built on src/scenario_simulator.py.
Answers: "what if demand increases by X%?", "what if lead time doubles?",
"what if we target a different service level?" -- live.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import streamlit as st

from utils import DATA_PROCESSED_DIR
from scenario_simulator import simulate, scenario_summary, PRESET_SCENARIOS

st.set_page_config(page_title="DecisionIQ | What-If", layout="wide")
st.title("🧪 What-If Scenario Simulator")
st.caption("Adjust policy levers and see the live effect on safety stock, reorder point, and cost.")

decisions = pd.read_parquet(DATA_PROCESSED_DIR / "decisions.parquet")
policy = pd.read_parquet(DATA_PROCESSED_DIR / "inventory_policy.parquet")

st.subheader("Quick Presets")
preset_choice = st.selectbox(
    "Jump to a common scenario",
    ["Custom"] + list(PRESET_SCENARIOS.keys()),
)

if preset_choice != "Custom":
    params = PRESET_SCENARIOS[preset_choice]
    demand_uplift = params.get("demand_uplift_pct", 0)
    lead_time_mult = params.get("lead_time_multiplier", 1.0)
    service_level = params.get("service_level_pct", 95)
else:
    demand_uplift, lead_time_mult, service_level = 0, 1.0, 95

col_x, col_y, col_z = st.columns(3)
demand_uplift = col_x.slider("Demand Change (%)", -50, 100, int(demand_uplift))
lead_time_mult = col_y.slider("Lead Time Multiplier", 0.5, 3.0, float(lead_time_mult), step=0.1)
service_level = col_z.slider("Target Service Level (%)", 80, 99, int(service_level))

sim = simulate(
    policy,
    demand_uplift_pct=demand_uplift,
    lead_time_multiplier=lead_time_mult,
    service_level_pct=service_level,
)
summary = scenario_summary(sim)

col_r1, col_r2, col_r3 = st.columns(3)
col_r1.metric("Avg Safety Stock (before → after)",
              f"{summary['avg_safety_stock_before']:.0f} → {summary['avg_safety_stock_after']:.0f}")
col_r2.metric("Holding Cost Δ", f"${summary['total_holding_cost_delta']:,.0f}")
col_r3.metric("SKUs Newly at Risk", summary["skus_newly_at_risk"])

st.info(
    f"At **{demand_uplift:+d}% demand**, **{lead_time_mult}x lead time**, and "
    f"**{service_level}% service level**: average safety stock moves from "
    f"{summary['avg_safety_stock_before']:.0f} to {summary['avg_safety_stock_after']:.0f} units "
    f"per SKU-store, changing total holding cost by **${summary['total_holding_cost_delta']:,.0f}**, "
    f"with **{summary['skus_newly_at_risk']}** SKU-store combinations newly falling below their "
    "reorder cushion. This is the exact trade-off a supply chain lead weighs before changing policy."
)

st.subheader("Simulated Policy Table")
st.dataframe(
    sim[["store_id", "sku_id", "abc_class", "safety_stock", "sim_safety_stock",
         "reorder_point", "sim_reorder_point", "sim_days_until_stockout",
         "sim_holding_cost_delta", "sim_stockout_cost"]].sort_values(
        "sim_stockout_cost", ascending=False
    ),
    use_container_width=True,
    height=450,
)
