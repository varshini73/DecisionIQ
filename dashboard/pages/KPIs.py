"""
dashboard/pages/KPIs.py
==========================
Executive KPI page. Traffic-light framing: Green = healthy, Yellow = watch,
Red = action required. Built for a manager doing a 30-second scan, not a
data scientist reading model metrics.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import DATA_PROCESSED_DIR

st.set_page_config(page_title="DecisionIQ | KPIs", layout="wide")
st.title("📊 Executive KPI Overview")

decisions = pd.read_parquet(DATA_PROCESSED_DIR / "decisions.parquet")
try:
    comparison = pd.read_csv(DATA_PROCESSED_DIR / "model_comparison.csv")
except FileNotFoundError:
    comparison = pd.DataFrame()

STATUS_COLORS = {"RED": "#e74c3c", "YELLOW": "#f1c40f", "GREEN": "#2ecc71"}

col1, col2, col3, col4 = st.columns(4)
total_value = (decisions["inventory_level"] * 10).sum()
red_pct = (decisions["status_light"] == "RED").mean()
yellow_pct = (decisions["status_light"] == "YELLOW").mean()
wape = comparison.loc[comparison["model"].str.contains("Ensemble", na=False), "WAPE"] \
    if not comparison.empty else pd.Series(dtype=float)
wape_val = wape.values[0] if len(wape) else np.nan

col1.metric("Total Inventory Value (est.)", f"${total_value:,.0f}")
col2.metric("🔴 SKUs Requiring Action", f"{red_pct:.1%}")
col3.metric("🟡 SKUs to Watch", f"{yellow_pct:.1%}")
col4.metric("Forecast Accuracy (WAPE)", f"{wape_val:.1%}" if not np.isnan(wape_val) else "n/a")

st.subheader("Network Health — Traffic Light View")
status_counts = decisions["status_light"].value_counts().reindex(["RED", "YELLOW", "GREEN"]).fillna(0)
fig = px.bar(
    x=status_counts.index, y=status_counts.values,
    color=status_counts.index,
    color_discrete_map=STATUS_COLORS,
    labels={"x": "Status", "y": "# of SKU-Store Combinations"},
    title="How many SKUs are Red / Yellow / Green right now",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Decision Mix")
decision_counts = decisions["decision"].value_counts().reset_index()
decision_counts.columns = ["decision", "count"]
fig2 = px.bar(decision_counts, x="decision", y="count", color="decision",
              title="Recommended Actions Breakdown")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("$ at Risk by Priority")
cost_by_priority = decisions.groupby("priority")["estimated_stockout_cost"].sum().reset_index()
fig3 = px.bar(cost_by_priority, x="priority", y="estimated_stockout_cost", color="priority",
              title="Total $ at Risk if Recommendations Are Ignored")
st.plotly_chart(fig3, use_container_width=True)

if not comparison.empty:
    st.subheader("Model Comparison")
    st.dataframe(comparison, use_container_width=True)

st.subheader("Stores Ranked by Risk")
store_risk = decisions.groupby("store_id").agg(
    red_count=("status_light", lambda s: (s == "RED").sum()),
    total_cost_at_risk=("estimated_stockout_cost", "sum"),
).sort_values("total_cost_at_risk", ascending=False).reset_index()
st.dataframe(store_risk, use_container_width=True)
