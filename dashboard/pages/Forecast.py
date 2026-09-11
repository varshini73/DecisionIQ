"""
dashboard/pages/Forecast.py
==============================
Actual vs. forecast for any SKU-store, plus the SHAP-based top drivers
behind the latest forecast -- so the answer isn't just "450 units", it's
"450 units, mainly because of X, Y, Z."
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import DATA_PROCESSED_DIR

st.set_page_config(page_title="DecisionIQ | Forecast", layout="wide")
st.title("🔮 Forecast Explorer")

forecasts = pd.read_parquet(DATA_PROCESSED_DIR / "forecast_results.parquet")
try:
    explanations = pd.read_parquet(DATA_PROCESSED_DIR / "explanations.parquet")
except FileNotFoundError:
    explanations = pd.DataFrame()

col_a, col_b = st.columns(2)
store_choice = col_a.selectbox("Store", sorted(forecasts["store_id"].unique()))
sku_options = sorted(forecasts.loc[forecasts["store_id"] == store_choice, "sku_id"].unique())
sku_choice = col_b.selectbox("SKU", sku_options)

series = forecasts[
    (forecasts["store_id"] == store_choice) & (forecasts["sku_id"] == sku_choice)
].sort_values("date")

fig = go.Figure()
fig.add_trace(go.Scatter(x=series["date"], y=series["actual_demand"],
                          name="Actual (demand-corrected)", mode="lines"))
fig.add_trace(go.Scatter(x=series["date"], y=series["forecast_demand"],
                          name="Forecast", mode="lines", line=dict(dash="dash")))
fig.update_layout(title=f"Demand vs. Forecast — {sku_choice} @ {store_choice}",
                   xaxis_title="Date", yaxis_title="Units")
st.plotly_chart(fig, use_container_width=True)

if len(series):
    st.caption(
        f"ABC class: **{series['abc_class'].iloc[-1]}** | "
        f"XYZ class: **{series['xyz_class'].iloc[-1]}** | "
        f"Latest inventory level: **{series['inventory_level'].iloc[-1]:.0f} units**"
    )

st.subheader("🧠 Why this forecast (SHAP top drivers)")
if not explanations.empty:
    row = explanations[
        (explanations["store_id"] == store_choice) & (explanations["sku_id"] == sku_choice)
    ].sort_values("date").tail(1)
    if len(row):
        drivers = row["top_drivers"].values[0]
        st.success(f"Top reasons: {drivers}")
    else:
        st.caption("No SHAP explanation available for this SKU-store yet.")
else:
    st.caption(
        "Run `python src/explainability.py` to generate SHAP-based driver "
        "explanations for the forecast."
    )
