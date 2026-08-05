"""
dashboard/pages/Recommendations.py
=====================================
The decision engine's full output, shown both as a filterable table and as
individual recommendation cards in the exact "Product / Forecast / Current
Stock / Recommended Action / Reason / Priority" format used in the project
write-up.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import streamlit as st

from utils import DATA_PROCESSED_DIR

st.set_page_config(page_title="DecisionIQ | Recommendations", layout="wide")
st.title("✅ Decision Recommendations")

decisions = pd.read_parquet(DATA_PROCESSED_DIR / "decisions.parquet")

col_f1, col_f2, col_f3 = st.columns(3)
priority_filter = col_f1.multiselect("Priority", decisions["priority"].unique(),
                                      default=list(decisions["priority"].unique()))
decision_filter = col_f2.multiselect("Decision Type", decisions["decision"].unique(),
                                      default=list(decisions["decision"].unique()))
abc_filter = col_f3.multiselect("ABC Class", decisions["abc_class"].dropna().unique(),
                                 default=list(decisions["abc_class"].dropna().unique()))

filtered = decisions[
    decisions["priority"].isin(priority_filter)
    & decisions["decision"].isin(decision_filter)
    & decisions["abc_class"].isin(abc_filter)
]

view_mode = st.radio("View as", ["Table", "Recommendation Cards"], horizontal=True)

if view_mode == "Table":
    st.dataframe(
        filtered[[
            "store_id", "sku_id", "abc_class", "decision", "priority", "status_light",
            "inventory_level", "recommended_qty", "days_until_stockout",
            "estimated_stockout_cost", "explanation",
        ]],
        use_container_width=True,
        height=550,
    )
    st.download_button("Download recommendations as CSV",
                        filtered.to_csv(index=False), "decisioniq_recommendations.csv")
else:
    top_n = st.slider("Number of cards to show", 5, 50, 10)
    for _, row in filtered.head(top_n).iterrows():
        action = (
            f"Reorder {row['recommended_qty']:.0f} units" if row["recommended_qty"] > 0
            else (f"Reduce stock by {abs(row['recommended_qty']):.0f} units"
                  if row["recommended_qty"] < 0 else "No action")
        )
        priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(row["priority"], "⚪")
        with st.container(border=True):
            st.markdown(f"### {priority_emoji} {row['sku_id']} — {row['store_id']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Stock", f"{row['inventory_level']:.0f} units")
            c2.metric("Recommended Action", action)
            c3.metric("Priority", row["priority"])
            st.caption(f"**Reason:** {row['explanation']}")
