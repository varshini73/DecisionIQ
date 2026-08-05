"""
dashboard/pages/Inventory.py
===============================
ABC/XYZ segmentation and stock health -- the page a category manager uses
to see which SKUs deserve tight control vs. which can run on autopilot.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import DATA_PROCESSED_DIR

st.set_page_config(page_title="DecisionIQ | Inventory", layout="wide")
st.title("📦 Inventory Segmentation (ABC / XYZ)")

decisions = pd.read_parquet(DATA_PROCESSED_DIR / "decisions.parquet")

st.markdown(
    """
**ABC** = revenue contribution (A = top ~20% of SKUs driving ~80% of revenue).
**XYZ** = demand variability (X = stable/predictable, Z = erratic).

Combined, these tell you where to spend review effort: **AZ items**
(high revenue, unpredictable demand) deserve the most attention;
**CX items** can run on a simple reorder rule with minimal oversight.
"""
)

col1, col2 = st.columns(2)
with col1:
    abc_counts = decisions["abc_class"].value_counts().reset_index()
    abc_counts.columns = ["class", "count"]
    fig1 = px.pie(abc_counts, names="class", values="count", title="ABC Distribution (SKU count)")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    xyz_counts = decisions["xyz_class"].value_counts().reset_index()
    xyz_counts.columns = ["class", "count"]
    fig2 = px.pie(xyz_counts, names="class", values="count", title="XYZ Distribution (SKU count)")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("ABC × XYZ Matrix — Where to Focus Attention")
matrix = decisions.groupby(["abc_class", "xyz_class"], observed=True).size().reset_index(name="count")
pivot = matrix.pivot(index="abc_class", columns="xyz_class", values="count").fillna(0)
st.dataframe(pivot, use_container_width=True)

st.subheader("Stock Health by ABC Class")
health = decisions.groupby("abc_class")["status_light"].value_counts(normalize=True).unstack().fillna(0)
fig3 = px.bar(health, barmode="stack", title="Share of RED/YELLOW/GREEN status per ABC class",
              color_discrete_map={"RED": "#e74c3c", "YELLOW": "#f1c40f", "GREEN": "#2ecc71"})
st.plotly_chart(fig3, use_container_width=True)
