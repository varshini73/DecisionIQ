"""
dashboard/app.py
==================
Main entry point. With Streamlit's multi-page structure, individual pages
live in dashboard/pages/ and appear automatically in the sidebar. This file
sets shared page config and shows a landing/overview screen.

Run:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st

st.set_page_config(page_title="DecisionIQ", layout="wide", page_icon="📦")

st.title("📦 DecisionIQ")
st.subheader("AI-Powered Retail Inventory Optimization & Decision Support System")

st.markdown(
    """
Helping retailers answer:
- **What should we stock?**
- **When should we reorder?**
- **How much should we order?**
- **Which stores are at risk of stockouts?**
- **How can inventory cost be reduced?**

Use the sidebar to navigate:

| Page | Purpose |
|---|---|
| **KPIs** | Executive overview, green/yellow/red status across the network |
| **Forecast** | Actual vs. forecast for any SKU-store, with SHAP driver breakdown |
| **Inventory** | ABC/XYZ segmentation and stock health |
| **Recommendations** | The decision engine's full recommendation list |
| **What-If** | Simulate demand shocks, lead-time changes, and service-level trade-offs |
"""
)

st.info(
    "This is a decision support system, not just a forecasting demo. Every "
    "page is built to answer a specific operational question a category "
    "manager or supply chain lead would actually ask."
)
