# Resume Description

## Option 1 (concise, 3 bullets)

**DecisionIQ: AI-Powered Retail Inventory Optimization** | Personal Project
- Built an end-to-end decision intelligence system that forecasts SKU-store
  level demand (LightGBM/XGBoost + Prophet ensemble) and converts it into
  reorder-quantity, safety-stock, and stockout-risk recommendations —
  reducing forecast error by [X]% over a seasonal-naive baseline (WAPE).
- Designed a rule-based decision engine that classifies every SKU-store
  combination into REORDER_NOW / REORDER_SOON / OVERSTOCKED / HOLD with an
  auto-generated business explanation and estimated $ cost of inaction.
- Shipped an interactive Streamlit dashboard (Plotly) with a what-if
  simulator letting users test service-level and lead-time trade-offs
  against holding cost in real time; backed by a SQLite persistence layer.

## Option 2 (single line)

Built DecisionIQ, an end-to-end decision-intelligence pipeline (demand
forecasting → safety stock/EOQ optimization → rule-based recommendation
engine → Streamlit dashboard) that turns retail sales data into
actionable reorder decisions with explainable, cost-quantified outputs.

## Tips for filling in the [X]%
Run `src/demand_forecasting.py` on your chosen dataset and pull the
`FVA_vs_baseline` column from `data/processed/model_comparison.csv` —
that number is literally designed to answer "how much better is my model
than doing nothing," which is exactly what a resume bullet needs.
