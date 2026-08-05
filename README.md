# DecisionIQ

**AI-Powered Retail Inventory Optimization & Decision Support System**

Helping retailers answer:
- **What should we stock?**
- **When should we reorder?**
- **How much should we order?**
- **Which stores are at risk of stockouts?**
- **How can inventory cost be reduced?**

DecisionIQ is a Decision Intelligence system, not just a forecasting model.
It takes raw sales data and turns it into specific business decisions —
how much to reorder, which SKUs are at risk of stockout, which are
overstocked, and *why* — with every number traceable to a dollar impact
and every recommendation explainable via SHAP.

---

## 1. Why This Project Is Different From a Typical ML Project

| Typical ML Project | DecisionIQ |
|---|---|
| Predicts sales/demand | Predicts demand **and** converts it into an action |
| Stops at RMSE/MAE | Stops at "reorder 340 units of SKU-1042 at Store 12 by Friday" |
| One model, one metric | Forecasting model + optimization layer + rule-based decision engine |
| Static notebook | Interactive dashboard with what-if simulation |
| "Here's the prediction" | "Here's the prediction, here's why, here's what it costs if you ignore it" |

This mirrors how firms like Mu Sigma actually operate: they call themselves a
**"Decision Sciences"** company, not a machine learning shop. The deliverable
a client cares about is a decision, and the model is just the engine underneath it.

---

## 2. Business Problem

Retailers with many SKUs across many stores face four repeating decisions:

1. **How much inventory to stock** for each SKU-store combination.
2. **Which products need reordering now** (before they stock out).
3. **Which products are overstocked** (tying up capital, at risk of markdown/waste).
4. **Which stores are at the highest risk of stockouts** this week/month.

Bad decisions here cost money on both sides:
- Understocking → lost sales, poor customer experience, market share loss.
- Overstocking → holding cost, capital lock-up, obsolescence/markdown losses.

DecisionIQ's job is to make each of those four decisions automatically,
with a number attached to it and a plain-English reason.

---

## 3. System Architecture

```
                     ┌────────────────────┐
                     │   Raw Sales Data    │
                     │ (SKU, Store, Date)  │
                     └─────────┬──────────┘
                               │
                     ┌─────────▼──────────┐
                     │ Data Preprocessing  │  (cleaning, missing values,
                     │  (src/data_         │   outlier handling, calendar
                     │  preprocessing.py)  │   alignment)
                     └─────────┬──────────┘
                               │
                     ┌─────────▼──────────┐
                     │ Feature Engineering │  (lags, rolling stats,
                     │ (src/feature_       │   seasonality, price/promo,
                     │  engineering.py)    │   ABC class, stockout flags)
                     └─────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
   ┌──────────▼─────────┐  ┌──▼───────────┐  ┌──▼────────────────┐
   │ Demand Forecasting │  │  ABC / XYZ    │  │  Safety Stock &    │
   │ (XGBoost/LightGBM/ │  │  Segmentation │  │  Reorder Point Calc│
   │  Prophet ensemble) │  │               │  │  (src/inventory_   │
   │ src/demand_         │  │               │  │   optimization.py) │
   │ forecasting.py      │  │               │  │                    │
   └──────────┬─────────┘  └──┬───────────┘  └──┬────────────────┘
              │                │                 │
              └────────────────┼─────────────────┘
                               │
                     ┌─────────▼──────────┐
                     │  Decision Engine    │  (reorder qty, stockout risk,
                     │ (src/decision_      │   overstock flag, cost impact,
                     │  engine.py)         │   explanation generator)
                     └─────────┬──────────┘
                               │
                     ┌─────────▼──────────┐
                     │   SQLite Database   │  (persisted forecasts,
                     │  (src/database.py)  │   recommendations, KPIs)
                     └─────────┬──────────┘
                               │
                     ┌─────────▼──────────┐
                     │  Streamlit Dashboard│  (KPIs, forecasts, what-if
                     │  (dashboard/app.py) │   simulator, recommendations)
                     └────────────────────┘
```

**Design principle:** every module has one job and produces a clean, typed
output the next module consumes. This is exactly how a production decision
system is built — not a single monolithic notebook.

---

## 4. Folder Structure

```
DecisionIQ/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   ├── raw/                       ← original downloaded dataset (gitignored)
│   ├── processed/                 ← cleaned + feature-engineered data (gitignored)
│   ├── external/                  ← optional external/reference data (e.g. holiday calendars)
│   └── generate_synthetic_data.py ← generates realistic sample data if not using Kaggle CSVs
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Comparison.ipynb
│   └── 04_Business_Insights.ipynb
│
├── src/
│   ├── utils.py                   ← paths, logging (plumbing)
│   ├── config.py                  ← business assumptions (service level, costs, thresholds)
│   ├── data_preprocessing.py      ← cleaning, missing values, calendar reindex
│   ├── feature_engineering.py     ← lags, rolling windows, ABC/XYZ, demand un-censoring
│   ├── demand_forecasting.py      ← trains & compares Naive/XGBoost/LightGBM/ensemble
│   ├── inventory_optimization.py  ← safety stock, reorder point, EOQ, costs
│   ├── decision_engine.py         ← orchestrates rules + explanations into recommendations
│   ├── explainability.py          ← SHAP-based "why this forecast" driver extraction
│   ├── scenario_simulator.py      ← reusable what-if logic (demand/lead-time/service-level)
│   ├── business_rules.py          ← pure, testable decision rules (the auditable layer)
│   └── database.py                ← SQLite schema + read/write helpers
│
├── models/                        ← saved trained models (.pkl / .json), gitignored
│
├── dashboard/
│   ├── app.py                     ← landing page / shared config
│   ├── pages/
│   │   ├── KPIs.py                ← executive traffic-light overview
│   │   ├── Forecast.py            ← actual vs. forecast + SHAP drivers
│   │   ├── Inventory.py           ← ABC/XYZ segmentation
│   │   ├── Recommendations.py     ← decision engine output (table + cards)
│   │   └── What_If.py             ← scenario simulator UI
│   └── assets/
│
├── reports/                       ← exported PDF summaries (generated, gitignored)
│
├── tests/
│   ├── test_forecasting.py        ← WAPE/RMSE/Bias metric tests
│   ├── test_inventory.py          ← safety stock/ROP/EOQ formula tests
│   └── test_decision_engine.py    ← business_rules.py rule tests
│
└── docs/
    ├── Architecture.md
    ├── Business_Case.md
    ├── Resume_Description.md
    ├── LinkedIn_Post.md
    └── Future_Improvements.md
```

---

## 5. Dataset Recommendation

Use a synthetic multi-store, multi-SKU retail dataset with daily granularity,
inventory levels, price, and promotions. Recommended, in order of fit:

1. **Retail Store Inventory Forecasting Dataset** (primary recommendation)
   https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset
   Has Store ID, Product ID, Category, Region, Date, Inventory Level, Units
   Sold, Units Ordered, Price, Discount, Promotion flag, Weather, Competitor
   pricing, Seasonality — this is close to ideal because it already has
   inventory-level fields, not just sales, which lets you build the
   stockout/overstock logic realistically.

2. **Inventory Demand Forecasting Dataset**
   https://www.kaggle.com/datasets/mirzayasirabdullah07/inventory-demand-forecasting-dataset
   Good alternative with multi-store demand + pricing fields.

3. **Store Item Demand Forecasting Challenge (Kaggle competition)**
   https://www.kaggle.com/competitions/demand-forecasting-kernels-only
   10 stores × 50 items, 5 years daily — excellent for the pure forecasting
   layer if you want a clean, well-known benchmark to compare model scores
   against public leaderboard notebooks (good for your write-up).

**Recommendation:** Start with #1 for the full inventory optimization pipeline
since it already contains inventory levels. If you want a cleaner, larger
history purely for forecasting-model comparison, blend in #3.

---

## 6. Data Preprocessing

Implemented in `src/data_preprocessing.py`:

1. **Load & validate schema** — enforce dtypes (dates parsed, IDs as category).
2. **Missing values** — forward-fill within a SKU-store series for short gaps;
   flag and drop series with >30% missing history.
3. **Duplicate handling** — dedupe on (Store, SKU, Date).
4. **Outlier treatment** — cap sales outliers using IQR per SKU-store group
   (not global — a spike that's an outlier for one SKU is normal for another).
5. **Calendar alignment** — reindex every SKU-store series to a continuous
   daily calendar, filling non-selling days with 0 (true zero-demand,
   important — don't drop them, they matter for demand distribution).
6. **Train/validation/test split** — **time-based split** (never random
   shuffle for time series): last 8 weeks = test, prior 8 weeks = validation.

---

## 7. Feature Engineering

Implemented in `src/feature_engineering.py`:

- **Lag features**: sales lag 1, 7, 14, 28 days.
- **Rolling statistics**: 7/14/30-day rolling mean, std, min, max of sales.
- **Calendar features**: day-of-week, week-of-year, month, is_weekend,
  is_month_start/end, holiday flag.
- **Trend/seasonality decomposition** features (from Prophet or STL) as inputs
  to the tree-based models — this is a common Mu-Sigma-style trick: use
  a time-series decomposition as a *feature generator* for a gradient-boosted
  model rather than treating it as competing with the ML model.
- **Price & promo features**: discount %, promo flag, price relative to
  category average.
- **Inventory features**: days-of-supply at time t, prior stockout flag
  (was this SKU-store out of stock in the last 14 days — censors demand!).
- **ABC/XYZ classification** (see below) as a categorical feature.

**Important nuance to mention in interviews:** stockouts *censor* demand —
if a product was out of stock, "units sold = 0" is not "demand = 0". The
preprocessing pipeline flags stockout days and either excludes them from
demand-model training or applies a demand-uncensoring correction
(replacing censored days with the rolling average of non-stockout days).
This is exactly the kind of subtlety that separates a decision-science
project from a plain Kaggle notebook.

---

## 8. ABC Inventory Analysis

Classic 80/20 inventory segmentation, implemented in
`inventory_optimization.py::abc_classification()`:

- **A items**: top ~20% of SKUs by revenue contribution → ~70-80% of revenue.
  Tight control, frequent review, low tolerance for stockouts.
- **B items**: next ~30% of SKUs → ~15-20% of revenue. Moderate control.
- **C items**: remaining ~50% of SKUs → ~5% of revenue. Loose control,
  higher acceptable stockout risk, order in bulk infrequently.

This classification feeds directly into the decision engine: an A-item
stockout risk gets flagged as **high priority**, a C-item overstock is
**low priority** even if the raw numbers look similar.

---

## 9. Models: Selection & Comparison

Implemented in `src/demand_forecasting.py`. Trained per SKU-store or per
SKU-category cluster (clustering reduces #models from thousands to dozens):

| Model | Role | Why included |
|---|---|---|
| **Naive / seasonal naive** | Baseline | Mandatory baseline — if your model can't beat "same day last week," it's not adding value. Always report this. |
| **Prophet** | Captures trend + seasonality + holidays cleanly | Good for interpretability, handles missing data and holiday effects out of the box. Weak on cross-SKU signal. |
| **LightGBM / XGBoost** | Primary model | Handles lag/rolling/price/promo features jointly across SKUs, usually wins on MAPE/WAPE in practice. |
| **Ensemble (avg of Prophet + LightGBM)** | Final production choice | Blends trend-stability of Prophet with feature-richness of GBM; typically reduces variance of errors. |

**Evaluation metrics** (reported per model, per ABC class — not just overall):
- **MAPE / WAPE** (Weighted Absolute Percentage Error — preferred over MAPE
  for retail since it doesn't blow up on near-zero-demand SKUs)
- **RMSE** (penalizes large misses — relevant for A-items)
- **Bias** (mean error — are we systematically over/under forecasting?)
- **Forecast Value Added (FVA)** — accuracy gain vs. the naive baseline.
  This metric is the single most "business-fluent" metric to report: it
  directly answers "is this model worth building?"

---

## 10. Inventory Optimization Layer

Implemented in `src/inventory_optimization.py`. This is what makes the
project *decision intelligence* instead of *forecasting*:

- **Safety Stock** = `z * σ_demand * sqrt(lead_time)`
  where `z` is the service-level factor (e.g., 1.65 for 95% service level),
  and `σ_demand` is forecast error std-dev (not raw demand std-dev — using
  forecast *error* is more accurate and is a detail worth calling out).
- **Reorder Point (ROP)** = `(avg_daily_demand * lead_time) + safety_stock`
- **Economic Order Quantity (EOQ)** = `sqrt((2 * D * S) / H)`
  where D = annual demand, S = ordering cost, H = holding cost per unit/year.
- **Reorder Quantity Recommendation** = `max(EOQ, ROP - current_inventory)`
- **Inventory Cost Calculation**: holding cost + ordering cost + stockout
  cost (lost margin), so every recommendation carries a dollar figure.

---

## 11. Decision Recommendation Engine

Implemented in `src/decision_engine.py`. For every SKU-store row, it outputs:

```json
{
  "sku": "SKU-1042",
  "store": "Store-12",
  "decision": "REORDER_NOW",
  "recommended_qty": 340,
  "urgency": "HIGH",
  "abc_class": "A",
  "days_until_stockout": 3,
  "estimated_cost_of_inaction": 4200.00,
  "explanation": "Forecasted demand (58 units/day) exceeds current inventory "
                  "runway of 3 days. This is an A-class SKU (top revenue "
                  "contributor) with a 12-day lead time, so ordering must "
                  "happen today to avoid a stockout that would cost an "
                  "estimated $4,200 in lost margin."
}
```

Decision categories: `REORDER_NOW`, `REORDER_SOON`, `HOLD`, `OVERSTOCKED_REDUCE`,
`MONITOR`. Every decision is generated from explicit, auditable rules on top
of the model outputs — this is the "explainability" layer, deliberately kept
rule-based (not a black box) because in real decision-science consulting,
the client needs to trust *and verify* the recommendation logic.

---

## 12. Dashboard Design (Streamlit)

`dashboard/app.py` — four tabs:

1. **Executive KPI Overview**: total inventory value, stockout rate,
   overstock rate, forecast accuracy (WAPE), inventory turnover — big
   number cards + trend sparkline, Plotly.
2. **Forecast Explorer**: pick SKU + store, see actual vs. forecast vs.
   confidence band, plus the ABC class and current inventory level overlaid.
3. **Recommendation Table**: sortable/filterable table of every decision
   engine output (urgency, cost of inaction, explanation) — this is the
   page a category manager would actually use daily.
4. **What-If Simulator**: sliders for lead time, service level target,
   promo uplift % — recompute safety stock / ROP / cost live, so the user
   can see "if I raise service level from 95% to 99%, holding cost goes up
   by $X but stockout risk drops by Y%." This tab is what turns the project
   from "a dashboard" into "a decision-support tool."

---

## 13. Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost/LightGBM · Prophet (optional)
· Streamlit · Plotly · SQLite

---

## 14. How to Run

```bash
pip install -r requirements.txt

# 1. Get data -- download the recommended Kaggle dataset into data/raw/,
#    OR generate realistic synthetic data:
python data/generate_synthetic_data.py

# 2. Run the pipeline in order
python src/data_preprocessing.py
python src/feature_engineering.py
python src/demand_forecasting.py
python src/inventory_optimization.py
python src/explainability.py        # optional but recommended (SHAP drivers)
python src/decision_engine.py
python src/database.py              # optional: persist to SQLite

# 3. Run tests
pytest tests/ -v

# 4. Launch the dashboard (multi-page: KPIs, Forecast, Inventory,
#    Recommendations, What-If all appear in the sidebar)
streamlit run dashboard/app.py
```

---

## 15. Explainability, Executive View & Scenario Simulation

These three additions push the project from "forecasting demo" to
"decision support system":

- **Explainability (`src/explainability.py`)** — every forecast is backed
  by its top SHAP drivers, e.g. *"Top reasons: Weekend (+18%), Promotion
  (+11%), Holiday (+8%)"* — shown on the Forecast dashboard page. Falls
  back to model feature importances if SHAP isn't installed.
- **Executive traffic-light view (`dashboard/pages/KPIs.py`)** — every
  SKU-store gets a RED (action required) / YELLOW (watch) / GREEN
  (healthy) status based on days-of-supply, so a manager can scan network
  health in seconds rather than reading a table.
- **What-if simulator (`src/scenario_simulator.py` +
  `dashboard/pages/What_If.py`)** — "what if demand increases by 25%?",
  "what if lead time doubles?", "what if we tighten to a 99% service
  level?" — recomputed live, with the resulting safety stock and holding
  cost delta shown immediately.

---

## 16. Future Improvements

See `docs/Future_Improvements.md`.
