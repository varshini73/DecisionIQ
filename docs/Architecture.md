# Architecture

## System Overview

```
Raw Sales Data
      │
      ▼
Data Preprocessing  (src/data_preprocessing.py)
  - schema enforcement, dedup, continuous calendar reindex
  - per-series outlier capping, low-history filtering
      │
      ▼
Feature Engineering  (src/feature_engineering.py)
  - demand un-censoring for stockout days
  - lags, rolling stats, calendar, price/promo, ABC/XYZ
      │
      ├─────────────────────────────┬────────────────────────────┐
      ▼                              ▼                            ▼
Demand Forecasting          Inventory Optimization        Explainability
(src/demand_forecasting.py)  (src/inventory_             (src/explainability.py)
  - naive / LightGBM /       optimization.py)              - SHAP top-driver
    XGBoost / ensemble         - safety stock, ROP, EOQ,     extraction per
  - WAPE / RMSE / Bias /       cost                          forecast
    FVA metrics
      │                              │                            │
      └──────────────┬───────────────┴──────────────┬─────────────┘
                      ▼                              ▼
              Business Rules                 Scenario Simulator
              (src/business_rules.py)         (src/scenario_simulator.py)
                - pure, testable rules          - demand/lead-time/service-
                - decision/urgency/status          level what-if recompute
                  classification
                      │
                      ▼
              Decision Engine  (src/decision_engine.py)
                - orchestrates rules + explanations into one
                  recommendation per SKU-store
                - outputs structured cards (Product / Forecast /
                  Current Stock / Recommended Action / Reason / Priority)
                      │
                      ▼
              SQLite Database  (src/database.py)
                      │
                      ▼
              Streamlit Dashboard  (dashboard/app.py + pages/)
                - KPIs (traffic light) | Forecast | Inventory |
                  Recommendations | What-If
```

## Why the code is split this way

**`config.py` vs. `utils.py`** — `config.py` holds every business assumption
(service level, lead time, cost rates, thresholds) in one place a
non-engineer stakeholder could review line by line. `utils.py` holds paths
and logging — plumbing nobody outside the dev team needs to see.

**`business_rules.py` vs. `decision_engine.py`** — the rules that decide
`REORDER_NOW` vs. `HOLD` are pure functions with no I/O, so they can be
unit-tested in isolation (see `tests/test_decision_engine.py`) and reviewed
by a business stakeholder without wading through pipeline orchestration
code. `decision_engine.py` is the "glue": it loads data, calls the rules,
joins in explanations, and writes results.

**`explainability.py` as its own module** — explainability is treated as a
first-class deliverable, not an afterthought bolted onto the forecasting
script. It can be run independently, cached separately (SHAP is expensive),
and swapped out (e.g., for LIME) without touching the forecasting code.

**`scenario_simulator.py` decoupled from the dashboard** — the same
what-if logic is callable from the dashboard, from a notebook, from a test,
or eventually from an API — it shouldn't be trapped inside Streamlit
callback code.

## Data flow contracts (what each stage promises the next)

| File produced | By | Guarantees |
|---|---|---|
| `data/processed/clean_panel.parquet` | `data_preprocessing.py` | One row per (store, sku, date), no gaps, no duplicates |
| `data/processed/features.parquet` | `feature_engineering.py` | All lag/rolling/calendar features populated, ABC/XYZ assigned |
| `data/processed/forecast_results.parquet` | `demand_forecasting.py` | `forecast` column aligned to `demand_est` for every test-period row |
| `data/processed/inventory_policy.parquet` | `inventory_optimization.py` | Safety stock, ROP, EOQ, costs computed per current SKU-store snapshot |
| `data/processed/explanations.parquet` | `explainability.py` | Top-3 SHAP drivers per SKU-store-date (optional but recommended) |
| `data/processed/decisions.parquet` / `.csv` | `decision_engine.py` | Final recommendation per SKU-store, ready for the dashboard or export |

This contract-based design is what lets each module be developed, tested,
and demoed independently — exactly how a consulting analytics team would
divide the work across people.
