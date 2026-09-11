"""
explainability.py
====================
Explainable AI layer. Instead of surfacing "forecast = 450 units", this
module surfaces the top drivers behind that number using SHAP values on the
trained GBM model, e.g.:

    Top reasons:
      Weekend           (+18%)
      Promotion         (+11%)
      Holiday           (+8%)

This is what separates a decision-support recommendation from a bare
prediction: the user can see *why*, not just *what*.

Run standalone:
    python src/explainability.py
"""

import numpy as np
import pandas as pd
import joblib
from utils import DATA_PROCESSED_DIR, MODELS_DIR, get_logger

logger = get_logger("explainability")

FEATURE_COLS = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_14", "roll_mean_30",
    "roll_std_7", "roll_std_14", "roll_std_30",
    "day_of_week", "week_of_year", "month", "is_weekend",
    "is_month_start", "is_month_end",
    "price_vs_category_avg", "discount_pct", "promotion",
    "days_of_supply", "recent_stockout_flag",
]

# Human-readable labels for feature names, used when composing explanations
FEATURE_LABELS = {
    "is_weekend": "Weekend",
    "promotion": "Promotion",
    "discount_pct": "Discount",
    "roll_mean_7": "Recent 7-day sales trend",
    "roll_mean_14": "Recent 14-day sales trend",
    "roll_mean_30": "Recent 30-day sales trend",
    "lag_7": "Same day last week",
    "lag_28": "Same day last month",
    "price_vs_category_avg": "Relative price vs. category",
    "days_of_supply": "Current stock runway",
    "recent_stockout_flag": "Recent stockout history",
    "month": "Seasonality (month)",
    "week_of_year": "Seasonality (week)",
    "is_month_start": "Start-of-month effect",
    "is_month_end": "End-of-month effect",
}


def load_model():
    for fname in ["demand_model.joblib", "lightgbm_demand_model.pkl", "xgboost_demand_model.pkl"]:
        path = MODELS_DIR / fname
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(
        "No trained model found in models/. Run demand_forecasting.py first."
    )


def compute_shap_values(model, X: pd.DataFrame):
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed -- falling back to model feature_importances_")
        return None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values


def top_drivers_for_row(model, X_row: pd.DataFrame, n: int = 3) -> list:
    """
    Returns the top-n feature drivers for a single prediction as
    [{"feature": "Weekend", "impact_pct": 18.0}, ...] sorted by |impact|.
    Falls back to global feature importances if SHAP isn't installed.
    """
    shap_values = compute_shap_values(model, X_row)

    if shap_values is not None:
        row_shap = shap_values[0] if shap_values.ndim > 1 else shap_values
        base_pred = max(abs(row_shap).sum(), 1e-9)
        contributions = list(zip(X_row.columns, row_shap))
    else:
        importances = getattr(model, "feature_importances_", np.ones(len(X_row.columns)))
        base_pred = max(importances.sum(), 1e-9)
        contributions = list(zip(X_row.columns, importances))

    contributions = sorted(contributions, key=lambda kv: abs(kv[1]), reverse=True)[:n]

    drivers = []
    for feat, val in contributions:
        label = FEATURE_LABELS.get(feat, feat)
        impact_pct = 100 * val / base_pred
        drivers.append({"feature": label, "impact_pct": round(float(impact_pct), 1)})
    return drivers


def explain_all(sample_n: int = None) -> pd.DataFrame:
    """
    Computes top drivers for every (or a sample of) row in the latest
    feature table and saves an explanations table the decision engine
    and dashboard can join on (store_id, sku_id, date).
    """
    model = load_model()
    df = pd.read_parquet(DATA_PROCESSED_DIR / "features.parquet").dropna(subset=FEATURE_COLS)
    if sample_n:
        df = df.sample(min(sample_n, len(df)), random_state=42)

    X = df[FEATURE_COLS]
    shap_values = compute_shap_values(model, X)

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        if shap_values is not None:
            row_shap = shap_values[i]
            base_pred = max(abs(row_shap).sum(), 1e-9)
            contributions = sorted(
                zip(FEATURE_COLS, row_shap), key=lambda kv: abs(kv[1]), reverse=True
            )[:3]
        else:
            importances = getattr(model, "feature_importances_", np.ones(len(FEATURE_COLS)))
            base_pred = max(importances.sum(), 1e-9)
            contributions = sorted(
                zip(FEATURE_COLS, importances), key=lambda kv: abs(kv[1]), reverse=True
            )[:3]

        drivers = [
            f"{FEATURE_LABELS.get(f, f)} ({100*v/base_pred:+.0f}%)" for f, v in contributions
        ]
        records.append({
            "store_id": row["store_id"], "sku_id": row["sku_id"], "date": row["date"],
            "top_drivers": "; ".join(drivers),
        })

    out = pd.DataFrame(records)
    out_path = DATA_PROCESSED_DIR / "explanations.parquet"
    out.to_parquet(out_path, index=False)
    logger.info(f"Saved SHAP-based explanations for {len(out):,} rows to {out_path}")
    return out


if __name__ == "__main__":
    explain_all(sample_n=2000)  # sample for speed; drop sample_n to run on full data
