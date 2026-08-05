"""
config.py
===========
Single source of truth for business assumptions and pipeline parameters.
Kept separate from utils.py deliberately: in a real consulting engagement,
this is the file the client stakeholder reviews and edits -- it should never
be buried inside logging/path helpers.

Every number here should be traceable to a business conversation, e.g.:
  "Ops confirmed 95% service level is the target for A-items."
  "Procurement confirmed average supplier lead time is 7 days."
"""

CONFIG = {
    # --- Service policy ---
    "service_level_z": 1.65,        # ~95% service level (z-score)
    "service_level_pct": 95,

    # --- Supply chain assumptions ---
    "default_lead_time_days": 7,
    "ordering_cost_per_order": 50.0,     # $ per purchase order placed
    "holding_cost_rate": 0.20,           # annual holding cost as % of unit cost
    "stockout_margin_loss_rate": 0.30,   # assumed margin % lost per stockout unit-day

    # --- Inventory segmentation ---
    "abc_thresholds": {"A": 0.80, "B": 0.95},   # cumulative revenue share cutoffs
    "xyz_thresholds": {"X": 0.5, "Y": 1.0},     # coefficient-of-variation cutoffs

    # --- Modeling ---
    "forecast_horizon_days": 28,
    "test_days": 56,
    "validation_days": 56,
    "random_seed": 42,

    # --- Decision engine thresholds ---
    "overstock_multiplier": 2.5,     # inventory > ROP * this => OVERSTOCKED_REDUCE
    "monitor_multiplier": 1.5,       # inventory > ROP * this => MONITOR

    # --- Executive dashboard status thresholds (traffic light) ---
    "status_thresholds": {
        "red_days_of_supply": 5,     # < 5 days of supply => RED (action required)
        "yellow_days_of_supply": 12, # < 12 days of supply => YELLOW (watch)
        # >= yellow threshold => GREEN
    },
}
