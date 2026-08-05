"""
business_rules.py
====================
Pure, auditable business logic for turning inventory/forecast numbers into
decisions. Deliberately separated from decision_engine.py: decision_engine.py
orchestrates the pipeline (I/O, looping over rows, saving results),
business_rules.py holds the actual rules a business stakeholder would want
to review, sign off on, and change without touching pipeline code.

Every function here is a pure function: same inputs -> same outputs, no I/O.
This is what makes them independently unit-testable (see tests/test_decision_engine.py).
"""

from config import CONFIG


def classify_decision(inventory_level: float, reorder_point: float,
                       days_until_stockout: float, lead_time_days: float) -> str:
    """
    Core decision rule. Kept simple and auditable on purpose -- a black-box
    model here would undermine the trust a decision-support system needs.
    """
    if days_until_stockout <= lead_time_days:
        return "REORDER_NOW"
    elif inventory_level <= reorder_point:
        return "REORDER_SOON"
    elif inventory_level > reorder_point * CONFIG["overstock_multiplier"]:
        return "OVERSTOCKED_REDUCE"
    elif inventory_level > reorder_point * CONFIG["monitor_multiplier"]:
        return "MONITOR"
    return "HOLD"


def classify_urgency(decision: str, abc_class: str) -> str:
    """
    Urgency blends the decision type with ABC class -- the same stock
    shortfall on an A-item and a C-item should not carry equal priority.
    """
    if decision == "REORDER_NOW" and abc_class == "A":
        return "CRITICAL"
    elif decision == "REORDER_NOW":
        return "HIGH"
    elif decision == "REORDER_SOON" and abc_class in ("A", "B"):
        return "MEDIUM"
    elif decision == "OVERSTOCKED_REDUCE" and abc_class == "A":
        return "MEDIUM"
    return "LOW"


def classify_status_light(days_of_supply: float) -> str:
    """
    Executive traffic-light status for the KPI dashboard.
    RED = action required, YELLOW = watch, GREEN = healthy.
    """
    t = CONFIG["status_thresholds"]
    if days_of_supply < t["red_days_of_supply"]:
        return "RED"
    elif days_of_supply < t["yellow_days_of_supply"]:
        return "YELLOW"
    return "GREEN"


def recommended_quantity(decision: str, reorder_point: float,
                          inventory_level: float, eoq: float) -> float:
    if decision in ("REORDER_NOW", "REORDER_SOON"):
        shortfall = max(reorder_point - inventory_level, 0)
        return round(max(eoq, shortfall))
    elif decision == "OVERSTOCKED_REDUCE":
        excess = inventory_level - reorder_point
        return -round(max(excess, 0))  # negative = reduce by this much
    return 0


def priority_label(urgency: str) -> str:
    """Human-facing priority label used in the recommendation card format."""
    return {
        "CRITICAL": "HIGH",
        "HIGH": "HIGH",
        "MEDIUM": "MEDIUM",
        "LOW": "LOW",
    }.get(urgency, "LOW")


def build_reason(demand_change_pct: float, driver_notes: list) -> str:
    """
    Compose the one-line business reason shown on a recommendation card,
    e.g. "Weekend demand is expected to increase while supplier lead time
    is 5 days." driver_notes comes from explainability.py's top SHAP drivers.
    """
    parts = []
    if demand_change_pct is not None:
        direction = "increase" if demand_change_pct >= 0 else "decrease"
        parts.append(f"Demand expected to {direction} by {abs(demand_change_pct):.0f}%")
    if driver_notes:
        parts.append("driven mainly by " + ", ".join(driver_notes))
    return "; ".join(parts) if parts else "Based on current inventory trajectory."
