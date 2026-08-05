"""
tests/test_decision_engine.py
Unit tests for the pure rule functions in business_rules.py -- these are
the rules a business stakeholder should be able to read and sign off on,
so they're tested independently of the pipeline orchestration.
Run: pytest tests/
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from business_rules import (
    classify_decision, classify_urgency, classify_status_light,
    recommended_quantity, priority_label,
)


def test_reorder_now_when_stockout_imminent():
    decision = classify_decision(
        inventory_level=20, reorder_point=50, days_until_stockout=2, lead_time_days=7
    )
    assert decision == "REORDER_NOW"


def test_reorder_soon_when_below_rop_but_not_urgent():
    decision = classify_decision(
        inventory_level=40, reorder_point=50, days_until_stockout=20, lead_time_days=7
    )
    assert decision == "REORDER_SOON"


def test_overstocked_when_inventory_far_above_rop():
    decision = classify_decision(
        inventory_level=300, reorder_point=50, days_until_stockout=60, lead_time_days=7
    )
    assert decision == "OVERSTOCKED_REDUCE"


def test_hold_when_inventory_is_healthy():
    decision = classify_decision(
        inventory_level=60, reorder_point=50, days_until_stockout=30, lead_time_days=7
    )
    assert decision == "HOLD"


def test_urgency_critical_for_a_class_reorder_now():
    assert classify_urgency("REORDER_NOW", "A") == "CRITICAL"


def test_urgency_high_for_non_a_reorder_now():
    assert classify_urgency("REORDER_NOW", "C") == "HIGH"


def test_urgency_low_for_hold():
    assert classify_urgency("HOLD", "A") == "LOW"


def test_status_light_red_for_low_supply():
    assert classify_status_light(2) == "RED"


def test_status_light_green_for_healthy_supply():
    assert classify_status_light(30) == "GREEN"


def test_recommended_quantity_positive_for_reorder():
    qty = recommended_quantity("REORDER_NOW", reorder_point=100, inventory_level=20, eoq=50)
    assert qty > 0


def test_recommended_quantity_negative_for_overstock():
    qty = recommended_quantity("OVERSTOCKED_REDUCE", reorder_point=50, inventory_level=200, eoq=30)
    assert qty < 0


def test_recommended_quantity_zero_for_hold():
    qty = recommended_quantity("HOLD", reorder_point=50, inventory_level=55, eoq=30)
    assert qty == 0


def test_priority_label_mapping():
    assert priority_label("CRITICAL") == "HIGH"
    assert priority_label("MEDIUM") == "MEDIUM"
    assert priority_label("LOW") == "LOW"
