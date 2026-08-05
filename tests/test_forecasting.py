"""
tests/test_forecasting.py
Unit tests for the forecast evaluation metrics used in demand_forecasting.py.
Run: pytest tests/
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from demand_forecasting import wape, rmse, bias


def test_wape_perfect_forecast_is_zero():
    y_true = [10, 20, 30]
    y_pred = [10, 20, 30]
    assert wape(y_true, y_pred) == 0.0


def test_wape_known_value():
    y_true = [10, 10, 10, 10]
    y_pred = [12, 8, 12, 8]
    # sum(|error|) = 8, sum(|true|) = 40 -> WAPE = 0.2
    assert abs(wape(y_true, y_pred) - 0.2) < 1e-9


def test_rmse_nonnegative():
    y_true = np.random.rand(50) * 100
    y_pred = np.random.rand(50) * 100
    assert rmse(y_true, y_pred) >= 0


def test_bias_direction():
    y_true = [10, 10, 10]
    y_pred_over = [15, 15, 15]
    y_pred_under = [5, 5, 5]
    assert bias(y_true, y_pred_over) > 0   # over-forecasting -> positive bias
    assert bias(y_true, y_pred_under) < 0  # under-forecasting -> negative bias


def test_wape_handles_zero_true_values_gracefully():
    y_true = [0, 0, 0]
    y_pred = [1, 2, 3]
    result = wape(y_true, y_pred)
    assert np.isfinite(result)
