"""Unit tests for the funding-carry validation script's pure (no-network) helpers."""
import numpy as np
import pandas as pd
import pytest

from scripts.validate_funding_carry import (
    EPOCHS_PER_YEAR,
    annualized,
    build_aligned_frame,
    effective_cost_per_leg,
    per_obs_sharpe,
)


def test_per_obs_sharpe_known_values():
    assert per_obs_sharpe([0.01, 0.01, 0.01]) == 0.0        # zero variance → 0
    assert per_obs_sharpe([1.0]) == 0.0                     # <2 points → 0
    r = np.array([0.02, -0.01, 0.03, 0.00])
    assert per_obs_sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1))


def test_annualized_scales_by_sqrt_epochs_per_year():
    assert annualized(0.3) == pytest.approx(0.3 * np.sqrt(EPOCHS_PER_YEAR))
    assert EPOCHS_PER_YEAR == 3 * 365


def test_build_aligned_frame_matches_funding_to_nearest_candle():
    idx8 = pd.date_range("2024-01-01", periods=4, freq="8h", tz="UTC")
    funding = pd.DataFrame({"funding_rate": [0.0004, 0.0003, 0.0002, 0.0001]}, index=idx8)
    spot = pd.DataFrame({"close": [100, 101, 102, 103], "volume": [10, 11, 12, 13],
                         "open": 0, "high": 0, "low": 0}, index=idx8)
    out = build_aligned_frame(funding, spot)
    assert list(out.columns) == ["funding_rate", "close", "volume"]
    assert len(out) == 4
    assert out["close"].tolist() == [100, 101, 102, 103]


def test_effective_cost_includes_fee_plus_nonnegative_slippage():
    idx = pd.date_range("2024-01-01", periods=5, freq="8h", tz="UTC")
    df = pd.DataFrame({"close": [100.0] * 5, "volume": [5000.0] * 5}, index=idx)
    total, taker, slippage = effective_cost_per_leg(df, notional_usd=10_000.0)
    assert taker == pytest.approx(0.0004)          # VIP0 taker
    assert slippage >= 0.0
    assert total == pytest.approx(taker + slippage)
