"""Task 1.4 — funding-rate carry: threshold logic, signal generation, hysteresis."""
import pandas as pd
import pytest

from config.constants import Signal
from strategies.funding_carry import FundingCarryStrategy
from strategies.registry import get_strategy, list_strategies


def _df(funding_rates, price=100.0):
    idx = pd.date_range("2024-01-01", periods=len(funding_rates), freq="8h", tz="UTC")
    return pd.DataFrame(
        {"funding_rate": funding_rates, "close": [price] * len(funding_rates)},
        index=idx,
    )


# Per-epoch rates chosen against a 3-epochs/day * 365 = 1095x annualiser:
_HIGH = 0.0004    # APR ≈ 43.8%  (> 30% entry)
_MID = 0.00015    # APR ≈ 16.4%  (between 10% exit and 30% entry)
_LOW = 0.00005    # APR ≈  5.5%  (< 10% exit)


class TestAnnualisation:
    def test_apr_uses_3_epochs_per_day(self):
        s = FundingCarryStrategy()
        assert s.annualized_apr(0.0004) == pytest.approx(0.0004 * 3 * 365)
        assert s.annualized_apr(0.0004) == pytest.approx(0.438, abs=1e-3)

    def test_exit_above_entry_is_rejected(self):
        with pytest.raises(ValueError):
            FundingCarryStrategy(entry_apr=0.20, exit_apr=0.30)


class TestThresholdAndSignals:
    def test_enters_only_when_apr_exceeds_entry(self):
        s = FundingCarryStrategy(entry_apr=0.30, exit_apr=0.10)
        sig = s.generate_signals(_df([_LOW, _HIGH, _LOW]))
        assert sig.tolist() == [0, 1, -1]      # enter on the high bar, exit on the next low

    def test_no_entry_when_funding_is_low(self):
        s = FundingCarryStrategy()
        assert s.generate_signals(_df([_LOW, _LOW, _MID, _LOW])).abs().sum() == 0

    def test_signal_is_state_change_not_state(self):
        # Held across multiple high bars → +1 once (entry), zeros while holding.
        s = FundingCarryStrategy()
        sig = s.generate_signals(_df([_HIGH, _HIGH, _HIGH]))
        assert sig.tolist() == [1, 0, 0]

    def test_hysteresis_holds_through_the_mid_zone(self):
        # low, high(enter), mid(hold), mid(hold), low(exit), high(re-enter)
        s = FundingCarryStrategy(entry_apr=0.30, exit_apr=0.10)
        seq = [_LOW, _HIGH, _MID, _MID, _LOW, _HIGH]
        assert s.in_carry_state(_df(seq)).tolist() == [False, True, True, True, False, True]
        assert s.generate_signals(_df(seq)).tolist() == [0, 1, 0, 0, -1, 1]

    def test_mid_zone_alone_never_enters(self):
        # APR in the (exit, entry) band without first crossing entry → no carry.
        s = FundingCarryStrategy()
        assert not s.in_carry_state(_df([_MID, _MID, _MID])).any()


class TestGetSignal:
    def test_buy_names_both_legs_and_apr(self):
        s = FundingCarryStrategy()
        ts = s.get_signal(_df([_HIGH]), "BTC/USDT")
        assert ts.signal == Signal.BUY
        assert ts.is_actionable()                         # confidence >= 0.60
        assert "LONG spot" in ts.reason and "SHORT perp" in ts.reason
        assert "APR" in ts.reason

    def test_hold_below_threshold(self):
        s = FundingCarryStrategy()
        ts = s.get_signal(_df([_LOW]), "BTC/USDT")
        assert ts.signal == Signal.HOLD and ts.confidence == 0.0

    def test_missing_funding_column_is_safe(self):
        s = FundingCarryStrategy()
        df = pd.DataFrame({"close": [100.0, 101.0]},
                          index=pd.date_range("2024-01-01", periods=2, freq="8h", tz="UTC"))
        assert (s.generate_signals(df) == 0).all()
        assert s.get_signal(df, "BTC/USDT").signal == Signal.HOLD


class TestCarryReturns:
    def test_collects_funding_while_held_net_of_entry_cost(self):
        s = FundingCarryStrategy()
        # Two held epochs then exit: [enter@HIGH, hold@HIGH, exit@LOW]
        r = s.carry_returns(_df([_HIGH, _HIGH, _LOW]), cost_per_leg=0.0004)
        # Bar 0: enter → funding _HIGH minus the 2-leg entry cost (2 * 0.0004 = 0.0008).
        assert r.iloc[0] == pytest.approx(_HIGH - 0.0008)
        # Bar 1: held, full funding, no cost.
        assert r.iloc[1] == pytest.approx(_HIGH)
        # Bar 2: not held (exited) → only the 2-leg exit cost, no funding.
        assert r.iloc[2] == pytest.approx(-0.0008)

    def test_flat_when_never_in_carry(self):
        s = FundingCarryStrategy()
        assert s.carry_returns(_df([_LOW, _LOW])).abs().sum() == 0.0


class TestRegistry:
    def test_registered_and_constructible(self):
        assert "funding_carry" in list_strategies()
        strat = get_strategy("funding_carry", entry_apr=0.25)
        assert isinstance(strat, FundingCarryStrategy)
        assert strat.entry_apr == 0.25
