"""Tests for the gilt yield backtester using synthetic data."""

import numpy as np
import pandas as pd
import pytest

from boe_sentiment.analysis.backtester import GiltBacktester, BacktestResult


def _make_synthetic_data(n: int = 36):
    """Generate synthetic signal and yield data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    signal = pd.Series(np.random.randn(n), index=dates, name="signal")
    yields = pd.Series(np.cumsum(np.random.randn(n) * 0.1) + 1.5, index=dates)
    return signal, yields


class TestBacktesterMethods:
    """Test individual backtest methods with synthetic data."""

    def test_cross_correlation_returns_tuple(self):
        signal, yields = _make_synthetic_data()
        yield_change = yields.diff().dropna()
        signal = signal.iloc[1:]
        cc_max, cc_lag = GiltBacktester._cross_correlation(signal, yield_change)
        assert isinstance(cc_max, float)
        assert isinstance(cc_lag, int)

    def test_cross_correlation_bounded(self):
        signal, yields = _make_synthetic_data()
        yield_change = yields.diff().dropna()
        signal = signal.iloc[1:]
        cc_max, _ = GiltBacktester._cross_correlation(signal, yield_change)
        assert -1.0 <= cc_max <= 1.0

    def test_information_coefficient_returns_float(self):
        signal, yields = _make_synthetic_data()
        data = pd.DataFrame({
            "signal": signal,
            "yield": yields,
        })
        data["yield_change"] = data["yield"].diff()
        data = data.dropna()
        ic = GiltBacktester._information_coefficient(data, horizon=1)
        assert isinstance(ic, float)
        assert -1.0 <= ic <= 1.0

    def test_regime_ols_returns_correct_shape(self):
        signal, yields = _make_synthetic_data()
        data = pd.DataFrame({
            "signal": signal,
            "yield": yields,
        })
        data["yield_change"] = data["yield"].diff()
        data = data.dropna()
        result = GiltBacktester._regime_ols(data)
        assert len(result) == 7

    def test_regime_ols_exposes_results_objects(self):
        signal, yields = _make_synthetic_data()
        data = pd.DataFrame({
            "signal": signal,
            "yield": yields,
        })
        data["yield_change"] = data["yield"].diff()
        data = data.dropna()
        _, _, _, n_high, n_low, res_h, res_l = GiltBacktester._regime_ols(data)
        assert n_high > 0
        assert n_low > 0
        assert res_h is not None
        assert res_l is not None
        assert hasattr(res_h, "summary")

    def test_regime_split_is_roughly_even(self):
        signal, yields = _make_synthetic_data(n=48)
        data = pd.DataFrame({
            "signal": signal,
            "yield": yields,
        })
        data["yield_change"] = data["yield"].diff()
        data = data.dropna()
        _, _, _, n_high, n_low, _, _ = GiltBacktester._regime_ols(data)
        assert abs(n_high - n_low) <= 2
