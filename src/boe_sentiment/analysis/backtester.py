"""
Backtests the hawkishness index against UK gilt yield movements.

Runs four tests:
- Granger causality
- Lead-lag cross-correlation
- Regime-conditional OLS (high vs low volatility periods)
- Information coefficient at 1M and 3M horizons
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

logger = logging.getLogger(__name__)

UK_10Y_GILT = "IRLTLT01GBM156N"


@dataclass
class BacktestResult:
    granger_pvalue: float
    granger_significant: bool
    optimal_lag: int
    cross_corr_max: float
    cross_corr_lag: int
    ic_1m: float
    ic_3m: float
    regime_beta_high: float
    regime_beta_low: float
    regime_difference_significant: bool
    n_observations: int = 0
    n_high_vol: int = 0
    n_low_vol: int = 0
    regime_results_high: object = None
    regime_results_low: object = None


class GiltBacktester:
    """Backtests the hawkishness index against UK gilt yield changes."""

    def __init__(
        self,
        fred_api_key: str,
        target_series: str = UK_10Y_GILT,
        max_lags: int = 6,
    ):
        self.fred_api_key = fred_api_key
        self.target_series = target_series
        self.max_lags = max_lags

    def fetch_gilt_yields(
        self,
        start: str = "2018-01-01",
        end: str = "2024-12-31",
    ) -> pd.Series:
        from fredapi import Fred

        fred = Fred(api_key=self.fred_api_key)
        series = fred.get_series(
            self.target_series,
            observation_start=start,
            observation_end=end,
        )
        series.index = pd.to_datetime(series.index).to_period("M").to_timestamp()
        series.name = "gilt_yield"
        logger.info(f"Fetched {len(series)} gilt yield observations")
        return series

    def run(self, hawkishness_index: pd.DataFrame) -> BacktestResult:
        start = hawkishness_index.index.min().strftime("%Y-%m-%d")
        end = hawkishness_index.index.max().strftime("%Y-%m-%d")
        gilts = self.fetch_gilt_yields(start=start, end=end)

        signal = hawkishness_index["zscore"].copy()
        signal.index = pd.to_datetime(signal.index).to_period("M").to_timestamp()

        merged = pd.DataFrame({"signal": signal, "yield": gilts}).dropna()
        merged["yield_change"] = merged["yield"].diff()
        merged = merged.dropna()

        if len(merged) < 12:
            raise ValueError(f"Not enough overlapping data: {len(merged)} observations.")

        logger.info(f"Running backtest on {len(merged)} observations")

        granger_p, optimal_lag = self._granger_test(
            merged["signal"], merged["yield_change"]
        )
        cc_max, cc_lag = self._cross_correlation(
            merged["signal"], merged["yield_change"]
        )
        ic_1m = self._information_coefficient(merged, horizon=1)
        ic_3m = self._information_coefficient(merged, horizon=3)
        beta_high, beta_low, regime_sig, n_high, n_low, res_h, res_l = (
            self._regime_ols(merged)
        )

        return BacktestResult(
            granger_pvalue=granger_p,
            granger_significant=granger_p < 0.05,
            optimal_lag=optimal_lag,
            cross_corr_max=cc_max,
            cross_corr_lag=cc_lag,
            ic_1m=ic_1m,
            ic_3m=ic_3m,
            regime_beta_high=beta_high,
            regime_beta_low=beta_low,
            regime_difference_significant=regime_sig,
            n_observations=len(merged),
            n_high_vol=n_high,
            n_low_vol=n_low,
            regime_results_high=res_h,
            regime_results_low=res_l,
        )

    def _granger_test(self, signal: pd.Series, yield_change: pd.Series):
        data = pd.concat([yield_change, signal], axis=1).dropna()
        data.columns = ["yield_change", "signal"]
        results = grangercausalitytests(data, maxlag=self.max_lags, verbose=False)
        pvals = {
            lag: results[lag][0]["ssr_ftest"][1]
            for lag in range(1, self.max_lags + 1)
        }
        optimal_lag = min(pvals, key=pvals.get)
        return pvals[optimal_lag], optimal_lag

    @staticmethod
    def _cross_correlation(
        signal: pd.Series,
        yield_change: pd.Series,
        max_lag: int = 6,
    ):
        correlations = {}
        for lag in range(-max_lag, max_lag + 1):
            shifted = signal.shift(lag)
            aligned = pd.concat([shifted, yield_change], axis=1).dropna()
            if len(aligned) < 5:
                continue
            r, _ = stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
            correlations[lag] = r
        if not correlations:
            return 0.0, 0
        best_lag = max(correlations, key=lambda k: abs(correlations[k]))
        return correlations[best_lag], best_lag

    @staticmethod
    def _information_coefficient(data: pd.DataFrame, horizon: int = 1) -> float:
        fwd = data["yield_change"].shift(-horizon)
        aligned = pd.concat([data["signal"], fwd], axis=1).dropna()
        if len(aligned) < 5:
            return float("nan")
        r, _ = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return float(r)

    @staticmethod
    def _regime_ols(data: pd.DataFrame, vol_window: int = 6):
        data = data.copy()
        data["yield_vol"] = data["yield_change"].rolling(vol_window).std()
        data = data.dropna()
        if len(data) < 12:
            return float("nan"), float("nan"), False, 0, 0, None, None

        median_vol = data["yield_vol"].median()
        high = data[data["yield_vol"] >= median_vol]
        low = data[data["yield_vol"] < median_vol]

        def _fit(df):
            if len(df) < 4:
                return float("nan"), None
            X = add_constant(df["signal"])
            res = OLS(df["yield_change"], X).fit()
            beta = float(res.params.get("signal", float("nan")))
            return beta, res

        beta_high, res_high = _fit(high)
        beta_low, res_low = _fit(low)

        try:
            _, p = stats.ttest_ind(
                high["yield_change"] - high["signal"] * beta_high,
                low["yield_change"] - low["signal"] * beta_low,
                equal_var=False,
            )
            significant = p < 0.10
        except Exception:
            significant = False

        return beta_high, beta_low, significant, len(high), len(low), res_high, res_low
