"""Publication-quality charts for the BoE Sentiment Engine."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "serif",
    "font.size": 11,
}


def plot_hawkishness_index(
    index: pd.DataFrame,
    gilt_yields: pd.Series | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the hawkishness index with optional gilt yields overlay."""
    with plt.rc_context(STYLE):
        n_panels = 2 if gilt_yields is not None else 1
        fig, axes = plt.subplots(
            n_panels, 1,
            figsize=(12, 8 if n_panels == 2 else 5),
            sharex=True,
        )
        if n_panels == 1:
            axes = [axes]

        ax1 = axes[0]
        ax1.bar(
            index.index,
            index["zscore"],
            color=["#c0392b" if v > 0 else "#2980b9" for v in index["zscore"]],
            alpha=0.7,
            width=20,
            label="Hawkishness (z-score)",
        )
        ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax1.set_ylabel("Hawkishness z-score", fontsize=11)
        ax1.set_title(
            "Bank of England MPC Hawkishness Index (2018-2024)",
            fontsize=13,
            fontweight="bold",
        )
        ax1.legend(frameon=False)

        for i in range(len(index)):
            if index["zscore"].iloc[i] > 0.5:
                ax1.axvspan(
                    index.index[i],
                    index.index[i] + pd.DateOffset(months=1),
                    alpha=0.08,
                    color="#c0392b",
                )

        if gilt_yields is not None:
            ax2 = axes[1]
            ax2.plot(
                gilt_yields.index,
                gilt_yields.values,
                color="#2c3e50",
                linewidth=1.5,
                label="UK 10Y Gilt Yield (%)",
            )
            ax2.set_ylabel("Gilt yield (%)", fontsize=11)
            ax2.set_xlabel("Date", fontsize=11)
            ax2.legend(frameon=False)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {output_path}")

        return fig


def plot_cross_correlation(
    signal: pd.Series,
    yield_change: pd.Series,
    max_lag: int = 6,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot cross-correlation between hawkishness and gilt yield changes."""
    from scipy import stats

    lags = range(-max_lag, max_lag + 1)
    correlations = []
    for lag in lags:
        shifted = signal.shift(lag)
        aligned = pd.concat([shifted, yield_change], axis=1).dropna()
        if len(aligned) < 5:
            correlations.append(0.0)
        else:
            r, _ = stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
            correlations.append(r)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ["#c0392b" if c > 0 else "#2980b9" for c in correlations]
        ax.bar(list(lags), correlations, color=colors, alpha=0.75)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

        n = len(signal.dropna())
        ci = 1.96 / np.sqrt(n)
        ax.axhline(ci, color="gray", linewidth=0.8, linestyle=":", label="95% CI")
        ax.axhline(-ci, color="gray", linewidth=0.8, linestyle=":")

        ax.set_xlabel("Lag (months; positive = sentiment leads)", fontsize=11)
        ax.set_ylabel("Pearson correlation", fontsize=11)
        ax.set_title(
            "Cross-correlation: MPC Hawkishness vs Gilt Yield Change",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(frameon=False)
        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {output_path}")

        return fig
