# BoE Sentiment Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An NLP pipeline that quantifies Bank of England Monetary Policy Committee (MPC) hawkishness from meeting minutes and backtests the resulting signal against UK gilt yields (2018-2024).

## What it does

1. **Scrapes** MPC minutes (2018-2024) directly from the Bank of England website
2. **Scores** each document using a two-layer sentiment model:
   - Base layer: Loughran-McDonald (2011) master dictionary (positive/negative word lists)
   - Override layer: 71 domain-specific monetary policy terms tuned for central bank language
   - Negation handling with a 3-word lookback window
3. **Builds** a composite hawkishness index with EMA smoothing and z-score normalisation
4. **Backtests** the signal against 10-year gilt yield changes via:
   - Granger causality testing
   - Lead-lag cross-correlation analysis
   - Regime-conditional OLS (high vs low yield volatility, median split on 6-month rolling std)
   - Information coefficient at 1M and 3M horizons

## Results

> Results pending re-run after LM dictionary integration. Run the pipeline to populate.

| Metric | Value |
| --- | --- |
| Granger causality p-value | -- |
| Optimal predictive lag | -- |
| IC at 1-month horizon | -- |
| IC at 3-month horizon | -- |
| Regime beta (high volatility) | -- |
| Regime beta (low volatility) | -- |
| N (observations after merge) | -- |

## Charts

Charts are generated after running the pipeline. See `docs/figures/`.

## Installation

```bash
git clone https://github.com/kostadinstoyanovuk/boe-sentiment-engine.git
cd boe-sentiment-engine
pip install .
```

Or with Poetry:

```bash
poetry install
```

## Setup

**1. Download the Loughran-McDonald master dictionary:**

```bash
python scripts/download_lm.py
```

This downloads the LM master dictionary CSV from [SRAF at Notre Dame](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) and saves it to `data/lm_master.csv`. If the automatic download fails, download the CSV manually from the link above and place it at `data/lm_master.csv`.

The pipeline will still run without this file (using override terms only), but the full LM base layer will be missing.

**2. Set up your FRED API key:**

```bash
cp .env.example .env
```

Edit `.env` and add your [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html).

## Usage

```python
import os
from dotenv import load_dotenv

from boe_sentiment.data.scraper import MPCScraper
from boe_sentiment.analysis.index_builder import HawkishnessIndexBuilder
from boe_sentiment.analysis.backtester import GiltBacktester

load_dotenv()

# Fetch MPC minutes
docs = MPCScraper(cache_dir="data/raw").fetch_minutes(2018, 2024)

# BoE Sentiment Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An NLP pipeline that quantifies Bank of England Monetary Policy Committee (MPC) hawkishness from meeting minutes and backtests the resulting signal against UK gilt yields (2018-2024).

## What it does

1. **Scrapes** MPC minutes (2018-2024) directly from the Bank of England website
2. **Scores** each document using a two-layer sentiment model:
   - Base layer: Loughran-McDonald (2011) master dictionary (positive/negative word lists)
   - Override layer: 71 domain-specific monetary policy terms tuned for central bank language
   - Negation handling with a 3-word lookback window
3. **Builds** a composite hawkishness index with EMA smoothing and z-score normalisation
4. **Backtests** the signal against 10-year gilt yield changes via:
   - Granger causality testing
   - Lead-lag cross-correlation analysis
   - Regime-conditional OLS (high vs low yield volatility, median split on 6-month rolling std)
   - Information coefficient at 1M and 3M horizons

## Results

| Metric | Value |
| --- | --- |
| Granger causality p-value | 0.2148 |
| Optimal predictive lag | 1 month |
| IC at 1-month horizon | 0.098 |
| IC at 3-month horizon | 0.085 |
| Regime beta (high volatility) | 0.111 (p=0.261) |
| Regime beta (low volatility) | 0.018 (p=0.634) |
| N (observations) | 52 |

The dictionary-based sentiment signal does not carry statistically significant predictive content for gilt yields at conventional levels. This is consistent with efficient incorporation of monetary policy expectations into bond prices ahead of minutes publication.

## Installation

```bash
git clone https://github.com/kostadinstoyanovuk/boe-sentiment-engine.git
cd boe-sentiment-engine
pip install .
```

Or with Poetry:

```bash
poetry install
```

## Setup

**1. Download the Loughran-McDonald master dictionary:**

```bash
python scripts/download_lm.py
```

This downloads the LM master dictionary CSV from [SRAF at Notre Dame](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) and saves it to `data/lm_master.csv`. If the automatic download fails, download the CSV manually from the link above and place it at `data/lm_master.csv`.

The pipeline will still run without this file (using override terms only), but the full LM base layer will be missing.

**2. Set up your FRED API key:**

```bash
cp .env.example .env
```

Edit `.env` and add your [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html).

## Usage

```python
import os
from dotenv import load_dotenv

from boe_sentiment.data.scraper import MPCScraper
from boe_sentiment.analysis.index_builder import HawkishnessIndexBuilder
from boe_sentiment.analysis.backtester import GiltBacktester

load_dotenv()

# Fetch MPC minutes
docs = MPCScraper(cache_dir="data/raw").fetch_minutes(2018, 2024)

# Build hawkishness index (dictionary model only, no GPU needed)
index = HawkishnessIndexBuilder(finbert_weight=0.0).build(docs)

# Backtest against gilt yields
bt = GiltBacktester(fred_api_key=os.getenv("BOE_FRED_API_KEY"))
r = bt.run(index)

print(f"Granger p-value:  {r.granger_pvalue:.4f}")
print(f"Optimal lag:      {r.optimal_lag} months")
print(f"IC (1M):          {r.ic_1m:.3f}")
print(f"IC (3M):          {r.ic_3m:.3f}")
print(f"Regime beta high: {r.regime_beta_high:.4f}")
print(f"Regime beta low:  {r.regime_beta_low:.4f}")
print(f"N observations:   {r.n_observations}")

# Full regime OLS output (for research reporting)
if r.regime_results_high:
    print(r.regime_results_high.summary())
if r.regime_results_low:
    print(r.regime_results_low.summary())
```

## Project structure

```
boe-sentiment-engine/
├── src/boe_sentiment/
│   ├── data/            # BoE scraper
│   ├── models/          # LM dictionary + FinBERT classifier
│   ├── analysis/        # Index builder + backtester
│   └── visualization/   # Publication-quality charts
├── scripts/             # LM dictionary download
├── tests/               # pytest suite
├── data/                # LM master CSV + cached MPC text
├── docs/figures/         # Output charts
├── pyproject.toml
└── .env.example
```

## Dictionary architecture

The sentiment model uses the Loughran-McDonald (2011) master dictionary as a base layer, mapping LM-Positive terms to hawkish polarity and LM-Negative terms to dovish polarity. A set of 71 domain-specific monetary policy terms then overrides the base classification where central bank language diverges from general financial sentiment.

For example, "inflation" does not appear in the LM positive/negative lists but is unambiguously hawkish in MPC minutes. The override layer captures these distinctions.

See `src/boe_sentiment/models/lm_dictionary.py` for the full term lists.

## Data sources

- [Bank of England MPC Minutes](https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes)
- [FRED: UK 10Y Gilt Yield (IRLTLT01GBM156N)](https://fred.stlouisfed.org/series/IRLTLT01GBM156N)
- [Loughran-McDonald Master Dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)

## References

- Loughran, T. and McDonald, B. (2011). *When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks.* Journal of Finance, 66(1), 35-65.
- Yang, Y., Uy, M.C.S. and Huang, A. (2020). *FinBERT: A Pretrained Language Model for Financial Communications.* arXiv:2006.08097.
- Apel, M. and Blix Grimaldi, M. (2012). *The Information Content of Central Bank Minutes.* Riksbank Research Paper Series, No. 92.

## Author

Kostadin Stoyanov - [GitHub](https://github.com/kostadinstoyanovuk) | [SSRN](https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=kostadinstoyanov)

## License

MIT
