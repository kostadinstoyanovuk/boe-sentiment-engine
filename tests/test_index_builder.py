"""Tests for the hawkishness index builder."""

from datetime import datetime

import pandas as pd
import pytest

from boe_sentiment.data.scraper import MPCDocument
from boe_sentiment.analysis.index_builder import HawkishnessIndexBuilder


def _make_doc(date_str: str, text: str) -> MPCDocument:
    return MPCDocument(
        date=datetime.strptime(date_str, "%Y-%m"),
        url="/test",
        text=text,
        n_words=len(text.split()),
    )


@pytest.fixture
def sample_docs():
    return [
        _make_doc("2023-01", "The committee voted to raise rates due to persistent inflation."),
        _make_doc("2023-03", "Economic weakness and unemployment suggest caution on rates."),
        _make_doc("2023-06", "Inflation remains elevated and the labour market is tight."),
        _make_doc("2023-09", "Growth has slowed and the committee sees downside risks."),
    ]


def test_build_returns_dataframe(sample_docs):
    builder = HawkishnessIndexBuilder(finbert_weight=0.0)
    index = builder.build(sample_docs)
    assert isinstance(index, pd.DataFrame)
    assert len(index) == 4


def test_build_has_required_columns(sample_docs):
    builder = HawkishnessIndexBuilder(finbert_weight=0.0)
    index = builder.build(sample_docs)
    for col in ["lm_score", "composite_score", "ema_score", "zscore"]:
        assert col in index.columns


def test_zscore_mean_near_zero(sample_docs):
    builder = HawkishnessIndexBuilder(finbert_weight=0.0)
    index = builder.build(sample_docs)
    assert abs(index["zscore"].mean()) < 0.01


def test_build_sorted_by_date(sample_docs):
    builder = HawkishnessIndexBuilder(finbert_weight=0.0)
    index = builder.build(list(reversed(sample_docs)))
    dates = list(index.index)
    assert dates == sorted(dates)
