"""Tests for the dictionary-based sentiment model."""

import pytest

from boe_sentiment.models.lm_dictionary import (
    LMDictionaryModel,
    HAWKISH_OVERRIDES,
    DOVISH_OVERRIDES,
    NEGATION_TERMS,
)


@pytest.fixture
def model():
    return LMDictionaryModel()


def test_hawkish_text(model):
    text = "The committee voted to raise interest rates due to persistent inflation."
    result = model.score(text)
    assert result.hawkish_count > result.dovish_count
    assert result.net_score > 0


def test_dovish_text(model):
    text = "Economic weakness and rising unemployment suggest easing is appropriate."
    result = model.score(text)
    assert result.dovish_count > result.hawkish_count
    assert result.net_score < 0


def test_negation_flips_polarity(model):
    text = "The committee decided not to tighten monetary policy."
    result = model.score(text)
    assert "NOT_tighten" in result.dovish_terms_found


def test_empty_text(model):
    result = model.score("")
    assert result.total_words == 0
    assert result.net_score == 0.0


def test_neutral_text(model):
    text = "The meeting was held on Thursday at noon in London."
    result = model.score(text)
    assert result.hawkish_count == 0
    assert result.dovish_count == 0
    assert result.net_score == 0.0


def test_net_score_bounded(model):
    text = "tighten tighten tighten raise raise hike hike inflation elevated"
    result = model.score(text)
    assert -1.0 <= result.net_score <= 1.0


def test_density_calculation(model):
    text = "inflation is elevated and persistent"
    result = model.score(text)
    assert result.hawkish_density == result.hawkish_count / result.total_words
    assert result.dovish_density == result.dovish_count / result.total_words


def test_override_sets_no_overlap():
    overlap = HAWKISH_OVERRIDES & DOVISH_OVERRIDES
    assert len(overlap) == 0, f"Overlap found: {overlap}"


def test_negation_window():
    model = LMDictionaryModel()
    # "not" is 4 tokens away from "tighten" - outside 3-word window
    text = "not something something something tighten"
    result = model.score(text)
    assert "tighten" in result.hawkish_terms_found


def test_negation_disabled():
    model = LMDictionaryModel(use_negation=False)
    text = "not tighten"
    result = model.score(text)
    assert "tighten" in result.hawkish_terms_found


def test_provenance_tracking(model):
    text = "inflation is elevated but unemployment is rising"
    result = model.score(text)
    assert result.override_hawkish + result.override_dovish > 0
