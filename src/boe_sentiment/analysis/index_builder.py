"""
Builds the hawkishness index from per-document sentiment scores.
Combines LM dictionary and optional FinBERT scores, applies EMA
smoothing, and outputs z-scored values for comparability.
"""

import logging

import pandas as pd

from boe_sentiment.data.scraper import MPCDocument
from boe_sentiment.models.lm_dictionary import LMDictionaryModel

logger = logging.getLogger(__name__)


class HawkishnessIndexBuilder:
    """
    Combines FinBERT and LM dictionary scores into a single index.
    Set finbert_weight=0.0 to use dictionary model only (no GPU needed).
    """

    def __init__(self, finbert_weight: float = 0.6, ema_span: int = 3):
        self.finbert_weight = finbert_weight
        self.lm_weight = 1.0 - finbert_weight
        self.ema_span = ema_span
        self._lm_model = LMDictionaryModel()

    def build(
        self,
        documents: list[MPCDocument],
        finbert_scores: dict = None,
    ) -> pd.DataFrame:
        """Build the index from scored MPC documents.

        Returns a DataFrame indexed by date with columns:
        lm_score, finbert_score, composite_score, ema_score, zscore.
        """
        rows = []
        for doc in sorted(documents, key=lambda d: d.date):
            lm = self._lm_model.score(doc.text).net_score
            fb = finbert_scores.get(doc.date) if finbert_scores else None

            if fb is not None:
                composite = self.finbert_weight * fb + self.lm_weight * lm
            else:
                composite = lm

            rows.append({
                "date": doc.date,
                "lm_score": lm,
                "finbert_score": fb,
                "composite_score": composite,
            })

        df = pd.DataFrame(rows).set_index("date")

        if self.ema_span and len(df) > 1:
            df["ema_score"] = (
                df["composite_score"].ewm(span=self.ema_span, adjust=False).mean()
            )
        else:
            df["ema_score"] = df["composite_score"]

        mu = df["ema_score"].mean()
        sigma = df["ema_score"].std()
        df["zscore"] = (df["ema_score"] - mu) / sigma if sigma > 0 else 0.0

        logger.info(f"Built index: {len(df)} periods, mean={mu:.3f}, std={sigma:.3f}")
        return df
