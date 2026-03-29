"""
FinBERT wrapper for scoring MPC minutes on a hawkish-dovish scale.
Loads ProsusAI/finbert and maps financial sentiment labels to [-1, +1].
Lazy-loaded to avoid slow imports when not needed.
"""

import re
import logging

import numpy as np

logger = logging.getLogger(__name__)

LABEL_WEIGHTS = {
    "positive": 0.6,
    "negative": -0.5,
    "neutral": 0.0,
}


class FinBERTClassifier:
    """Wraps ProsusAI/finbert for document-level hawkishness scoring."""

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, batch_size: int = 16, max_length: int = 512, device: str = None):
        self.batch_size = batch_size
        self.max_length = max_length
        self._device = device
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        import torch
        from transformers import pipeline

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading FinBERT on {self._device}")
        self._pipeline = pipeline(
            "text-classification",
            model=self.MODEL_NAME,
            device=0 if self._device == "cuda" else -1,
            truncation=True,
            max_length=self.max_length,
        )

    def score_sentences(self, sentences: list[str]) -> list[dict]:
        self._load()
        results = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            outputs = self._pipeline(batch)
            for sent, out in zip(batch, outputs):
                label = out["label"].lower()
                conf = out["score"]
                results.append({
                    "text": sent,
                    "label": label,
                    "confidence": conf,
                    "hawkish_score": LABEL_WEIGHTS.get(label, 0.0) * conf,
                })
        return results

    def score_document(self, text: str, min_words: int = 5) -> float:
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text.strip())
            if len(s.split()) >= min_words
        ]
        if not sentences:
            return 0.0
        scored = self.score_sentences(sentences)
        return float(np.mean([s["hawkish_score"] for s in scored]))
