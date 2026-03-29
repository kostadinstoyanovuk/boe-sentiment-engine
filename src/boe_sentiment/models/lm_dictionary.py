"""
Dictionary-based sentiment model for MPC minutes.

Loads the Loughran-McDonald (2011) master dictionary as a base sentiment
layer, mapping LM-Positive terms to hawkish and LM-Negative terms to
dovish polarity. A set of 71 domain-specific monetary policy terms then
overrides the base classification where central bank language diverges
from general financial sentiment.

Negation is handled with a 3-word lookback window that flips polarity.
"""

import re
import csv
import logging
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

LM_MASTER_PATH = Path(__file__).resolve().parents[3] / "data" / "lm_master.csv"

# Monetary policy terms that override LM base classification.
# These capture language specific to central bank communication
# that the general-purpose LM dictionary does not distinguish.
HAWKISH_OVERRIDES = frozenset({
    "tighten", "tightening", "tightened", "restrictive", "restriction",
    "raise", "raised", "raising", "hike", "hikes", "hiking", "hiked",
    "increase", "increased", "increasing", "above", "upside", "upward",
    "elevated", "persistent", "entrenched", "sticky", "embedded",
    "inflation", "inflationary", "overshoot", "overshooting", "overheating",
    "wage", "wages", "wage-price", "second-round", "expectations",
    "unanchored", "tight", "tightness", "robust", "resilient", "resilience",
    "strong", "strength", "buoyant", "demand", "overheated",
    "further", "additional", "sustained", "sufficiently",
    "vigilant", "vigilance", "determined",
})

DOVISH_OVERRIDES = frozenset({
    "ease", "easing", "eased", "accommodative", "accommodation",
    "cut", "cuts", "cutting", "reduce", "reduced", "reducing",
    "decrease", "decreased", "below", "downside", "downward",
    "subdued", "muted", "moderate", "moderation", "moderating",
    "weakness", "weak", "slowdown", "slowing", "deceleration",
    "recession", "contraction", "contracting", "unemployment",
    "slack", "undershoot", "undershooting",
    "disinflation", "deflation",
    "uncertainty", "uncertain", "cautious", "caution", "gradual",
    "patient", "patience", "monitor", "monitoring", "flexible",
})

NEGATION_TERMS = frozenset({
    "not", "no", "never", "neither", "nor", "without",
    "fail", "failed", "unlikely", "insufficient",
})


def _load_lm_master(path: Path) -> tuple[set, set]:
    """Load the LM master dictionary CSV.
    Returns (negative_words, positive_words).
    """
    negative = set()
    positive = set()

    if not path.exists():
        logger.warning(
            f"LM master dictionary not found at {path}. "
            "Run scripts/download_lm.py or see README for instructions. "
            "Falling back to override terms only."
        )
        return negative, positive

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row.get("Word", "").strip().lower()
            if not word:
                continue
            neg_val = row.get("Negative", "0").strip()
            pos_val = row.get("Positive", "0").strip()
            if neg_val and neg_val != "0":
                negative.add(word)
            if pos_val and pos_val != "0":
                positive.add(word)

    logger.info(f"Loaded LM master: {len(negative)} negative, {len(positive)} positive")
    return negative, positive


@dataclass
class DocumentScore:
    hawkish_count: int
    dovish_count: int
    total_words: int
    net_score: float
    hawkish_density: float
    dovish_density: float
    lm_base_hawkish: int = 0
    lm_base_dovish: int = 0
    override_hawkish: int = 0
    override_dovish: int = 0
    hawkish_terms_found: list = field(default_factory=list)
    dovish_terms_found: list = field(default_factory=list)


class LMDictionaryModel:
    """
    Scores MPC minutes on a hawkish-dovish spectrum.

    Architecture:
        Layer 1 (base): LM-Positive -> hawkish, LM-Negative -> dovish
        Layer 2 (override): 71 monetary policy terms take precedence
        Negation: 3-word lookback window flips polarity
    """

    def __init__(self, use_negation: bool = True, normalise: bool = True):
        self.use_negation = use_negation
        self.normalise = normalise

        lm_neg, lm_pos = _load_lm_master(LM_MASTER_PATH)

        # Build final sets: overrides take precedence over LM base.
        # Remove any LM word that appears in the opposite override set.
        self._hawkish = (lm_pos - DOVISH_OVERRIDES) | HAWKISH_OVERRIDES
        self._dovish = (lm_neg - HAWKISH_OVERRIDES) | DOVISH_OVERRIDES

        # Track provenance for diagnostics
        self._is_override = HAWKISH_OVERRIDES | DOVISH_OVERRIDES
        self._lm_base_count = len(lm_pos) + len(lm_neg)

        logger.info(
            f"Dictionary: {len(self._hawkish)} hawkish, "
            f"{len(self._dovish)} dovish "
            f"({self._lm_base_count} LM base + {len(self._is_override)} overrides)"
        )

    def score(self, text: str) -> DocumentScore:
        tokens = self._tokenise(text)
        total = len(tokens)
        if total == 0:
            return DocumentScore(0, 0, 0, 0.0, 0.0, 0.0)

        hawkish_hits = []
        dovish_hits = []
        lm_h, lm_d, ov_h, ov_d = 0, 0, 0, 0

        for i, token in enumerate(tokens):
            negated = self._is_negated(tokens, i) if self.use_negation else False

            if token in self._hawkish:
                is_ov = token in self._is_override
                if negated:
                    dovish_hits.append(f"NOT_{token}")
                    if is_ov:
                        ov_d += 1
                    else:
                        lm_d += 1
                else:
                    hawkish_hits.append(token)
                    if is_ov:
                        ov_h += 1
                    else:
                        lm_h += 1

            elif token in self._dovish:
                is_ov = token in self._is_override
                if negated:
                    hawkish_hits.append(f"NOT_{token}")
                    if is_ov:
                        ov_h += 1
                    else:
                        lm_h += 1
                else:
                    dovish_hits.append(token)
                    if is_ov:
                        ov_d += 1
                    else:
                        lm_d += 1

        h = len(hawkish_hits)
        d = len(dovish_hits)
        denom = total if self.normalise else max(h + d, 1)
        net = max(-1.0, min(1.0, (h - d) / denom))

        return DocumentScore(
            hawkish_count=h,
            dovish_count=d,
            total_words=total,
            net_score=net,
            hawkish_density=h / total,
            dovish_density=d / total,
            lm_base_hawkish=lm_h,
            lm_base_dovish=lm_d,
            override_hawkish=ov_h,
            override_dovish=ov_d,
            hawkish_terms_found=hawkish_hits,
            dovish_terms_found=dovish_hits,
        )

    @staticmethod
    def _tokenise(text: str) -> list:
        text = text.lower()
        text = re.sub(r"[^a-z\s\-]", " ", text)
        return text.split()

    @staticmethod
    def _is_negated(tokens: list, idx: int, window: int = 3) -> bool:
        start = max(0, idx - window)
        return any(t in NEGATION_TERMS for t in tokens[start:idx])
