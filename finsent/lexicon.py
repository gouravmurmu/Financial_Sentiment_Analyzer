"""A finance-specific sentiment lexicon with phrase rules and negation handling.

It needs no training data or downloads, so it is the fallback scorer for live headlines when FinBERT is
not installed. Word weights are hand-tuned for market news (e.g. "liability" and "tax" are neutral here,
unlike in general-purpose lexicons).
"""

from __future__ import annotations

import numpy as np

from .text import NEGATIONS, tokenize

POSITIVE = {
    # strong (2.0)
    "surge": 2, "surges": 2, "surged": 2, "soar": 2, "soars": 2, "soared": 2, "skyrocket": 2, "skyrockets": 2,
    "rally": 1.5, "rallies": 1.5, "rallied": 1.5, "jump": 1.5, "jumps": 1.5, "jumped": 1.5, "record": 1.5,
    "beat": 1.5, "beats": 1.5, "upgrade": 2, "upgrades": 2, "upgraded": 2, "outperform": 1.5,
    "outperforms": 1.5, "outperformed": 1.5, "breakthrough": 2, "blockbuster": 2, "bullish": 2,
    "boom": 1.5, "booming": 1.5, "windfall": 1.5,
    # moderate (1.0)
    "gain": 1, "gains": 1, "gained": 1, "rise": 1, "rises": 1, "rose": 1, "climb": 1, "climbs": 1,
    "climbed": 1, "growth": 1, "grow": 1, "grows": 1, "grew": 1, "profit": 1, "profitable": 1.5,
    "profits": 1, "strong": 1, "stronger": 1, "strength": 1, "robust": 1, "upbeat": 1.5, "optimistic": 1,
    "optimism": 1, "expand": 1, "expands": 1, "expansion": 1, "approve": 1, "approves": 1, "approved": 1,
    "approval": 1, "win": 1, "wins": 1, "won": 1, "award": 1, "awarded": 1, "boost": 1, "boosts": 1,
    "boosted": 1, "raise": 0.8, "raises": 0.8, "raised": 0.8, "higher": 0.8, "improve": 1, "improves": 1,
    "improved": 1, "improvement": 1, "recover": 1, "recovers": 1, "recovery": 1, "rebound": 1,
    "rebounds": 1, "positive": 1, "exceed": 1.5, "exceeds": 1.5, "exceeded": 1.5, "top": 0.6, "tops": 1,
    "buy": 0.8, "buyback": 1, "dividend": 0.6, "partnership": 0.6, "launch": 0.4, "launches": 0.4,
    "innovative": 0.8, "momentum": 0.6, "accelerate": 1, "accelerates": 1, "resilient": 1, "upside": 1,
    "tailwind": 1, "tailwinds": 1, "milestone": 0.8, "efficient": 0.6, "success": 1, "successful": 1,
}

NEGATIVE = {
    # strong
    "plunge": 2, "plunges": 2, "plunged": 2, "plummet": 2, "plummets": 2, "crash": 2, "crashes": 2,
    "collapse": 2, "collapses": 2, "tumble": 1.5, "tumbles": 1.5, "tumbled": 1.5, "sink": 1.5,
    "sinks": 1.5, "sank": 1.5, "slump": 1.5, "slumps": 1.5, "miss": 1.5, "misses": 1.5, "missed": 1.5,
    "downgrade": 2, "downgrades": 2, "downgraded": 2, "bankruptcy": 2.5, "bankrupt": 2.5, "fraud": 2.5,
    "scandal": 2, "default": 2, "defaults": 2, "lawsuit": 1.5, "sues": 1.5, "sued": 1.5, "probe": 1.5,
    "investigation": 1.5, "recall": 1.5, "recalls": 1.5, "layoffs": 1.5, "layoff": 1.5, "bearish": 2,
    "selloff": 1.5, "sell-off": 1.5, "warning": 1.5, "warns": 1.5, "warned": 1.5, "crisis": 2,
    # moderate
    "fall": 1, "falls": 1, "fell": 1, "drop": 1, "drops": 1, "dropped": 1, "decline": 1, "declines": 1,
    "declined": 1, "loss": 1.2, "losses": 1.2, "lose": 1, "loses": 1, "weak": 1, "weaker": 1,
    "weakness": 1, "slow": 0.8, "slows": 0.8, "slowdown": 1, "cut": 1, "cuts": 1, "lower": 0.8,
    "lowers": 1, "lowered": 1, "risk": 0.6, "risks": 0.6, "concern": 1, "concerns": 1, "fear": 1,
    "fears": 1, "worry": 1, "worries": 1, "uncertainty": 1, "volatile": 0.6, "pressure": 0.8,
    "headwind": 1, "headwinds": 1, "delay": 1, "delays": 1, "delayed": 1, "halt": 1.2, "halts": 1.2,
    "fine": 1, "fined": 1.2, "penalty": 1.2, "shortfall": 1.5, "disappoint": 1.5, "disappoints": 1.5,
    "disappointing": 1.5, "negative": 1, "struggle": 1, "struggles": 1, "downturn": 1.2, "sell": 0.8,
    "underperform": 1.5, "underperforms": 1.5, "dilution": 1, "shortage": 1, "tariff": 0.6,
    "tariffs": 0.6, "inflation": 0.5, "recession": 1.5, "downside": 1, "volatility": 0.4,
    "breach": 1.5, "outage": 1.2, "resigns": 1, "exit": 0.5, "suspend": 1.2, "suspends": 1.2,
}

# Multi-word rules checked before single words; matched tokens are not re-scored.
PHRASES = {
    ("beats", "estimates"): 2, ("beat", "estimates"): 2, ("raises", "guidance"): 2.5,
    ("raised", "guidance"): 2.5, ("raises", "outlook"): 2.5, ("raises", "forecast"): 2.5,
    ("price", "target", "raised"): 2, ("all", "time", "high"): 2, ("record", "high"): 2,
    ("better", "than", "expected"): 2, ("ahead", "of", "expectations"): 2,
    ("misses", "estimates"): -2, ("missed", "estimates"): -2, ("cuts", "guidance"): -2.5,
    ("lowers", "guidance"): -2.5, ("cuts", "outlook"): -2.5, ("cuts", "forecast"): -2.5,
    ("price", "target", "cut"): -2, ("worse", "than", "expected"): -2, ("below", "expectations"): -2,
    ("short", "seller"): -1.5, ("job", "cuts"): -1.5, ("profit", "warning"): -2.5,
    ("52", "week", "low"): -1.5, ("52", "week", "high"): 1.5,
}

LEXICON = {**POSITIVE, **{w: -v for w, v in NEGATIVE.items()}}
_MAX_PHRASE = max(len(p) for p in PHRASES)


def score(text: str, neg_window: int = 3) -> float:
    """Polarity in [-1, 1]; 0 means no sentiment-bearing words were found."""
    toks = tokenize(text.replace("-", " ").replace("sell off", "selloff"))
    total, hits, i, flip_until = 0.0, 0, 0, -1
    while i < len(toks):
        if toks[i] in NEGATIONS:
            flip_until = i + neg_window
            i += 1
            continue
        w, step = None, 1
        for n in range(_MAX_PHRASE, 1, -1):
            if tuple(toks[i:i + n]) in PHRASES:
                w, step = PHRASES[tuple(toks[i:i + n])], n
                break
        if w is None:
            w = LEXICON.get(toks[i])
        if w is not None:
            total += -w if i <= flip_until else w
            hits += 1
        i += step
    if hits == 0:
        return 0.0
    return float(np.tanh(total / 2.0))


def predict(texts, threshold: float = 0.15) -> tuple[list[str], np.ndarray]:
    """Labels and an approximate [neg, neu, pos] probability matrix from lexicon polarity."""
    s = np.array([score(t) for t in texts])
    labels = np.where(s > threshold, "positive", np.where(s < -threshold, "negative", "neutral"))
    pos = np.clip(s, 0, 1)
    neg = np.clip(-s, 0, 1)
    neu = 1 - np.abs(s)
    probs = np.column_stack([neg, neu, pos])
    probs = probs / probs.sum(axis=1, keepdims=True)
    return labels.tolist(), probs
