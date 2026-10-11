"""Light text normalisation that keeps the words that carry financial meaning."""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Words a generic stop list removes but that flip or carry sentiment in finance headlines.
KEEP = {"no", "not", "nor", "never", "down", "up", "above", "below", "against", "off", "over", "under",
        "less", "more", "most", "few", "cannot", "without", "only", "top", "bottom"}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS - KEEP)
NEGATIONS = frozenset({"no", "not", "never", "without", "cannot", "nor", "neither", "fails", "failed"})

_TOKEN = re.compile(r"[a-z][a-z0-9']*|\d+(?:\.\d+)?%?")
_URL = re.compile(r"https?://\S+")
_TICKER = re.compile(r"\$[A-Za-z]{1,5}\b")


def normalise(text: str) -> str:
    """Lowercase, drop URLs and cashtags, unify percent and quotes."""
    if not isinstance(text, str):
        return ""
    text = _URL.sub(" ", text)
    text = _TICKER.sub(" ", text)
    text = text.replace("’", "'").replace("‘", "'").replace("per cent", "%").replace("percent", "%")
    return text.lower()


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalise(text))


def clean_text(text: str, drop_stopwords: bool = True) -> str:
    tokens = tokenize(text)
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)
