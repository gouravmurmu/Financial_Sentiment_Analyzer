import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finsent import demo, index, models  # noqa: E402


@pytest.fixture(scope="session")
def corpus():
    return demo.labelled_corpus(n=1500)


@pytest.fixture(scope="session")
def engine(corpus):
    return models.SentimentEngine.train("logreg", corpus)


@pytest.fixture(scope="session")
def market():
    return demo.market_data(["AAPL", "TSLA"], start="2023-01-02", end="2025-06-30")


@pytest.fixture(scope="session")
def merged(market, engine):
    prices, news = market
    scored = news.reset_index(drop=True).join(engine.score(news["headline"]))
    return index.merge_returns(index.daily_index(scored, prices.index), prices, "SPY")
