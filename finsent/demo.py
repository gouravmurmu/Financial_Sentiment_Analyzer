"""Offline demo data: a labelled headline corpus and a simulated news + price history.

The simulation has a known structure so the research tools have something real to find: headlines
react strongly to same-day returns, and a slow-moving news "tone" factor carries a small amount of
information about the next day's return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPANIES = {
    "AAPL": ("Apple", "iPhone", "services"), "MSFT": ("Microsoft", "Azure", "cloud"),
    "NVDA": ("Nvidia", "GPU", "data center"), "TSLA": ("Tesla", "Model Y", "EV"),
    "JPM": ("JPMorgan", "investment banking", "lending"), "AMZN": ("Amazon", "AWS", "retail"),
    "GOOGL": ("Alphabet", "search", "advertising"), "META": ("Meta", "Reels", "advertising"),
}
PROFILES = {  # beta to market, daily idiosyncratic vol, news per day, sensitivity to tone
    "AAPL": (1.05, 0.011, 4.0, 0.0030), "MSFT": (1.0, 0.010, 3.5, 0.0028),
    "NVDA": (1.6, 0.022, 5.0, 0.0050), "TSLA": (1.5, 0.028, 6.0, 0.0060),
    "JPM": (1.1, 0.010, 2.5, 0.0022), "AMZN": (1.2, 0.014, 3.5, 0.0035),
    "GOOGL": (1.1, 0.013, 3.0, 0.0030), "META": (1.3, 0.018, 3.5, 0.0040),
}
SOURCES = ["Reuters", "Bloomberg", "CNBC", "MarketWatch", "Barron's", "Financial Times", "WSJ",
           "Yahoo Finance", "Investor's Business Daily", "Seeking Alpha"]

FILL = {
    "n": lambda r: str(r.integers(2, 40)), "pct": lambda r: str(r.integers(3, 45)),
    "px": lambda r: str(r.integers(80, 900)),
    "ord": lambda r: r.choice(["second", "third", "fourth", "fifth"]),
    "day": lambda r: r.choice(["Tuesday", "Wednesday", "Thursday"]),
    "conf": lambda r: r.choice(["Morgan Stanley", "Goldman Sachs", "JPMorgan", "Bernstein"]),
    "month": lambda r: r.choice(["March", "June", "September", "November"]),
}


def _render(template: str, ticker: str, rng: np.random.Generator) -> str:
    co, prod, seg = COMPANIES.get(ticker, (ticker, "flagship product", "core"))
    values = {"co": co, "prod": prod, "seg": seg}
    for key, fn in FILL.items():
        if "{" + key + "}" in template:
            values[key] = fn(rng)
    return template.format(**values)


# Building blocks for the labelled corpus: subject moves and reasons combine into many phrasings.
MOVES = {
    "positive": ["shares surge", "stock jumps", "shares climb", "rallies", "hits record high", "gains",
                 "beats estimates", "raises guidance", "lifts outlook", "is upgraded to buy", "tops forecasts",
                 "posts record profit", "shares rebound", "outperforms the market", "soars"],
    "negative": ["shares plunge", "stock tumbles", "shares slide", "sinks", "hits 52-week low", "falls",
                 "misses estimates", "cuts guidance", "lowers outlook", "is downgraded to sell",
                 "falls short of forecasts", "swings to a loss", "shares slump", "lags the market", "plummets"],
    "neutral": ["shares little changed", "stock flat", "trades sideways", "to report results",
                "to present at conference", "names new CFO", "files quarterly report", "holds investor day",
                "confirms event date", "completes reorganisation", "updates product lineup",
                "schedules earnings call", "reaffirms guidance", "keeps dividend unchanged", "trades in line"],
}
REASONS = {
    "positive": ["on strong {seg} demand", "as {prod} sales accelerate", "after upbeat outlook",
                 "on margin expansion", "after a ${n} billion buyback", "as {seg} growth beats expectations",
                 "after winning a major contract", "on robust holiday sales",
                 "after regulators approve new {prod} plans",
                 "as costs fall"],
    "negative": ["on weak {seg} demand", "as {prod} sales slow", "after a gloomy outlook",
                 "on shrinking margins", "after a regulatory probe", "as {seg} growth disappoints",
                 "after losing a major contract", "amid layoffs", "after a recall of {prod} units", "as costs climb"],
    "neutral": ["ahead of the Fed decision", "as investors await jobs data", "in quiet trading",
                "before earnings next week", "in line with expectations", "as planned", "this {day}",
                "at the {conf} conference", "as markets close mixed", "after the long weekend"],
}


def headline(label: str, ticker: str, rng: np.random.Generator) -> str:
    """One synthetic headline with the given sentiment.

    Headlines combine a price move or event with a reason. About a fifth of positive and negative ones
    are contrast headlines ("rises despite weak demand"), labelled by the move, which simple word
    counting gets wrong.
    """
    move = str(rng.choice(MOVES[label]))
    if label != "neutral" and rng.random() < 0.2:
        other = "negative" if label == "positive" else "positive"
        reason = "despite " + str(rng.choice(REASONS[other])).split(" ", 1)[1]
    else:
        reason = str(rng.choice(REASONS[label] + REASONS["neutral"][:3]))
    co = COMPANIES.get(ticker, (ticker,))[0]
    text = _render(f"{co} {move} {reason}", ticker, rng)
    return text[0].upper() + text[1:]


def labelled_corpus(n: int = 4000, noise: float = 0.06, seed: int = 7) -> pd.DataFrame:
    """Labelled headlines for training and benchmarking classifiers.

    See :func:`headline` for how sentences are built. A share of labels is flipped to mimic annotator
    disagreement, so no model can score 100%.
    """
    rng = np.random.default_rng(seed)
    tickers = list(COMPANIES)
    labels = list(MOVES)
    rows = []
    for i in range(n):
        label = str(rng.choice(labels, p=[0.3, 0.3, 0.4]))
        text = headline(label, tickers[i % len(tickers)], rng)
        if rng.random() < noise:
            label = str(rng.choice([lab for lab in labels if lab != label]))
        rows.append((text, label))
    return pd.DataFrame(rows, columns=["headline", "label"])


def market_data(tickers, start="2022-01-03", end=None, seed: int = 11,
                benchmark: str = "SPY") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulated adjusted close prices and timestamped headlines for ``tickers``.

    Returns (prices, news). ``news`` has columns ticker, published (tz-aware, New York), headline,
    source and true_label.
    """
    rng = np.random.default_rng(seed)
    end = end or pd.Timestamp.today().normalize()
    days = pd.bdate_range(start, end)
    n = len(days)

    # Market with two volatility regimes.
    regime = np.zeros(n, dtype=int)
    for t in range(1, n):
        stay = 0.985 if regime[t - 1] == 0 else 0.95
        regime[t] = regime[t - 1] if rng.random() < stay else 1 - regime[t - 1]
    mkt = np.where(regime == 0, rng.normal(0.0005, 0.008, n), rng.normal(-0.0008, 0.018, n))

    prices = {benchmark: 400 * np.exp(np.cumsum(mkt))}
    news = []
    for k, tk in enumerate(tickers):
        beta, ivol, rate, gamma = PROFILES.get(tk, (1.0, 0.015, 3.0, 0.003))
        # Slow-moving news tone: AR(1) that leads returns by one day.
        tone = np.zeros(n)
        for t in range(1, n):
            tone[t] = 0.85 * tone[t - 1] + rng.normal(0, 0.55)
        ret = beta * mkt + rng.normal(0, ivol, n)
        ret[1:] += gamma * np.tanh(tone[:-1])
        prices[tk] = (100 + 50 * k) * np.exp(np.cumsum(ret))

        z_ret = ret / ret.std()
        for t, day in enumerate(days):
            count = rng.poisson(rate * (1 + 0.6 * abs(z_ret[t])))
            for _ in range(count):
                # Mostly headlines about today's move, partly about the forward-looking tone.
                drive = 0.75 * z_ret[t] + 0.6 * tone[t] + rng.normal(0, 0.7)
                p_pos = 1 / (1 + np.exp(-(drive - 0.6)))
                p_neg = 1 / (1 + np.exp(drive + 0.6))
                u = rng.random()
                label = "positive" if u < p_pos else "negative" if u < p_pos + p_neg else "neutral"
                hour = rng.choice([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
                ts = day + pd.Timedelta(hours=int(hour), minutes=int(rng.integers(0, 60)))
                news.append((tk, ts, headline(label, tk, rng),
                             SOURCES[rng.integers(len(SOURCES))], label))

    price_df = pd.DataFrame(prices, index=days)
    price_df.index.name = "date"
    news_df = pd.DataFrame(news, columns=["ticker", "published", "headline", "source", "true_label"])
    news_df["published"] = news_df["published"].dt.tz_localize("America/New_York")
    news_df["url"] = ""
    return price_df, news_df.sort_values("published").reset_index(drop=True)
