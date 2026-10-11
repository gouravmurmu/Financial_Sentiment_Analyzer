"""Live data: adjusted prices from Yahoo Finance and headlines from Yahoo Finance and Google News RSS.

Free news feeds only reach back a few weeks, so for long-horizon research upload a headline archive
(any CSV with a date and headline column) or use the demo data.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import pandas as pd
import requests

from .config import MARKET_TZ

log = logging.getLogger(__name__)
NEWS_COLUMNS = ["ticker", "published", "headline", "source", "url"]
_UA = {"User-Agent": "Mozilla/5.0 (finsent research dashboard)"}


def fetch_prices(tickers, start, end=None) -> pd.DataFrame:
    """Adjusted daily closes, one column per ticker."""
    import yfinance as yf

    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False, threads=True)
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no prices")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(close, pd.DataFrame):
        close = close.to_frame()
    if close.shape[1] == 1 and len(tickers) == 1:
        close.columns = list(tickers)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "date"
    return close.dropna(how="all")


def _yahoo_news(ticker: str) -> list[dict]:
    import yfinance as yf

    items = yf.Ticker(ticker).get_news(count=100) or []
    rows = []
    for it in items:
        c = it.get("content", it)
        title = c.get("title")
        when = c.get("pubDate") or c.get("providerPublishTime")
        if not title or not when:
            continue
        if isinstance(when, (int, float)):
            ts = pd.to_datetime(when, unit="s", utc=True)
        else:
            ts = pd.to_datetime(when, utc=True)
        url = (c.get("canonicalUrl") or {}).get("url") or c.get("link", "")
        source = (c.get("provider") or {}).get("displayName") or c.get("publisher", "Yahoo Finance")
        rows.append({"ticker": ticker, "published": ts, "headline": title, "source": source, "url": url})
    return rows


def _google_news(ticker: str, company: str | None = None) -> list[dict]:
    query = quote_plus(f'"{company or ticker}" stock')
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    resp = requests.get(url, headers=_UA, timeout=15)
    resp.raise_for_status()
    rows = []
    for item in ET.fromstring(resp.content).iter("item"):
        title = item.findtext("title") or ""
        source = item.findtext("source") or "Google News"
        if title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        try:
            ts = pd.Timestamp(parsedate_to_datetime(item.findtext("pubDate"))).tz_convert("UTC")
        except (TypeError, ValueError):
            continue
        rows.append({"ticker": ticker, "published": ts, "headline": title, "source": source,
                     "url": item.findtext("link") or ""})
    return rows


def fetch_news(tickers, companies: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Recent headlines for each ticker from every source that answers. Returns (news, errors)."""
    rows, errors = [], []
    for tk in tickers:
        for name, fn in (("Yahoo Finance", lambda t=tk: _yahoo_news(t)),
                         ("Google News", lambda t=tk: _google_news(t, (companies or {}).get(t)))):
            try:
                rows += fn()
            except Exception as exc:  # noqa: BLE001 - any feed failure should not stop the others
                errors.append(f"{name} ({tk}): {exc}")
                log.warning("news fetch failed for %s via %s: %s", tk, name, exc)
    news = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    if news.empty:
        return news, errors
    news["published"] = pd.to_datetime(news["published"], utc=True).dt.tz_convert(MARKET_TZ)
    news["key"] = news["headline"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    news = news.drop_duplicates(["ticker", "key"]).drop(columns="key")
    return news.sort_values("published").reset_index(drop=True), errors


def parse_uploaded_news(df: pd.DataFrame, default_ticker: str) -> pd.DataFrame:
    """Normalise a user CSV with a date/time column and a headline/title/text column."""
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = next((cols[c] for c in ("headline", "title", "text", "sentence", "news") if c in cols), None)
    date_col = next((cols[c] for c in ("published", "datetime", "date", "time", "timestamp") if c in cols), None)
    if text_col is None or date_col is None:
        raise ValueError("The file needs a date column and a headline (or title/text) column")
    out = pd.DataFrame({"headline": df[text_col].astype(str),
                        "published": pd.to_datetime(df[date_col], errors="coerce")})
    tick_col = next((cols[c] for c in ("ticker", "symbol", "stock") if c in cols), None)
    out["ticker"] = df[tick_col].astype(str).str.upper() if tick_col else default_ticker
    out["source"] = df[cols["source"]].astype(str) if "source" in cols else "Upload"
    out["url"] = df[cols["url"]].astype(str) if "url" in cols else ""
    out = out.dropna(subset=["published"])
    ts = out["published"]
    out["published"] = ts.dt.tz_localize(MARKET_TZ) if ts.dt.tz is None else ts.dt.tz_convert(MARKET_TZ)
    return out[NEWS_COLUMNS].sort_values("published").reset_index(drop=True)


def parse_labelled_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a labelled training file (e.g. Financial PhraseBank) to headline/label columns."""
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = next((cols[c] for c in ("headline", "sentence", "text", "title") if c in cols), None)
    label_col = next((cols[c] for c in ("label", "sentiment", "class") if c in cols), None)
    if text_col is None or label_col is None:
        raise ValueError("The file needs a text column (headline/sentence/text) and a label column")
    mapping = {"pos": "positive", "neg": "negative", "neu": "neutral", "1": "positive", "-1": "negative",
               "0": "neutral", "bullish": "positive", "bearish": "negative"}
    labels = df[label_col].astype(str).str.lower().str.strip().replace(mapping)
    out = pd.DataFrame({"headline": df[text_col].astype(str), "label": labels})
    return out[out["label"].isin(["positive", "neutral", "negative"])].reset_index(drop=True)
