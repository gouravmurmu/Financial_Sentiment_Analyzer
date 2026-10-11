"""Turn scored headlines into a daily sentiment index aligned with trading sessions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MARKET_CLOSE_HOUR, MARKET_TZ


def assign_session(published: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Trading day whose close first reflects each headline.

    News published after the close, or on a weekend or holiday, belongs to the next session.
    """
    ts = pd.to_datetime(published)
    ts = ts.dt.tz_convert(MARKET_TZ) if ts.dt.tz is not None else ts
    local = ts.dt.tz_localize(None) if ts.dt.tz is not None else ts
    day = local.dt.normalize()
    after_close = local.dt.hour >= MARKET_CLOSE_HOUR
    day = day.where(~after_close, day + pd.Timedelta(days=1))
    days = pd.DatetimeIndex(trading_days).sort_values()
    pos = days.searchsorted(day.to_numpy(), side="left")
    valid = pos < len(days)
    out = pd.Series(pd.NaT, index=published.index, dtype="datetime64[ns]")
    out[valid] = days[pos[valid]]
    return out


def daily_index(scored: pd.DataFrame, trading_days: pd.DatetimeIndex, halflife: float = 3.0,
                window: int = 20) -> pd.DataFrame:
    """Daily sentiment features per ticker.

    ``scored`` needs ticker, published, score, label and confidence. The index is a confidence-weighted
    average of headline scores, smoothed with an exponential decay so quiet days keep the recent tone.
    """
    df = scored.copy()
    df["date"] = assign_session(df["published"], trading_days)
    df = df.dropna(subset=["date"])
    df["w"] = df["confidence"].clip(lower=0.05)
    df["ws"] = df["w"] * df["score"]
    days = pd.DatetimeIndex(trading_days).sort_values()
    out = []
    for tk, g in df.groupby("ticker"):
        agg = g.groupby("date").agg(
            n=("score", "size"), raw=("score", "mean"), ws=("ws", "sum"), w=("w", "sum"),
            pos=("label", lambda s: (s == "positive").mean()), neg=("label", lambda s: (s == "negative").mean()),
        ).reindex(days)
        agg["n"] = agg["n"].fillna(0)
        agg[["ws", "w"]] = agg[["ws", "w"]].fillna(0)
        alpha = 1 - np.exp(np.log(0.5) / halflife)
        agg["index"] = agg["ws"].ewm(alpha=alpha, adjust=False).mean() / \
            agg["w"].ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)
        agg["daily"] = (agg["ws"] / agg["w"].replace(0, np.nan))
        roll = agg["daily"].rolling(window, min_periods=5)
        mean, std = roll.mean(), roll.std()
        agg["surprise"] = ((agg["daily"] - mean.shift(1)) / std.shift(1)).fillna(0).clip(-5, 5)
        vm, vs = agg["n"].rolling(window, min_periods=5).mean(), agg["n"].rolling(window, min_periods=5).std()
        agg["volume_z"] = ((agg["n"] - vm.shift(1)) / vs.shift(1).replace(0, np.nan)).fillna(0).clip(-5, 5)
        agg["breadth"] = (agg["pos"] - agg["neg"]).fillna(0)
        agg["ticker"] = tk
        out.append(agg.drop(columns=["ws", "w"]).rename_axis("date").reset_index())
    cols = ["ticker", "date", "n", "raw", "daily", "index", "surprise", "volume_z", "breadth", "pos", "neg"]
    if not out:
        return pd.DataFrame(columns=cols)
    return pd.concat(out, ignore_index=True)[cols]


def merge_returns(idx: pd.DataFrame, prices: pd.DataFrame, benchmark: str | None = None) -> pd.DataFrame:
    """Join each ticker's daily sentiment with its log return, next-day return and market return."""
    rets = np.log(prices).diff()
    frames = []
    for tk, g in idx.groupby("ticker"):
        if tk not in rets:
            continue
        r = rets[tk].rename("ret")
        f = g.set_index("date").join(r, how="inner")
        f["close"] = prices[tk].reindex(f.index)
        f["next_ret"] = f["ret"].shift(-1)
        if benchmark and benchmark in rets:
            f["mkt"] = rets[benchmark].reindex(f.index)
            f["excess"] = f["ret"] - f["mkt"]
        frames.append(f.reset_index())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
