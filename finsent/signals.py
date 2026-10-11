"""Does news sentiment lead prices? Lead-lag correlations, Granger tests, rolling correlation and event studies."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


def lead_lag(sent: pd.Series, ret: pd.Series, max_lag: int = 5) -> pd.DataFrame:
    """Correlation of sentiment on day t with returns on day t+k.

    Positive k means sentiment comes first (it would help forecast); negative k means prices moved first
    and the news reacted.
    """
    rows = []
    for k in range(-max_lag, max_lag + 1):
        pair = pd.concat([sent, ret.shift(-k)], axis=1).dropna()
        if len(pair) < 10:
            continue
        r, p = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
        rows.append({"lag": k, "corr": r, "p_value": p, "n": len(pair), "ci": 1.96 / np.sqrt(len(pair))})
    return pd.DataFrame(rows)


def granger(sent: pd.Series, ret: pd.Series, max_lag: int = 5) -> pd.DataFrame:
    """Granger causality F-tests in both directions; the smallest p-value across lags is reported."""
    from statsmodels.tsa.stattools import grangercausalitytests

    data = pd.concat([ret.rename("ret"), sent.rename("sent")], axis=1).dropna()
    if len(data) < 6 * max_lag + 10:
        return pd.DataFrame()
    rows = []
    for name, cols in (("Sentiment → returns", ["ret", "sent"]), ("Returns → sentiment", ["sent", "ret"])):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(data[cols], maxlag=max_lag)
        pvals = {lag: r[0]["ssr_ftest"][1] for lag, r in res.items()}
        best = min(pvals, key=pvals.get)
        rows.append({"direction": name, "best_lag": best, "p_value": pvals[best],
                     "significant": pvals[best] < 0.05})
    return pd.DataFrame(rows)


def rolling_corr(sent: pd.Series, ret: pd.Series, window: int = 63) -> pd.Series:
    return sent.rolling(window, min_periods=window // 2).corr(ret)


def event_study(df: pd.DataFrame, signal: str = "surprise", threshold: float = 1.5, pre: int = 5,
                post: int = 10, ret_col: str = "ret") -> pd.DataFrame:
    """Average cumulative return around strongly positive and strongly negative news days.

    ``df`` is one ticker's daily frame. Event windows that overlap an earlier event of the same sign are
    skipped so one news burst is counted once. Returns mean and 95% band of cumulative return by day.
    """
    df = df.reset_index(drop=True)
    r = df[ret_col].to_numpy()
    out = []
    for kind, mask in (("Positive shock", df[signal] >= threshold), ("Negative shock", df[signal] <= -threshold)):
        paths, last = [], -10 ** 9
        for t in np.flatnonzero(mask.to_numpy()):
            if t - pre < 0 or t + post >= len(df) or t - last <= post:
                continue
            last = t
            window = np.nan_to_num(r[t - pre:t + post + 1])
            # Cumulative return relative to the close before the event day.
            cum = np.cumsum(window) - np.cumsum(window)[pre - 1]
            paths.append(cum)
        if not paths:
            continue
        m = np.array(paths)
        se = m.std(axis=0, ddof=1) / np.sqrt(len(m)) if len(m) > 1 else np.zeros(m.shape[1])
        for i, day in enumerate(range(-pre, post + 1)):
            out.append({"event": kind, "day": day, "car": m[:, i].mean(), "lo": m[:, i].mean() - 1.96 * se[i],
                        "hi": m[:, i].mean() + 1.96 * se[i], "events": len(m)})
    return pd.DataFrame(out)


def quintile_returns(df: pd.DataFrame, signal: str = "index", ret_col: str = "next_ret") -> pd.DataFrame:
    """Average next-day return for each sentiment quintile (1 = most negative)."""
    d = df[[signal, ret_col]].dropna()
    if len(d) < 25:
        return pd.DataFrame()
    d["quintile"] = pd.qcut(d[signal].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    g = d.groupby("quintile", observed=True)[ret_col]
    return pd.DataFrame({"mean": g.mean(), "se": g.std() / np.sqrt(g.count()), "n": g.count()}).reset_index()
