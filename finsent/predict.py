"""Walk-forward next-day direction models that use sentiment, and a simple trading backtest.

Every prediction is made with a model fitted only on earlier days. A price-only model is trained the
same way, so the dashboard can show whether sentiment adds anything beyond momentum and volatility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SENTIMENT_FEATURES = ["index", "daily_f", "surprise", "breadth", "volume_z", "index_chg5"]
PRICE_FEATURES = ["ret", "ret_5", "ret_20", "vol_20", "mkt_ret"]
MODELS = {
    "logit": "Logistic regression",
    "gbm": "Gradient boosting",
}
FEATURE_LABELS = {
    "index": "Sentiment index", "daily_f": "Today's sentiment", "surprise": "Sentiment surprise",
    "breadth": "Positive minus negative share", "volume_z": "News volume (z)", "index_chg5": "5-day sentiment change",
    "ret": "Today's return", "ret_5": "5-day return", "ret_20": "20-day return", "vol_20": "20-day volatility",
    "mkt_ret": "Market return",
}


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features known at today's close and the target (next-day up move) for one ticker."""
    f = df.sort_values("date").set_index("date").copy()
    f["daily_f"] = f["daily"].fillna(0)
    f["index"] = f["index"].ffill().fillna(0)
    f["index_chg5"] = f["index"].diff(5)
    f["ret_5"] = f["ret"].rolling(5).sum()
    f["ret_20"] = f["ret"].rolling(20).sum()
    f["vol_20"] = f["ret"].rolling(20).std()
    f["mkt_ret"] = f["mkt"] if "mkt" in f else 0.0
    f["target"] = (f["next_ret"] > 0).astype(float).where(f["next_ret"].notna())
    return f


def _model(kind: str):
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=100, min_samples_leaf=40,
                                              l2_regularization=1.0, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))


def walk_forward(feat: pd.DataFrame, features: list[str], kind: str = "logit", min_train: int = 250,
                 refit_every: int = 21) -> pd.Series:
    """Out-of-sample P(up tomorrow) using an expanding training window refitted every ``refit_every`` days."""
    data = feat.dropna(subset=features)
    probs = pd.Series(np.nan, index=data.index)
    labelled = data["target"].notna().to_numpy()
    for start in range(min_train, len(data), refit_every):
        train = data.iloc[:start][labelled[:start]]
        if train["target"].nunique() < 2:
            continue
        model = _model(kind).fit(train[features], train["target"])
        block = data.iloc[start:start + refit_every]
        probs.loc[block.index] = model.predict_proba(block[features])[:, 1]
    return probs


def score(prob: pd.Series, target: pd.Series) -> dict:
    d = pd.concat([prob.rename("p"), target.rename("y")], axis=1).dropna()
    if len(d) < 20 or d["y"].nunique() < 2:
        return {"accuracy": np.nan, "auc": np.nan, "brier": np.nan, "n": len(d)}
    return {"accuracy": accuracy_score(d["y"], d["p"] > 0.5), "auc": roc_auc_score(d["y"], d["p"]),
            "brier": brier_score_loss(d["y"], d["p"]), "n": len(d)}


def backtest(prob: pd.Series, next_ret: pd.Series, band: float = 0.02, cost_bps: float = 5.0,
             long_only: bool = False) -> pd.DataFrame:
    """Hold long when P(up) > 0.5 + band, short (or flat) when below 0.5 - band; pay costs on position changes."""
    d = pd.concat([prob.rename("p"), next_ret.rename("r")], axis=1).dropna()
    pos = np.where(d["p"] > 0.5 + band, 1.0, np.where(d["p"] < 0.5 - band, 0.0 if long_only else -1.0, 0.0))
    d["position"] = pos
    turnover = np.abs(np.diff(np.r_[0.0, pos]))
    d["strategy"] = d["position"] * (np.exp(d["r"]) - 1) - turnover * cost_bps / 1e4
    d["buy_hold"] = np.exp(d["r"]) - 1
    d["equity"] = (1 + d["strategy"]).cumprod()
    d["bh_equity"] = (1 + d["buy_hold"]).cumprod()
    return d


def perf(daily: pd.Series) -> dict:
    daily = daily.dropna()
    if daily.empty:
        return {}
    eq = (1 + daily).cumprod()
    years = len(daily) / 252
    vol = daily.std() * np.sqrt(252)
    return {"total": eq.iloc[-1] - 1, "cagr": eq.iloc[-1] ** (1 / max(years, 1e-9)) - 1, "vol": vol,
            "sharpe": daily.mean() * 252 / vol if vol > 0 else np.nan,
            "max_dd": (eq / eq.cummax() - 1).min(), "hit_rate": (daily[daily != 0] > 0).mean()}


@dataclass
class ForecastReport:
    probs: dict          # model name -> out-of-sample probability series
    scores: pd.DataFrame
    importance: pd.Series
    features: pd.DataFrame


def run(df: pd.DataFrame, kinds=("logit", "gbm"), min_train: int = 250, refit_every: int = 21) -> ForecastReport:
    """Fit sentiment-plus-price models and price-only baselines walk-forward for one ticker."""
    feat = make_features(df)
    full = SENTIMENT_FEATURES + PRICE_FEATURES
    probs, rows = {}, []
    for kind in kinds:
        for label, cols in ((f"{MODELS[kind]} · sentiment + price", full),
                            (f"{MODELS[kind]} · price only", PRICE_FEATURES)):
            p = walk_forward(feat, cols, kind, min_train, refit_every)
            probs[label] = p
            rows.append({"model": label, "uses_sentiment": cols is full, **score(p, feat["target"])})
    base = feat["target"].dropna()
    tested = next(iter(probs.values())).dropna().index if probs else base.index
    up_rate = base.loc[base.index.intersection(tested)].mean() if len(tested) else np.nan
    rows.append({"model": "Always predict 'up'", "uses_sentiment": False,
                 "accuracy": max(up_rate, 1 - up_rate) if up_rate == up_rate else np.nan, "auc": 0.5,
                 "brier": up_rate * (1 - up_rate), "n": len(tested)})
    scores = pd.DataFrame(rows)

    train = feat.dropna(subset=full + ["target"])
    imp = pd.Series(dtype=float)
    if len(train) > 50 and train["target"].nunique() == 2:
        logit = _model("logit").fit(train[full], train["target"])
        coef = logit[-1].coef_[0]
        imp = pd.Series(coef, index=[FEATURE_LABELS[c] for c in full]).sort_values(key=abs)
    return ForecastReport(probs, scores, imp, feat)
