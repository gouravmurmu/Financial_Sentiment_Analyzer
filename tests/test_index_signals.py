import numpy as np
import pandas as pd

from finsent import index, predict, signals


def test_after_close_and_weekend_news_moves_to_next_session():
    days = pd.bdate_range("2024-01-01", "2024-01-10")
    ts = pd.Series(pd.to_datetime(["2024-01-02 10:00", "2024-01-02 16:30", "2024-01-06 12:00"])
                   .tz_localize("America/New_York"))
    out = index.assign_session(ts, days)
    assert list(out.dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-08"]


def test_daily_index_has_one_row_per_ticker_day(market, merged):
    prices, _ = market
    assert merged.groupby("ticker").size().max() <= len(prices)
    assert merged["index"].dropna().between(-1, 1).all()
    assert {"ret", "next_ret", "mkt", "surprise"} <= set(merged.columns)


def test_sentiment_reacts_to_same_day_returns(merged):
    g = merged[merged["ticker"] == "TSLA"].set_index("date")
    ll = signals.lead_lag(g["daily"].fillna(0), g["ret"], 3)
    assert ll.loc[ll["lag"] == 0, "corr"].item() > 0.2
    assert ll.loc[ll["lag"] == 0, "corr"].item() == ll["corr"].max()


def test_granger_and_event_study(merged):
    g = merged[merged["ticker"] == "TSLA"]
    gr = signals.granger(g.set_index("date")["index"].ffill().fillna(0), g.set_index("date")["ret"])
    assert len(gr) == 2 and gr["p_value"].between(0, 1).all()
    es = signals.event_study(g, threshold=1.5)
    assert set(es["event"]) == {"Positive shock", "Negative shock"}
    day0 = es[es["day"] == 0].set_index("event")["car"]
    assert day0["Positive shock"] > 0 > day0["Negative shock"]
    assert (es.loc[es["day"] == -1, "car"].abs() < 1e-12).all()


def test_walk_forward_is_out_of_sample(merged):
    g = merged[merged["ticker"] == "AAPL"]
    feat = predict.make_features(g)
    probs = predict.walk_forward(feat, predict.PRICE_FEATURES + predict.SENTIMENT_FEATURES, "logit",
                                 min_train=250, refit_every=42)
    tested = probs.dropna()
    first_train_end = feat.dropna(subset=predict.PRICE_FEATURES + predict.SENTIMENT_FEATURES).index[250]
    assert tested.index.min() == first_train_end
    assert tested.between(0, 1).all()


def test_run_reports_baselines_and_backtest(merged):
    rep = predict.run(merged[merged["ticker"] == "AAPL"], kinds=("logit",), refit_every=63)
    assert {"Always predict 'up'"} <= set(rep.scores["model"])
    assert rep.scores["uses_sentiment"].sum() == 1
    p = rep.probs["Logistic regression · sentiment + price"]
    bt = predict.backtest(p, rep.features["next_ret"], band=0.0, cost_bps=0)
    # With no band and no costs, every tested day is long or short.
    assert set(np.unique(bt["position"])) <= {-1.0, 1.0}
    stats = predict.perf(bt["strategy"])
    assert stats["max_dd"] <= 0 and np.isfinite(stats["sharpe"])
