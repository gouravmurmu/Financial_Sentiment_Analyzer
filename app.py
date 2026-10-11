"""Streamlit dashboard: news sentiment, signal research and sentiment-driven forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sentiment Lens", page_icon="📰", layout="wide")

from finsent import data, demo, index, models, predict, signals  # noqa: E402
from finsent.config import BENCHMARK, DEFAULT_TICKERS  # noqa: E402
from finsent.ui import charts, theme  # noqa: E402

theme.apply()
PERIODS = {"1 year": 1, "2 years": 2, "3 years": 3, "5 years": 5}
COMPANY_NAMES = {k: v[0] for k, v in demo.COMPANIES.items()}


# ---------------- data loading ----------------
@st.cache_data(show_spinner=False, ttl=900)
def load_market(source: str, tickers: tuple, years: int, upload: pd.DataFrame | None):
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).normalize()
    notes = []
    if source != "Demo":
        try:
            prices = data.fetch_prices(list(tickers) + [BENCHMARK], start)
            if source == "Live":
                news, errors = data.fetch_news(tickers, COMPANY_NAMES)
                if news.empty:
                    raise ValueError("no headlines came back from the news feeds")
                if errors:
                    notes.append("Some news feeds did not answer: " + "; ".join(errors[:3]))
            else:
                news = upload[upload["ticker"].isin(tickers)]
            return prices, news, "live", notes
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Live data unavailable ({exc}). Showing demo data instead.")
    prices, news = demo.market_data(list(tickers), start=start, benchmark=BENCHMARK)
    if source == "Upload" and upload is not None:
        news = upload[upload["ticker"].isin(tickers)]
        notes.append("Prices could not be downloaded, so the uploaded headlines are shown against simulated prices.")
    return prices, news, "demo", notes


@st.cache_resource(show_spinner=False)
def get_engine(backend: str, corpus_key: str, _corpus: pd.DataFrame | None):
    return models.SentimentEngine.train(backend, _corpus)


@st.cache_data(show_spinner=False)
def score_news(news: pd.DataFrame, backend: str, corpus_key: str) -> pd.DataFrame:
    eng = get_engine(backend, corpus_key, st.session_state.get("corpus"))
    return pd.concat([news.reset_index(drop=True), eng.score(news["headline"])], axis=1)


@st.cache_data(show_spinner=False)
def build_index(scored: pd.DataFrame, prices: pd.DataFrame, halflife: float):
    idx = index.daily_index(scored, prices.index, halflife=halflife)
    return idx, index.merge_returns(idx, prices, BENCHMARK)


@st.cache_data(show_spinner=False)
def forecast(frame: pd.DataFrame, kinds: tuple):
    return predict.run(frame, kinds=kinds, refit_every=42)


@st.cache_data(show_spinner=False)
def run_benchmark(corpus: pd.DataFrame, names: tuple):
    return models.benchmark(corpus, names)


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("## 📰 Sentiment Lens")
    source = st.radio("Data", ["Demo", "Live", "Upload"], horizontal=True,
                      help="Live pulls prices from Yahoo Finance and recent headlines from Yahoo Finance and "
                           "Google News. Upload lets you bring a headline archive for longer history.")
    upload = None
    if source == "Upload":
        f = st.file_uploader("Headlines CSV (date, headline, optional ticker/source)", type="csv")
        if f is not None:
            try:
                upload = data.parse_uploaded_news(pd.read_csv(f), DEFAULT_TICKERS[0])
                st.caption(f"{len(upload):,} headlines for {upload['ticker'].nunique()} tickers")
            except ValueError as exc:
                st.error(str(exc))
        if upload is None:
            st.info("Upload a file to continue. Until then the demo data is shown.")
            source = "Demo"
    uploaded = set(upload["ticker"]) if upload is not None else set()
    universe = sorted(set(DEFAULT_TICKERS) | set(demo.COMPANIES) | uploaded)
    tickers = st.multiselect("Watchlist", universe, default=DEFAULT_TICKERS if upload is None
                             else sorted(upload["ticker"].unique())[:6])
    extra = st.text_input("Add tickers", placeholder="e.g. AMD, NFLX")
    tickers = list(dict.fromkeys(tickers + [t.strip().upper() for t in extra.split(",") if t.strip()]))
    years = PERIODS[st.selectbox("History", list(PERIODS), index=2)]

    st.markdown("### Sentiment model")
    options = ["logreg", "svm", "nb", "lexicon"] + (["finbert"] if models.finbert_available() else [])
    backend = st.selectbox("Scorer", options, format_func=models.MODEL_NAMES.get,
                           help="FinBERT appears here when `transformers` and `torch` are installed.")
    if not models.finbert_available():
        st.caption("Install `requirements-finbert.txt` to add FinBERT.")
    halflife = st.slider("Index half-life (days)", 1.0, 10.0, 3.0, 0.5,
                         help="How quickly old headlines fade out of the daily sentiment index.")

if not tickers:
    st.warning("Pick at least one ticker in the sidebar.")
    st.stop()

corpus_key = st.session_state.get("corpus_key", "demo")
with st.spinner("Loading prices and headlines..."):
    prices, news, mode, notes = load_market(source, tuple(tickers), years, upload)
for n in notes:
    st.warning(n, icon="⚠️")
if news.empty:
    st.error("No headlines to analyse for these tickers.")
    st.stop()
with st.spinner(f"Scoring {len(news):,} headlines with {models.MODEL_NAMES[backend]}..."):
    scored = score_news(news, backend, corpus_key)
    idx, merged = build_index(scored, prices, halflife)

available = [t for t in tickers if t in set(merged.get("ticker", []))]
if not available:
    st.error("None of the selected tickers have both prices and headlines.")
    st.stop()

theme.hero("Sentiment Lens", "Score financial headlines, build a daily sentiment index, and test whether the "
           "news actually moves prices.",
           tag=("DEMO DATA" if mode == "demo" else "LIVE DATA") + f" · {len(news):,} HEADLINES")

focus = st.segmented_control("Ticker", available, default=available[0], label_visibility="collapsed") or available[0]
frame = merged[merged["ticker"] == focus].sort_values("date").reset_index(drop=True)
tnews = scored[scored["ticker"] == focus]
last = frame.iloc[-1]
recent = tnews[tnews["published"] >= tnews["published"].max() - pd.Timedelta(days=30)]

k = st.columns(5)
k[0].metric("Last close", f"${last['close']:,.2f}", f"{np.expm1(last['ret']):+.2%}")
prev = frame["index"].iloc[-6] if len(frame) > 6 else np.nan
k[1].metric("Sentiment index", f"{last['index']:+.2f}", f"{last['index'] - prev:+.2f} in 5 days"
            if prev == prev else None)
k[2].metric("Bullish / bearish (30d)", f"{(recent['label'] == 'positive').mean():.0%} / "
            f"{(recent['label'] == 'negative').mean():.0%}", help="Share of the last 30 days' headlines scored "
            "positive and negative.")
k[3].metric("Headlines (30d)", f"{len(recent):,}")
corr = frame[["daily", "ret"]].dropna().corr().iloc[0, 1] if frame["daily"].notna().sum() > 10 else np.nan
k[4].metric("News–return corr.", f"{corr:.2f}" if corr == corr else "n/a", help="Correlation of daily "
            "sentiment with that day's return.")

tabs = st.tabs(["📊 Overview", "📰 News feed", "🧪 Headline lab", "🔍 Signal lab", "🔮 Forecast"])

# ---------------- overview ----------------
with tabs[0]:
    st.plotly_chart(charts.price_sentiment(frame, focus), width="stretch")
    if len(available) > 1:
        st.plotly_chart(charts.watchlist_heatmap(idx[idx["ticker"].isin(available)]), width="stretch")
        theme.section("Watchlist", "Latest sentiment, its 5-day change and the 30-day price move.")
        rows = []
        for tk in available:
            g = merged[merged["ticker"] == tk].sort_values("date")
            rows.append({"Ticker": tk, "Sentiment": g["index"].iloc[-1],
                         "5d change": g["index"].iloc[-1] - g["index"].iloc[-6],
                         "30d return": np.expm1(g["ret"].tail(21).sum()),
                         "Headlines 30d": int(g["n"].tail(21).sum()),
                         "Trend": g["index"].tail(60).fillna(0).round(3).tolist()})
        st.dataframe(pd.DataFrame(rows).sort_values("Sentiment", ascending=False), hide_index=True,
                     width="stretch", column_config={
                         "Sentiment": st.column_config.NumberColumn(format="%+.2f"),
                         "5d change": st.column_config.NumberColumn(format="%+.2f"),
                         "30d return": st.column_config.NumberColumn(format="percent"),
                         "Trend": st.column_config.LineChartColumn("Sentiment, 60 days", y_min=-1, y_max=1)})

# ---------------- news feed ----------------
with tabs[1]:
    c1, c2, c3 = st.columns([2, 1, 1])
    q = c1.text_input("Search headlines", placeholder="earnings, guidance, recall...")
    labs = c2.pills("Sentiment", ["positive", "neutral", "negative"], selection_mode="multi",
                    default=["positive", "neutral", "negative"], format_func=str.title) or []
    who = c3.multiselect("Tickers", available, default=[focus])
    feed = scored[scored["ticker"].isin(who or available) & scored["label"].isin(labs)]
    if q:
        feed = feed[feed["headline"].str.contains(q, case=False, regex=False)]
    left, mid, right = st.columns(3)
    with left:
        st.plotly_chart(charts.label_donut(feed["label"]), width="stretch")
    with mid:
        st.plotly_chart(charts.by_source(feed), width="stretch")
    with right:
        st.plotly_chart(charts.by_hour(feed), width="stretch")
    view = feed.sort_values("published", ascending=False).head(500)
    show = pd.DataFrame({
        "Published": view["published"].dt.tz_localize(None), "Ticker": view["ticker"],
        "Headline": view["headline"],
        "Sentiment": view["label"].map({"positive": "🟢 Positive", "neutral": "🟡 Neutral", "negative": "🔴 Negative"}),
        "Score": view["score"], "Confidence": view["confidence"], "Source": view["source"],
    })
    if view["url"].astype(bool).any():
        show["Link"] = view["url"].where(view["url"].astype(bool))
    st.dataframe(show, hide_index=True, width="stretch", height=480, column_config={
        "Published": st.column_config.DatetimeColumn(format="D MMM YYYY, HH:mm"),
        "Score": st.column_config.NumberColumn(format="%+.2f", help="−1 bearish to +1 bullish"),
        "Confidence": st.column_config.NumberColumn(format="percent"),
        "Link": st.column_config.LinkColumn(display_text="open")})
    st.caption(f"Showing the latest {len(show):,} of {len(feed):,} matching headlines.")
    st.download_button("Download scored headlines (CSV)", feed.to_csv(index=False).encode(), "scored_headlines.csv",
                       "text/csv")

# ---------------- headline lab ----------------
with tabs[2]:
    theme.section("Try any headline", "Every available model scores it side by side.")
    text = st.text_area("Headlines (one per line)", "Apple beats estimates and raises full-year guidance\n"
                        "Tesla shares slide despite record deliveries\nNvidia to present at Morgan Stanley conference",
                        height=110)
    lines = [t.strip() for t in text.splitlines() if t.strip()]
    if lines:
        pick = st.selectbox("Headline to inspect", lines)
        rows = []
        for name in options:
            eng = get_engine(name, corpus_key, st.session_state.get("corpus"))
            r = eng.score([pick]).iloc[0]
            rows.append({"model": models.MODEL_NAMES[name], **r.to_dict()})
        res = pd.DataFrame(rows)
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.plotly_chart(charts.model_compare(res), width="stretch")
        with right:
            eng = get_engine(backend if backend in ("logreg", "lexicon") else "logreg", corpus_key,
                             st.session_state.get("corpus"))
            expl = eng.explain(pick)
            if expl.empty:
                st.info("No sentiment-bearing words found in this headline.")
            else:
                st.plotly_chart(charts.contributions(expl), width="stretch")
        batch = get_engine(backend, corpus_key, st.session_state.get("corpus")).score(lines)
        batch.insert(0, "headline", lines)
        st.dataframe(batch[["headline", "label", "score", "confidence"]], hide_index=True, width="stretch",
                     column_config={"score": st.column_config.NumberColumn(format="%+.2f"),
                                    "confidence": st.column_config.NumberColumn(format="percent")})

    theme.section("Model benchmark", "5-fold cross-validation on the labelled training corpus. Upload your own "
                  "labelled file (e.g. Financial PhraseBank) to retrain and re-benchmark every model.")
    lab_file = st.file_uploader("Labelled corpus CSV (text + label columns)", type="csv", key="corpus_upload")
    if lab_file is not None:
        try:
            corpus = data.parse_labelled_corpus(pd.read_csv(lab_file, encoding_errors="replace"))
            st.session_state["corpus"], st.session_state["corpus_key"] = corpus, f"upload-{lab_file.name}-{len(corpus)}"
            st.success(f"Training on {len(corpus):,} labelled headlines. Models retrain automatically.")
        except ValueError as exc:
            st.error(str(exc))
    corpus = st.session_state.get("corpus")
    if corpus is None:
        corpus = demo.labelled_corpus()
        st.caption("Using the built-in demo corpus (synthetic headlines with 6% label noise).")
    bench_models = tuple(m for m in options if m != "finbert")
    if st.toggle("Run benchmark", value=True):
        with st.spinner("Cross-validating models..."):
            bench = run_benchmark(corpus, bench_models)
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.plotly_chart(charts.benchmark_bars(bench["table"]), width="stretch")
        with right:
            which = st.selectbox("Confusion matrix", bench["table"]["key"], format_func=models.MODEL_NAMES.get)
            st.plotly_chart(charts.confusion(bench["confusion"][which], models.MODEL_NAMES[which]), width="stretch")
        st.dataframe(bench["table"].drop(columns="key"), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in bench["table"].columns if c not in ("model", "key")})

# ---------------- signal lab ----------------
with tabs[3]:
    sig = frame.set_index("date")
    usable = sig["daily"].notna().sum()
    if usable < 40:
        st.info(f"Only {usable} trading days have headlines for {focus}. Signal tests need a few months of news; "
                "upload a headline archive or switch to demo data.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            ll = signals.lead_lag(sig["daily"].fillna(0), sig["ret"])
            st.plotly_chart(charts.lead_lag_chart(ll), width="stretch")
        with c2:
            q5 = signals.quintile_returns(frame)
            if not q5.empty:
                st.plotly_chart(charts.quintile_chart(q5), width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            thr = st.slider("Shock threshold (sentiment surprise, σ)", 1.0, 3.0, 1.5, 0.25)
            es = signals.event_study(frame, threshold=thr)
            if es.empty:
                st.info("No news shocks at this threshold.")
            else:
                st.plotly_chart(charts.event_chart(es), width="stretch")
        with c2:
            window = st.slider("Rolling window (days)", 21, 252, 63, 21)
            st.plotly_chart(charts.rolling_corr_chart(signals.rolling_corr(sig["daily"], sig["ret"], window), window),
                            width="stretch")
        theme.section("Granger causality", "Does yesterday's sentiment help predict today's return beyond past "
                      "returns, and vice versa? p < 0.05 means yes at the 5% level.")
        gr = signals.granger(sig["index"].ffill().fillna(0), sig["ret"])
        if not gr.empty:
            gr["Verdict"] = np.where(gr["significant"], "✅ Significant", "— Not significant")
            st.dataframe(gr.drop(columns="significant"), hide_index=True, width="stretch",
                         column_config={"p_value": st.column_config.NumberColumn("p-value", format="%.4f"),
                                        "best_lag": "Best lag (days)", "direction": "Direction"})

# ---------------- forecast ----------------
with tabs[4]:
    if frame["next_ret"].notna().sum() < 320:
        st.info("Walk-forward forecasting needs at least 320 trading days of prices and news. Choose a longer "
                "history, upload a headline archive, or use demo data.")
    else:
        theme.section("Next-day direction models", "Each forecast uses a model trained only on earlier days "
                      "(expanding window, refit every two months). Price-only versions show what sentiment adds.")
        with st.spinner("Running walk-forward models..."):
            rep = forecast(frame, ("logit", "gbm"))
        sc = rep.scores.copy()
        st.dataframe(sc.rename(columns={"model": "Model", "uses_sentiment": "Uses news", "accuracy": "Accuracy",
                                        "auc": "ROC AUC", "brier": "Brier", "n": "Days tested"}),
                     hide_index=True, width="stretch",
                     column_config={"Accuracy": st.column_config.NumberColumn(format="percent"),
                                    "ROC AUC": st.column_config.NumberColumn(format="%.3f"),
                                    "Brier": st.column_config.NumberColumn(format="%.4f")})
        best = sc[sc["uses_sentiment"]].sort_values("auc", ascending=False)["model"].iloc[0]
        model_name = st.selectbox("Model to trade", list(rep.probs), index=list(rep.probs).index(best))
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            st.plotly_chart(charts.probability_chart(rep.probs[model_name], rep.features["close"]), width="stretch")
        with c2:
            if not rep.importance.empty:
                st.plotly_chart(charts.importance_chart(rep.importance), width="stretch")
        theme.section("Backtest", "Trade the forecast at each close and hold for one day. Not investment advice.")
        b1, b2, b3 = st.columns(3)
        band = b1.slider("Confidence band around 50%", 0.0, 0.1, 0.02, 0.01)
        cost = b2.slider("Cost per trade (bps)", 0.0, 30.0, 5.0, 1.0)
        long_only = b3.toggle("Long only", value=False)
        bt = predict.backtest(rep.probs[model_name], rep.features["next_ret"], band, cost, long_only)
        st.plotly_chart(charts.equity_chart(bt), width="stretch")
        s, b = predict.perf(bt["strategy"]), predict.perf(bt["buy_hold"])
        m = st.columns(5)
        m[0].metric("Strategy return", f"{s['total']:+.1%}", f"{s['total'] - b['total']:+.1%} vs B&H")
        m[1].metric("Sharpe", f"{s['sharpe']:.2f}", f"{s['sharpe'] - b['sharpe']:+.2f}")
        m[2].metric("Max drawdown", f"{s['max_dd']:.1%}", f"{s['max_dd'] - b['max_dd']:+.1%}")
        m[3].metric("Volatility", f"{s['vol']:.1%}", f"{s['vol'] - b['vol']:+.1%}", delta_color="inverse")
        m[4].metric("Time in market", f"{(bt['position'] != 0).mean():.0%}")

st.caption("Sentiment Lens · research tool, not investment advice · "
           "data: Yahoo Finance, Google News, or simulated demo data")
