"""Plotly figures for the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import AMBER, DIVERGING, EMERALD, GRID, INDIGO, MUTED, ROSE, SENTIMENT_COLORS, SKY, VIOLET


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def price_sentiment(df: pd.DataFrame, ticker: str, height: int = 560) -> go.Figure:
    """Price on top, sentiment index (green above zero, red below) and headline counts underneath."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.5, 0.3, 0.2])
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close", line=dict(color=INDIGO, width=2.2),
                             fill="tozeroy", fillcolor=_rgba(INDIGO, 0.06)), 1, 1)
    s = df["index"]
    fig.add_trace(go.Scatter(x=df["date"], y=s.clip(lower=0), name="Bullish", line=dict(width=0),
                             fill="tozeroy", fillcolor=_rgba(EMERALD, 0.45), hoverinfo="skip"), 2, 1)
    fig.add_trace(go.Scatter(x=df["date"], y=s.clip(upper=0), name="Bearish", line=dict(width=0),
                             fill="tozeroy", fillcolor=_rgba(ROSE, 0.45), hoverinfo="skip"), 2, 1)
    fig.add_trace(go.Scatter(x=df["date"], y=s, name="Sentiment index", line=dict(color="#3A3F58", width=1.4)),
                  2, 1)
    fig.add_trace(go.Bar(x=df["date"], y=df["n"], name="Headlines", marker_color=_rgba(VIOLET, 0.55)), 3, 1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", zeroline=True, row=2, col=1)
    fig.update_yaxes(title_text="News", row=3, col=1)
    lo = df["close"].min()
    fig.update_yaxes(range=[lo * 0.95, df["close"].max() * 1.02], row=1, col=1)
    fig.update_layout(height=height, title=f"{ticker}: price vs news sentiment", bargap=0.1)
    return fig


def watchlist_heatmap(idx: pd.DataFrame, weeks: int = 40) -> go.Figure:
    w = idx.set_index("date").groupby("ticker")["daily"].resample("W-FRI").mean().unstack(0).tail(weeks)
    z = w.T
    fig = go.Figure(go.Heatmap(z=z.values, x=[d.strftime("%d %b") for d in z.columns], y=z.index,
                               colorscale=DIVERGING, zmid=0, zmin=-0.6, zmax=0.6, xgap=2, ygap=2,
                               colorbar=dict(title="Tone", thickness=10),
                               hovertemplate="%{y} · week to %{x}<br>Average tone %{z:.2f}<extra></extra>"))
    fig.update_layout(height=90 + 46 * len(z), title="Weekly news tone across the watchlist", hovermode="closest")
    return fig


def label_donut(labels: pd.Series) -> go.Figure:
    counts = labels.value_counts().reindex(["positive", "neutral", "negative"]).fillna(0)
    fig = go.Figure(go.Pie(labels=[c.title() for c in counts.index], values=counts.values, hole=0.62, sort=False,
                           marker=dict(colors=[SENTIMENT_COLORS[c] for c in counts.index], line=dict(color="white",
                                                                                                    width=3))))
    fig.add_annotation(text=f"<b>{int(counts.sum()):,}</b><br>headlines", showarrow=False, font=dict(size=15))
    fig.update_layout(height=320, title="Headline mix", showlegend=True, margin=dict(t=60, b=10))
    return fig


def by_source(news: pd.DataFrame, top: int = 10) -> go.Figure:
    g = news.groupby(["source", "label"]).size().unstack(fill_value=0)
    g = g.loc[g.sum(axis=1).sort_values().tail(top).index]
    fig = go.Figure()
    for lab in ["negative", "neutral", "positive"]:
        if lab in g:
            fig.add_bar(y=g.index, x=g[lab], name=lab.title(), orientation="h", marker_color=SENTIMENT_COLORS[lab])
    fig.update_layout(barmode="stack", height=320, title="Headlines by source", hovermode="y unified")
    return fig


def by_hour(news: pd.DataFrame) -> go.Figure:
    h = news.assign(hour=news["published"].dt.hour).groupby(["hour", "label"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for lab in ["negative", "neutral", "positive"]:
        if lab in h:
            fig.add_bar(x=h.index, y=h[lab], name=lab.title(), marker_color=SENTIMENT_COLORS[lab])
    fig.add_vrect(x0=9.5, x1=16, fillcolor=_rgba(SKY, 0.1), line_width=0)
    fig.update_layout(barmode="stack", height=320, title="When news lands (shaded = open)",
                      xaxis=dict(dtick=2, title="Hour"))
    return fig


def model_compare(scores: pd.DataFrame) -> go.Figure:
    """Stacked probability bars, one row per model."""
    fig = go.Figure()
    for lab, col in (("negative", "p_negative"), ("neutral", "p_neutral"), ("positive", "p_positive")):
        fig.add_bar(y=scores["model"], x=scores[col], orientation="h", name=lab.title(),
                    marker_color=SENTIMENT_COLORS[lab], text=(scores[col] * 100).round(0).astype(int).astype(str) + "%",
                    textposition="inside")
    fig.update_layout(barmode="stack", height=90 + 46 * len(scores), xaxis=dict(range=[0, 1], tickformat=".0%"),
                      title="Class probabilities by model", hovermode="closest")
    return fig


def contributions(expl: pd.DataFrame) -> go.Figure:
    e = expl.sort_values("weight")
    fig = go.Figure(go.Bar(x=e["weight"], y=e["term"], orientation="h",
                           marker_color=[EMERALD if w > 0 else ROSE for w in e["weight"]]))
    fig.update_layout(height=60 + 30 * max(len(e), 3), title="Which words pushed the score", hovermode="closest")
    return fig


def benchmark_bars(table: pd.DataFrame) -> go.Figure:
    t = table.sort_values("macro_f1")
    fig = go.Figure()
    fig.add_bar(y=t["model"], x=t["accuracy"], name="Accuracy", orientation="h", marker_color=_rgba(INDIGO, 0.85))
    fig.add_bar(y=t["model"], x=t["macro_f1"], name="Macro F1", orientation="h", marker_color=_rgba(EMERALD, 0.85))
    fig.update_layout(barmode="group", height=110 + 55 * len(t), xaxis=dict(range=[0, 1], tickformat=".0%"),
                      title="Cross-validated accuracy", hovermode="closest")
    return fig


def confusion(cm: pd.DataFrame, title: str) -> go.Figure:
    pct = cm.div(cm.sum(axis=1), axis=0)
    fig = go.Figure(go.Heatmap(z=pct.values, x=[c.title() for c in cm.columns], y=[c.title() for c in cm.index],
                               colorscale=[[0, "#F8F9FC"], [1, INDIGO]], zmin=0, zmax=1, showscale=False,
                               text=cm.values, texttemplate="%{text}<br>%{z:.0%}", xgap=3, ygap=3))
    fig.update_layout(height=330, title=title, xaxis_title="Predicted", yaxis_title="Actual",
                      yaxis=dict(autorange="reversed"), hovermode="closest")
    return fig


def lead_lag_chart(ll: pd.DataFrame) -> go.Figure:
    colors = [EMERALD if k > 0 else INDIGO if k == 0 else AMBER for k in ll["lag"]]
    fig = go.Figure(go.Bar(x=ll["lag"], y=ll["corr"], marker_color=colors, customdata=ll["p_value"],
                           hovertemplate="lag %{x}: r = %{y:.3f}, p = %{customdata:.3f}<extra></extra>"))
    ci = ll["ci"].mean()
    fig.add_hrect(y0=-ci, y1=ci, fillcolor=_rgba(MUTED, 0.1), line_width=0)
    fig.update_layout(height=360, title="Sentiment today vs return k days later (grey = noise band)",
                      hovermode="closest",
                      xaxis=dict(title="k (negative: prices moved first · positive: sentiment leads)", dtick=1),
                      yaxis_title="Correlation")
    return fig


def rolling_corr_chart(rc: pd.Series, window: int) -> go.Figure:
    fig = go.Figure(go.Scatter(x=rc.index, y=rc.values, line=dict(color=VIOLET, width=2), fill="tozeroy",
                               fillcolor=_rgba(VIOLET, 0.1), name="Correlation"))
    fig.add_hline(y=0, line_color=GRID)
    fig.update_layout(height=320, title=f"Rolling {window}-day correlation: sentiment vs same-day return",
                      showlegend=False)
    return fig


def event_chart(es: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for kind, color in (("Positive shock", EMERALD), ("Negative shock", ROSE)):
        e = es[es["event"] == kind]
        if e.empty:
            continue
        fig.add_trace(go.Scatter(x=np.r_[e["day"], e["day"][::-1]], y=np.r_[e["hi"], e["lo"][::-1]],
                                 fill="toself", fillcolor=_rgba(color, 0.14), line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=e["day"], y=e["car"], name=f"{kind} ({int(e['events'].iloc[0])} events)",
                                 line=dict(color=color, width=2.6)))
    fig.add_vline(x=0, line_dash="dot", line_color=MUTED)
    fig.add_hline(y=0, line_color=GRID)
    fig.update_layout(height=380, title="Cumulative return around news shocks", yaxis_tickformat=".1%",
                      xaxis_title="Trading days from the news day")
    return fig


def quintile_chart(q: pd.DataFrame) -> go.Figure:
    colors = [ROSE, "#F48FA3", AMBER, "#6FD3B1", EMERALD]
    fig = go.Figure(go.Bar(x=[f"Q{int(i)}" for i in q["quintile"]], y=q["mean"], marker_color=colors,
                           error_y=dict(type="data", array=1.96 * q["se"], color=MUTED, thickness=1.2)))
    fig.update_layout(height=360, title="Next-day return by sentiment quintile", yaxis_tickformat=".2%",
                      xaxis_title="Q1 = most negative · Q5 = most positive", hovermode="closest")
    return fig


def probability_chart(prob: pd.Series, close: pd.Series) -> go.Figure:
    p = prob.dropna().tail(250)
    c = close.reindex(p.index)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=c.index, y=c.values, name="Close", line=dict(color=_rgba(INDIGO, 0.5), width=1.6)),
                  secondary_y=False)
    fig.add_trace(go.Bar(x=p.index, y=p.values - 0.5, base=0.5, name="P(up tomorrow)",
                         marker_color=[EMERALD if v > 0.5 else ROSE for v in p.values]), secondary_y=True)
    fig.update_yaxes(title_text="Price", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="P(up)", secondary_y=True, range=[0.3, 0.7], tickformat=".0%")
    fig.update_layout(height=380, title="Out-of-sample forecasts, last 250 days")
    return fig


def importance_chart(imp: pd.Series) -> go.Figure:
    sentiment_terms = ("Sentiment", "sentiment", "News", "Positive", "Today's sentiment")
    colors = [VIOLET if any(t in i for t in sentiment_terms) else SKY for i in imp.index]
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=colors))
    fig.update_layout(height=380, title="Model weights (violet = news features)",
                      hovermode="closest")
    return fig


def equity_chart(bt: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.index, y=bt["equity"], name="Sentiment strategy", line=dict(color=INDIGO, width=2.4)))
    fig.add_trace(go.Scatter(x=bt.index, y=bt["bh_equity"], name="Buy & hold", line=dict(color=MUTED, width=1.6,
                                                                                       dash="dot")))
    fig.update_layout(height=380, title="Growth of $1 (after costs)", yaxis_tickformat="$.2f")
    return fig


def score_hist(scores: pd.Series) -> go.Figure:
    fig = go.Figure(go.Histogram(x=scores, nbinsx=40, marker_color=_rgba(INDIGO, 0.75)))
    fig.update_layout(height=300, title="Distribution of headline scores", hovermode="closest", bargap=0.05,
                      xaxis_title="Score (−1 bearish · +1 bullish)")
    return fig


