"""Light, vibrant theme: indigo for the brand, emerald / rose / amber for positive / negative / neutral."""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

INDIGO = "#5B5BF7"
EMERALD = "#12B886"
ROSE = "#F0506E"
AMBER = "#F5A524"
SKY = "#22B8CF"
VIOLET = "#A55EEA"
INK = "#1E2235"
MUTED = "#6B7185"
GRID = "#ECEEF5"
BG = "#F6F7FB"

SENTIMENT_COLORS = {"positive": EMERALD, "neutral": AMBER, "negative": ROSE}
SERIES = [INDIGO, EMERALD, ROSE, AMBER, SKY, VIOLET, "#FF7A45", "#2F9E44"]
DIVERGING = [[0, ROSE], [0.5, "#F8F9FC"], [1, EMERALD]]

pio.templates["finsent"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color=INK, size=13),
    title=dict(font=dict(size=16, color=INK), x=0.01, xanchor="left", y=0.97, yanchor="top"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=SERIES,
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, ticks=""),
    yaxis=dict(gridcolor=GRID, zerolinecolor="#D9DCE8", linecolor=GRID, ticks=""),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=10, r=10, t=70, b=10), hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=GRID),
    hovermode="x unified",
))
pio.templates.default = "plotly_white+finsent"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea {{
  font-family: 'Plus Jakarta Sans', Inter, sans-serif;
}}
.block-container {{ padding-top: 3.2rem; max-width: 1400px; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #FFFFFF 0%, #F3F2FF 100%);
  border-right: 1px solid {GRID}; }}
.hero {{ background: linear-gradient(120deg, {INDIGO} 0%, {VIOLET} 45%, {SKY} 100%); color: white;
  border-radius: 22px; padding: 26px 30px; margin-bottom: 18px; box-shadow: 0 12px 30px rgba(91,91,247,.22); }}
.hero h1 {{ color: white; font-size: 2.0rem; font-weight: 800; margin: 0 0 4px 0; letter-spacing: -.02em; }}
.hero p {{ color: rgba(255,255,255,.9); margin: 0; font-size: 1.0rem; }}
.hero .tag {{ display: inline-block; background: rgba(255,255,255,.2); border-radius: 999px; padding: 3px 12px;
  font-size: .78rem; font-weight: 600; margin-bottom: 10px; letter-spacing: .04em; }}
[data-testid="stMetric"] {{ background: white; border: 1px solid {GRID}; border-radius: 16px;
  padding: 14px 16px; box-shadow: 0 2px 10px rgba(30,34,53,.04); }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-weight: 600; font-size: .82rem; }}
[data-testid="stMetricValue"] {{ font-weight: 800; font-size: 1.55rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; background: white; padding: 6px; border-radius: 14px;
  border: 1px solid {GRID}; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 10px; padding: 8px 16px; font-weight: 600; }}
.stTabs [aria-selected="true"] {{ background: #EEEEFF; color: {INDIGO}; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
.section {{ font-weight: 800; font-size: 1.1rem; margin: 14px 0 6px 0; color: {INK}; }}
.note {{ color: {MUTED}; font-size: .88rem; }}
.pill {{ display: inline-block; border-radius: 999px; padding: 2px 10px; font-size: .78rem; font-weight: 700; }}
.pill.positive {{ background: #E3F9F1; color: #0B8A63; }}
.pill.negative {{ background: #FDE8EC; color: #C8284A; }}
.pill.neutral {{ background: #FFF4DE; color: #B87503; }}
.headline-card {{ background: white; border: 1px solid {GRID}; border-radius: 14px; padding: 12px 14px;
  margin-bottom: 8px; }}
.headline-card .meta {{ color: {MUTED}; font-size: .8rem; margin-top: 4px; }}
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, tag: str = ""):
    tag_html = f'<div class="tag">{tag}</div>' if tag else ""
    st.markdown(f'<div class="hero">{tag_html}<h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def section(text: str, note: str = ""):
    st.markdown(f'<div class="section">{text}</div>' + (f'<div class="note">{note}</div>' if note else ""),
                unsafe_allow_html=True)


def pill(label: str) -> str:
    return f'<span class="pill {label}">{label}</span>'
