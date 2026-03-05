import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import RAW_DATA_DIR, PREDICTIONS_DIR, DEFAULT_TICKER

st.set_page_config(page_title="Financial Sentiment Analyzer", page_icon="📈", layout="wide")

st.title("📈 Financial News Sentiment & Stock Price Correlation Dashboard")
st.markdown("This dashboard correlates financial news sentiment with actual stock price movements.")

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker", value=DEFAULT_TICKER).upper()

# Try loading data
sentiment_file = os.path.join(PREDICTIONS_DIR, f"{ticker}_yahoo_news_finbert_preds_aggregated.csv")
stock_file = os.path.join(RAW_DATA_DIR, f"{ticker}_stock_data.csv")
merged_file = os.path.join(PREDICTIONS_DIR, f"{ticker}_merged_analysis.csv")
raw_news_file = os.path.join(PREDICTIONS_DIR, f"{ticker}_yahoo_news_finbert_preds.csv")

@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

merged_df = load_data(merged_file)
raw_news_df = load_data(raw_news_file)

if merged_df.empty:
    st.warning(f"No fully processed data found for {ticker}. Please run the main pipeline first (`python main.py --ticker {ticker}`).")
else:
    # Top Level Metrics
    current_price = merged_df['close'].iloc[-1]
    prev_price = merged_df['close'].iloc[-2] if len(merged_df) > 1 else current_price
    price_change = ((current_price - prev_price) / prev_price) * 100
    
    current_sentiment = merged_df['sentiment_score'].iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close Price", f"${current_price:.2f}", f"{price_change:.2f}%")
    col2.metric("Latest Daily Sentiment Score", f"{current_sentiment:.2f}")
    col3.metric("Total Articles Analyzed", f"{merged_df['article_count'].sum()}")
    
    from scipy.stats import pearsonr
    merged_clean = merged_df.dropna(subset=['sentiment_score', 'daily_return'])
    if len(merged_clean) > 1:
        corr, _ = pearsonr(merged_clean['sentiment_score'], merged_clean['daily_return'])
        col4.metric("Sentiment vs Return(%) Correlation", f"{corr:.3f}")
    else:
        col4.metric("Sentiment vs Return(%) Correlation", "N/A")
    
    # Dual Axis Chart
    st.subheader(f"{ticker} Price vs Sentiment Trend")
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged_df['date'], y=merged_df['close'], name="Close Price", yaxis="y1"))
    
    smoothed_sentiment = merged_df['sentiment_score'].rolling(window=3, min_periods=1).mean()
    fig.add_trace(go.Scatter(x=merged_df['date'], y=smoothed_sentiment, name="Sentiment (3d MA)", line=dict(dash='dot'), yaxis="y2"))
    
    fig.update_layout(
        yaxis=dict(title="Close Price", color="blue"),
        yaxis2=dict(title="Sentiment Score", color="red", overlaying="y", side="right"),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Scatter Plot - Sentiment vs Returns
        st.subheader("Sentiment vs Stock Return (%)")
        scatter_fig = px.scatter(merged_clean, x='sentiment_score', y='daily_return', trendline="ols",
                                 labels={'sentiment_score': 'Daily Sentiment Score', 'daily_return': 'Daily Return (%)'})
        st.plotly_chart(scatter_fig, use_container_width=True)
    
    with col_chart2:
        # Sentiment Distribution
        if not raw_news_df.empty:
            st.subheader("Sentiment Distribution")
            dist_fig = px.pie(raw_news_df, names='finbert_sentiment', hole=0.3,
                              color_discrete_map={'positive':'#00CC96', 'neutral':'#B6E880', 'negative':'#EF553B'})
            st.plotly_chart(dist_fig, use_container_width=True)
    
    if not raw_news_df.empty:
        st.subheader("Latest Headlines")
        st.dataframe(raw_news_df[['date', 'headline', 'finbert_sentiment', 'finbert_confidence']].sort_values(by='date', ascending=False).head(20), use_container_width=True)
