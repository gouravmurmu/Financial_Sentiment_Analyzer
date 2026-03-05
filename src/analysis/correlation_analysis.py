import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
from scipy.stats import pearsonr

# Add base dir to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import RAW_DATA_DIR, PREDICTIONS_DIR, FIGURES_DIR
from src.utils.helpers import setup_logger

logger = setup_logger('correlation_analysis')

def run_correlation_analysis(stock_file, sentiment_file):
    """
    Merges stock prices with aggregated sentiment and computes Pearson correlation.
    Generates and saves correlation plots.
    """
    logger.info("Starting correlation analysis...")
    
    if not os.path.exists(stock_file) or not os.path.exists(sentiment_file):
        logger.error("Required input files not found.")
        return None
        
    stock_df = pd.read_csv(stock_file)
    sentiment_df = pd.read_csv(sentiment_file)
    
    stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
    
    merged = pd.merge(stock_df, sentiment_df, on='date', how='inner')
    
    if merged.empty:
        logger.warning("No overlapping dates found between stock data and sentiment data.")
        return None
    
    # Calculate daily returns
    merged = merged.sort_values(by='date')
    if 'close' in merged.columns:
        close_col = 'close'
    else:
        # Fallback if close is not found but adj close or something is
        close_cols = [c for c in merged.columns if 'close' in c.lower()]
        close_col = close_cols[0] if close_cols else merged.columns[1]
        
    merged['daily_return'] = merged[close_col].pct_change()
    merged.dropna(subset=['sentiment_score', 'daily_return'], inplace=True)
    
    if len(merged) < 2:
         logger.warning("Not enough data points after merge & dropna.")
         return None
         
    # Compute correlation
    corr, p_value = pearsonr(merged['sentiment_score'], merged['daily_return'])
    logger.info(f"Pearson Correlation (Sentiment vs Daily Return): {corr:.4f} (p-value: {p_value:.4f})")
    
    corr_close, _ = pearsonr(merged['sentiment_score'], merged[close_col])
    logger.info(f"Pearson Correlation (Sentiment vs Close Price): {corr_close:.4f}")
    
    # Visualizations
    sns.set_theme(style="whitegrid")
    
    ticker = os.path.basename(stock_file).split('_')[0]
    if os.path.basename(stock_file) == "stock_file":
        ticker = "Stock"
        
    # Plot 1: Trend
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{close_col} Price', color='tab:blue')
    ax1.plot(merged['date'], merged[close_col], color='tab:blue', label='Price')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Sentiment Score', color='tab:red')
    smoothed_sentiment = merged['sentiment_score'].rolling(window=3, min_periods=1).mean()
    ax2.plot(merged['date'], smoothed_sentiment, color='tab:red', linestyle='--', label='Sentiment (3-d MA)')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    fig.tight_layout()
    plt.title(f"{ticker} Price and Sentiment Trend")
    trend_path = os.path.join(FIGURES_DIR, f"{ticker}_price_sentiment_trend.png")
    plt.savefig(trend_path)
    plt.close()
    
    # Plot 2: Scatter
    plt.figure(figsize=(8, 6))
    sns.regplot(x='sentiment_score', y='daily_return', data=merged, scatter_kws={'alpha':0.5})
    plt.title(f"Sentiment Score vs {ticker} Daily Return (r={corr:.2f})")
    plt.xlabel("Daily Sentiment Score")
    plt.ylabel("Daily Return (%)")
    scatter_path = os.path.join(FIGURES_DIR, f"{ticker}_sentiment_return_scatter.png")
    plt.savefig(scatter_path)
    plt.close()
    
    logger.info(f"Saved plots to {FIGURES_DIR}")
    
    # Save merged data
    final_output = os.path.join(PREDICTIONS_DIR, f"{ticker}_merged_analysis.csv")
    merged.to_csv(final_output, index=False)
    
    return final_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Correlation Analysis')
    parser.add_argument('--stock_file', type=str, required=True, help='Filename of stock data in RAW_DATA_DIR')
    parser.add_argument('--sentiment_file', type=str, required=True, help='Filename of aggregated sentiment in PREDICTIONS_DIR')
    
    args = parser.parse_args()
    
    s_path = os.path.join(RAW_DATA_DIR, args.stock_file)
    sent_path = os.path.join(PREDICTIONS_DIR, args.sentiment_file)
    
    run_correlation_analysis(s_path, sent_path)
