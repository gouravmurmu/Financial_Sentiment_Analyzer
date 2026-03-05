import os
import argparse
import sys

from src.utils.config import DEFAULT_TICKER, START_DATE, END_DATE
from src.data_collection.collect_stock_data import fetch_stock_data
from src.data_collection.collect_news_yahoo import scrape_yahoo_news
from src.models.finbert_inference import run_finbert_inference
from src.analysis.aggregate_sentiment import aggregate_daily_sentiment
from src.analysis.correlation_analysis import run_correlation_analysis
from src.utils.helpers import setup_logger

logger = setup_logger("main_pipeline")

def run_pipeline(ticker=DEFAULT_TICKER):
    logger.info("="*50)
    logger.info(f"STARTING PIPELINE FOR {ticker}")
    logger.info("="*50)
    
    logger.info("Step 1: Data Collection")
    stock_file = fetch_stock_data(ticker, START_DATE, END_DATE)
    news_file = scrape_yahoo_news(ticker)
    
    if not stock_file or not news_file:
        logger.error("Data collection failed. Exiting.")
        return
        
    logger.info("Step 2: FinBERT Inference")
    preds_file = run_finbert_inference(news_file)
    if not preds_file:
        logger.error("Inference failed. Exiting.")
        return
        
    logger.info("Step 3: Sentiment Aggregation")
    agg_file = aggregate_daily_sentiment(preds_file)
    if not agg_file:
        logger.error("Aggregation failed. Exiting.")
        return
        
    logger.info("Step 4: Correlation Analysis")
    final_analysis = run_correlation_analysis(stock_file, agg_file)
    
    logger.info("="*50)
    logger.info(f"PIPELINE COMPLETE. Results saved to {final_analysis}")
    logger.info(f"Run `streamlit run dashboard/app.py` to view.")
    logger.info("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default=DEFAULT_TICKER)
    args = parser.parse_args()
    
    run_pipeline(args.ticker)
