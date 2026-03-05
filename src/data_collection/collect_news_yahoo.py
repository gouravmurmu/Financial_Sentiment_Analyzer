import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import argparse
import sys

# Add base dir to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import RAW_DATA_DIR, DEFAULT_TICKER
from src.utils.helpers import setup_logger

logger = setup_logger('collect_news_yahoo')

def scrape_yahoo_news(ticker=DEFAULT_TICKER):
    """
    Scrapes Yahoo Finance news headlines for a given ticker.
    Saves to standardized format: date, headline
    """
    logger.info(f"Scraping Yahoo Finance news for {ticker}")
    
    try:
        import yfinance as yf
        ticker_obj = yf.Ticker(ticker)
        news_items = ticker_obj.news
        
        headlines = []
        for item in news_items:
            title = item.get('content', {}).get('title')
            if title and len(title) > 20:
                headlines.append(title)
                
        if not headlines:
            logger.warning(f"No headlines found for {ticker}.")
            return None
            
        df = pd.DataFrame({
            'date': [datetime.now().strftime('%Y-%m-%d')] * len(headlines),
            'headline': headlines
        })
        
        output_file = os.path.join(RAW_DATA_DIR, f"{ticker}_yahoo_news.csv")
        
        if os.path.exists(output_file):
            df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            df.to_csv(output_file, index=False)
            
        logger.info(f"Successfully scraped and saved {len(df)} headlines to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error scraping Yahoo Finance: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape Yahoo Finance news')
    parser.add_argument('--ticker', type=str, default=DEFAULT_TICKER, help='Stock ticker symbol')
    
    args = parser.parse_args()
    scrape_yahoo_news(args.ticker)
