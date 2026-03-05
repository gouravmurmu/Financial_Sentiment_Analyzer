import os
import yfinance as yf
import pandas as pd
import argparse
import sys

# Add base dir to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import RAW_DATA_DIR, DEFAULT_TICKER, START_DATE, END_DATE
from src.utils.helpers import setup_logger

logger = setup_logger('collect_stock_data')

def fetch_stock_data(ticker=DEFAULT_TICKER, start_date=START_DATE, end_date=END_DATE):
    """
    Fetches historical stock data using yfinance and saves it to a CSV file.
    """
    logger.info(f"Fetching stock data for {ticker} from {start_date} to {end_date}")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            logger.warning(f"No data found for {ticker}")
            return None
            
        # If columns are MultiIndex, flatten them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
            
        # Reset index to make Date a column
        data.reset_index(inplace=True)
        # Standardize column names
        data.columns = [col.lower() for col in data.columns]
        
        # Save to CSV
        output_file = os.path.join(RAW_DATA_DIR, f"{ticker}_stock_data.csv")
        data.to_csv(output_file, index=False)
        logger.info(f"Successfully saved {len(data)} rows to {output_file}")
        
        return output_file
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect stock data using yfinance')
    parser.add_argument('--ticker', type=str, default=DEFAULT_TICKER, help='Stock ticker symbol')
    parser.add_argument('--start', type=str, default=START_DATE, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=END_DATE, help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    fetch_stock_data(args.ticker, args.start, args.end)
