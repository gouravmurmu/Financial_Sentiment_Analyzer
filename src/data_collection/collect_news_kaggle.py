import os
import pandas as pd
import argparse
import sys

# Add base dir to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import RAW_DATA_DIR
from src.utils.helpers import setup_logger

logger = setup_logger('collect_news_kaggle')

def process_kaggle_dataset(input_csv):
    """
    Loads a financial news dataset CSV and saves it in a standardized format:
    date, headline, sentiment
    """
    logger.info(f"Processing Kaggle dataset: {input_csv}")
    
    if not os.path.exists(input_csv):
        logger.error(f"Input file not found: {input_csv}")
        return None
        
    try:
        # Note: Example handles general format without header (e.g. FinancialPhraseBank)
        df = pd.read_csv(input_csv, encoding='latin-1', header=None, names=['sentiment', 'headline'])
        
        output_data = pd.DataFrame()
        output_data['headline'] = df['headline'] if 'headline' in df.columns else df.iloc[:, 1]
        output_data['sentiment'] = df['sentiment'] if 'sentiment' in df.columns else df.iloc[:, 0]
        
        # Determine valid date range size and create dummy date distribution for merging
        date_range = pd.date_range(end='2023-12-31', periods=len(df), freq='D')
        output_data['date'] = date_range.strftime('%Y-%m-%d')
        
        # Filter strictly
        output_data = output_data[['date', 'headline', 'sentiment']]
        
        output_file = os.path.join(RAW_DATA_DIR, "kaggle_financial_news.csv")
        output_data.to_csv(output_file, index=False)
        
        logger.info(f"Successfully processed and saved {len(output_data)} rows to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error processing Kaggle dataset: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Kaggle financial news dataset')
    parser.add_argument('--input', type=str, required=True, help='Path to raw Kaggle CSV file')
    
    args = parser.parse_args()
    process_kaggle_dataset(args.input)
