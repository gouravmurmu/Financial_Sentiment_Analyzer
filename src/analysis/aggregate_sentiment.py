import os
import pandas as pd
import argparse
import sys

# Add base dir to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import PREDICTIONS_DIR, PROCESSED_DATA_DIR
from src.utils.helpers import setup_logger

logger = setup_logger('aggregate_sentiment')

def create_numeric_sentiment(val):
    val_lower = str(val).lower()
    if 'positive' in val_lower or val == 1 or val == '1':
        return 1
    elif 'negative' in val_lower or val == -1 or val == '-1':
        return -1
    else:
        return 0

def aggregate_daily_sentiment(input_file):
    """
    Groups headlines by date and computes a daily average sentiment score.
    Positive = 1, Neutral = 0, Negative = -1
    """
    logger.info(f"Aggregating sentiment for {input_file}")
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return None
        
    df = pd.read_csv(input_file)
    
    sentiment_col = 'finbert_sentiment' if 'finbert_sentiment' in df.columns else 'sentiment'
    
    if sentiment_col not in df.columns or 'date' not in df.columns:
        logger.error(f"Required columns 'date' and '{sentiment_col}' missing.")
        return None
        
    df['numeric_sentiment'] = df[sentiment_col].apply(create_numeric_sentiment)
    
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    daily_sentiment = df.groupby('date').agg(
        sentiment_score=('numeric_sentiment', 'mean'),
        article_count=('headline', 'count')
    ).reset_index()
    
    input_basename = os.path.basename(input_file)
    output_filename = input_basename.replace('.csv', '_aggregated.csv')
    output_path = os.path.join(PREDICTIONS_DIR, output_filename)
    
    daily_sentiment.to_csv(output_path, index=False)
    logger.info(f"Saved aggregated sentiment to {output_path}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate sentiment to daily scores')
    parser.add_argument('--input', type=str, required=True, help='Filename of predictions or dataset')
    
    args = parser.parse_args()
    
    input_path = os.path.join(PREDICTIONS_DIR, args.input)
    if not os.path.exists(input_path):
        input_path = os.path.join(PROCESSED_DATA_DIR, args.input)
        
    aggregate_daily_sentiment(input_path)
