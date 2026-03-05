import os
import pandas as pd
import argparse
import sys

# Add base dir to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.helpers import setup_logger
from src.preprocessing.clean_text import clean_text

logger = setup_logger('build_dataset')

def preprocess_dataset(input_file, output_file):
    """
    Applies preprocessing to the dataset and saves it to the processed directory.
    """
    logger.info(f"Preprocessing {input_file}")
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return False
        
    try:
        df = pd.read_csv(input_file)
        
        if 'headline' not in df.columns:
            logger.error(f"'headline' column not found in {input_file}")
            return False
            
        # Apply text cleaning
        logger.info("Cleaning text...")
        df['cleaned_headline'] = df['headline'].apply(clean_text)
        
        # Drop empty rows after cleaning
        df = df[df['cleaned_headline'].str.strip() != '']
        
        # Save processed dataset
        df.to_csv(output_file, index=False)
        logger.info(f"Successfully saved {len(df)} processed rows to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess raw datasets')
    parser.add_argument('--input', type=str, required=True, help='Filename of raw data in RAW_DATA_DIR')
    parser.add_argument('--output', type=str, required=True, help='Filename for processed data in PROCESSED_DATA_DIR')
    
    args = parser.parse_args()
    
    input_path = os.path.join(RAW_DATA_DIR, args.input)
    output_path = os.path.join(PROCESSED_DATA_DIR, args.output)
    
    preprocess_dataset(input_path, output_path)
