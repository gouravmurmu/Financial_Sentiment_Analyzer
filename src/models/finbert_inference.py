import os
import pandas as pd
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
from tqdm import tqdm

# Add base dir to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, PREDICTIONS_DIR
from src.utils.helpers import setup_logger

logger = setup_logger('finbert_inference')

def run_finbert_inference(input_file):
    logger.info(f"Running FinBERT inference on {input_file}")
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return None
        
    df = pd.read_csv(input_file)
    
    if 'headline' not in df.columns:
        logger.error("No 'headline' column in dataset.")
        return None
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Loading ProsusAI/finbert model on {device}...")
    
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    
    sentiments = []
    confidences = []
    
    logger.info("Predicting sentiments...")
    batch_size = 32
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df['headline'].iloc[i:i+batch_size].astype(str).tolist()
        
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            top_conf, top_class = torch.max(predictions, dim=-1)
            
            # FinBert labels: 0: positive, 1: negative, 2: neutral (ProsusAI config based)
            pred_labels = [model.config.id2label[idx.item()] for idx in top_class]
            
            sentiments.extend(pred_labels)
            confidences.extend(top_conf.cpu().numpy().tolist())
            
    df['finbert_sentiment'] = sentiments
    df['finbert_confidence'] = confidences
    
    # Save predictions
    input_basename = os.path.basename(input_file)
    output_filename = input_basename.replace('.csv', '_finbert_preds.csv')
    output_path = os.path.join(PREDICTIONS_DIR, output_filename)
    
    # Ensure date is retained if it exists
    cols_to_save = ['headline', 'finbert_sentiment', 'finbert_confidence']
    if 'date' in df.columns:
        cols_to_save.insert(0, 'date')
        
    df[cols_to_save].to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FinBERT inference')
    parser.add_argument('--input', type=str, required=True, help='Filename of data')
    
    args = parser.parse_args()
    
    input_path = os.path.join(RAW_DATA_DIR, args.input)
    if not os.path.exists(input_path):
        input_path = os.path.join(PROCESSED_DATA_DIR, args.input)
        
    run_finbert_inference(input_path)
