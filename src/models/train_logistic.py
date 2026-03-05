import os
import pandas as pd
import argparse
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import sys

# Add base dir to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import PROCESSED_DATA_DIR, LOGISTIC_MODEL_DIR
from src.utils.helpers import setup_logger

logger = setup_logger('train_logistic')

def train_logistic_regression(input_file):
    """
    Trains a Logistic Regression model using TF-IDF features.
    Saves the model and vectorizer.
    """
    logger.info(f"Loading data from {input_file}")
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return
        
    df = pd.read_csv(input_file)
    
    # Cleaning up nulls
    df.dropna(subset=['cleaned_headline', 'sentiment'], inplace=True)
    
    X = df['cleaned_headline'].values
    y = df['sentiment'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, multi_class='auto')
    model.fit(X_train_vec, y_train)
    
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    logger.info(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))
    
    # Save model and vectorizer
    model_path = os.path.join(LOGISTIC_MODEL_DIR, 'model.pkl')
    vec_path = os.path.join(LOGISTIC_MODEL_DIR, 'vectorizer.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Vectorizer saved to {vec_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Logistic Regression model')
    parser.add_argument('--input', type=str, required=True, help='Filename of processed data in PROCESSED_DATA_DIR')
    
    args = parser.parse_args()
    input_path = os.path.join(PROCESSED_DATA_DIR, args.input)
    train_logistic_regression(input_path)
