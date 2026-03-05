import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGISTIC_MODEL_DIR = os.path.join(MODELS_DIR, 'logistic_regression')
LSTM_MODEL_DIR = os.path.join(MODELS_DIR, 'lstm')
FINBERT_MODEL_DIR = os.path.join(MODELS_DIR, 'finbert')

OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, 'predictions')

# Default configurations
DEFAULT_TICKER = 'AAPL'
START_DATE = '2020-01-01'
END_DATE = '2023-12-31'

# Create directories if they don't exist
DIRS_TO_CREATE = [
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    LOGISTIC_MODEL_DIR, LSTM_MODEL_DIR, FINBERT_MODEL_DIR,
    FIGURES_DIR, PREDICTIONS_DIR
]

for d in DIRS_TO_CREATE:
    os.makedirs(d, exist_ok=True)
