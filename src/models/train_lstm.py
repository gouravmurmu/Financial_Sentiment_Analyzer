import os
import pandas as pd
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import sys
from tqdm import tqdm

# Add base dir to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import PROCESSED_DATA_DIR, LSTM_MODEL_DIR
from src.utils.helpers import setup_logger

logger = setup_logger('train_lstm')

# --- Hyperparameters ---
MAX_LEN = 50
MAX_WORDS = 10000
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001

class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden_state = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc(self.dropout(hidden_state))
        return out


def tokenize_and_pad(texts, max_len, max_words):
    words = [word for text in texts for word in str(text).split()]
    word_counts = Counter(words)
    vocab = [word for word, count in word_counts.most_common(max_words - 1)]
    word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}
    word2idx['<PAD>'] = 0
    
    sequences = []
    for text in texts:
        seq = [word2idx.get(word, 0) for word in str(text).split()]
        if len(seq) < max_len:
            seq += [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        sequences.append(seq)
        
    return sequences, word2idx


def train_lstm(input_file):
    logger.info(f"Loading data from {input_file}")
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return
        
    df = pd.read_csv(input_file).dropna(subset=['cleaned_headline', 'sentiment'])
    
    X = df['cleaned_headline'].values
    y = df['sentiment'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    # Tokenize
    logger.info("Tokenizing text...")
    X_seq, word2idx = tokenize_and_pad(X, MAX_LEN, MAX_WORDS)
    vocab_size = len(word2idx)
    
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_encoded, test_size=0.2, random_state=42)
    
    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = LSTMClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    logger.info("Training LSTM model...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        
        for batch_inputs, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(predictions.data, 1)
            train_correct += (predicted == batch_labels).sum().item()
            
        train_acc = train_correct / len(train_dataset)
        logger.info(f"Epoch {epoch+1} - Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}")
        
    model_path = os.path.join(LSTM_MODEL_DIR, 'lstm_model.pth')
    torch.save(model.state_dict(), model_path)
    
    artifacts = {
        'word2idx': word2idx,
        'label_encoder': label_encoder,
        'hyperparameters': {
            'max_len': MAX_LEN,
            'max_words': MAX_WORDS,
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'vocab_size': vocab_size,
            'output_dim': num_classes
        }
    }
    artifacts_path = os.path.join(LSTM_MODEL_DIR, 'artifacts.pkl')
    with open(artifacts_path, 'wb') as f:
        pickle.dump(artifacts, f)
        
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Artifacts saved to {artifacts_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM model')
    parser.add_argument('--input', type=str, required=True, help='Filename of processed data in PROCESSED_DATA_DIR')
    
    args = parser.parse_args()
    input_path = os.path.join(PROCESSED_DATA_DIR, args.input)
    train_lstm(input_path)
