import re
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Preprocesses text by:
    - lowercasing
    - removing punctuation and special characters
    - removing stopwords
    - removing extra whitespace
    """
    if not isinstance(text, str):
        return ""
        
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and special characters (keep only alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Remove stopwords and extra whitespace
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    
    # Join and return
    return ' '.join(cleaned_words)
