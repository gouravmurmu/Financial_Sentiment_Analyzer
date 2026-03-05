# Financial News Sentiment Analyzer with Stock Price Correlation Dashboard

## Project Overview
This project is an end-to-end Machine Learning and Data Science pipeline that collects financial news headlines and stock price data, performs sentiment analysis using multiple paradigms (Logistic Regression, LSTM, FinBERT), and visualizes the correlation between news sentiment and stock price movements in an interactive Streamlit dashboard.

## Tech Stack
- **Language**: Python 3.10+
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `streamlit`, `plotly`
- **Classic ML**: `scikit-learn`, `nltk`
- **Deep Learning**: `torch`, `transformers` (Hugging Face)
- **Data Collection**: `yfinance`, `requests`, `beautifulsoup4`

## Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the NLTK stopwords (handled automatically in preprocessing).

## How to Run
1. Run the entire pipeline to collect data, train models, and compute correlations:
   ```bash
   python main.py
   ```
2. Launch the interactive dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```

## Resume Description
- **Financial News Sentiment Analyzer**: Designed and implemented an end-to-end data science pipeline to ingest financial news and stock data, utilizing Logistic Regression, custom LSTM networks, and Hugging Face FinBERT for NLP sentiment classification.
- **Correlative Data Analysis**: Engineered a data aggregation module to combine daily sentiment metrics with `yfinance` market data, performing structural correlation analysis.
- **Interactive Dashboarding**: Developed a production-ready Streamlit frontend featuring Plotly visualizations to interactively showcase market trends, sentiment breakdowns, and model comparisons.
