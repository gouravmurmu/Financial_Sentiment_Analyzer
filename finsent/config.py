import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FINSENT_DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.environ.get("FINSENT_MODELS_DIR", ROOT / "models"))

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "JPM"]
BENCHMARK = "SPY"
FINBERT_MODEL = os.environ.get("FINSENT_FINBERT_MODEL", "ProsusAI/finbert")

LABELS = ("negative", "neutral", "positive")

# Headlines published after the US close move the next session's price.
MARKET_CLOSE_HOUR = 16
MARKET_TZ = "America/New_York"
