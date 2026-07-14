"""
Central configuration for the options trading bot.
All constants and environment variable loading in one place.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "state"
POSITIONS_FILE = STATE_DIR / "positions.json"
DAILY_LOG_FILE = STATE_DIR / "daily_log.json"

# ── Alpaca API ─────────────────────────────────────────────────────────────────
ALPACA_API_KEY: str = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY: str = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_PAPER: bool = True
ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL: str = "https://data.alpaca.markets"

# ── Modal (inference + Phase 2 training) ──────────────────────────────────────
MODAL_TOKEN_ID: str = os.environ.get("MODAL_TOKEN_ID", "")
MODAL_TOKEN_SECRET: str = os.environ.get("MODAL_TOKEN_SECRET", "")
MODAL_GPU_TYPE: str = "A10G"
MODAL_INFERENCE_TIMEOUT: int = 120
MODAL_TRAINING_GPU_PRICE: float = 1.10  # A10G at $1.10/hr

# ── Lightning.ai (Phase 1 training) ───────────────────────────────────────────
LIGHTNING_GPU: str = "L4"
LIGHTNING_GPU_PRICE_HR: float = 0.48   # L4 at $0.48/hr
LIGHTNING_BUDGET: float = 15.0         # Phase 1 budget

# ── Training Budget ────────────────────────────────────────────────────────────
TRAINING_TOTAL_BUDGET: float = 45.0    # $15 Lightning + $30 Modal
TRAINING_DATA_EXAMPLES: int = 160_000
TRAINING_EPOCHS: int = 8

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
HF_MODEL_REPO: str = "Rohan556/options-llm-lora"
HF_CHECKPOINT_REPO: str = "Rohan556/options-llm-checkpoints"
BASE_MODEL_NAME: str = "meta-llama/Meta-Llama-3-8B-Instruct"

# ── Polygon.io ────────────────────────────────────────────────────────────────
POLYGON_API_KEY: str = os.environ.get("POLYGON_API_KEY", "")

# ── Email ──────────────────────────────────────────────────────────────────────
EMAIL_USER: str = os.environ.get("EMAIL_USER", "")
EMAIL_PASS: str = os.environ.get("EMAIL_PASS", "")
EMAIL_RECIPIENT: str = os.environ.get("EMAIL_RECIPIENT", "")
EMAIL_SMTP_HOST: str = "smtp.gmail.com"
EMAIL_SMTP_PORT: int = 587

# ── LLM Parameters ────────────────────────────────────────────────────────────
MIN_CONFIDENCE: float = 0.70
LLM_MAX_NEW_TOKENS: int = 1024
LLM_TEMPERATURE: float = 0.3

# ── Risk Parameters ────────────────────────────────────────────────────────────
MAX_POSITION_PCT: float = 0.20          # No single position > 20% of equity
MAX_OPEN_POSITIONS: int = 5             # Max concurrent open positions
DAILY_LOSS_LIMIT_PCT: float = -0.15     # Halt if daily P&L < -15%
MAX_BID_ASK_SPREAD_PCT: float = 0.10    # Reject contracts with spread > 10%
MIN_OPEN_INTEREST: int = 100            # Reject contracts with OI < 100
HARD_EXIT_LOSS_PCT: float = 1.0         # Close if loss > 100% of debit paid
DTE_EXIT_THRESHOLD: int = 1             # Close any position at 1 DTE

# ── Trading Parameters ─────────────────────────────────────────────────────────
MIN_DTE: int = 7
MAX_DTE: int = 45
PROFIT_TARGET_PCT: float = 0.50         # Close if profit > 50% of debit

# ── Watchlist ──────────────────────────────────────────────────────────────────
WATCHLIST: list[str] = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "AMD", "NFLX", "JPM", "V", "UNH", "JNJ", "PG", "MA", "HD", "BAC",
]

# ── Supported Strategies ───────────────────────────────────────────────────────
STRATEGIES = [
    "long_call",
    "long_put",
    "bull_call_spread",
    "bear_put_spread",
    "iron_condor",
    "straddle",
    "strangle",
    "calendar_spread",
]

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
