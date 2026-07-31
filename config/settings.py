import os
from pathlib import Path

# ==============================================================================
# 1. FILE SYSTEM & STORAGE PATHS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# Target filenames for saving/loading weights
LATEST_CHECKPOINT_FILE = "checkpoint_latest.pt"
BEST_MODEL_FILE = "best_trading_model.pt"


# ==============================================================================
# 2. CORE RL AGENT HYPERPARAMETERS
# ==============================================================================
# Network Dimensions
STATE_DIMENSION = 10
ACTION_DIMENSION = 3   # 0: Hold, 1: Buy, 2: Sell

# Optimization Settings
LEARNING_RATE = 5e-4   # Lower learning rate used for stable Noisy Layer updates
GAMMA = 0.99           # Discount factor for future rewards
BATCH_SIZE = 64        # Experience batch size for training steps

# Prioritized Experience Replay (PER) Controls
REPLAY_BUFFER_CAPACITY = 50_000
PER_ALPHA = 0.6        # Prioritization aggressiveness (0 = uniform, 1 = full)
PER_BETA_START = 0.4   # Importance Sampling weight correction factor
PER_BETA_GROWTH = 1e-3 # Incremental step growth rate for Beta up to 1.0


# ==============================================================================
# 3. MARKET MICROSTRUCTURE SIMULATION CONFIG
# ==============================================================================
INITIAL_PORTFOLIO_BALANCE = 10_000.0

# Execution frictions to mimic real-world trading
TRANSACTION_FEE_RATE = 0.0015   # 0.15% broker/exchange fee per transaction
MARKET_SLIPPAGE_RATE = 0.0005   # 0.05% fill-price penalty due to execution delays

# Safety Limits
CRITICAL_DRAWDOWN_LIMIT = 0.40 # Hard stop-out if portfolio value falls below 40%


# ==============================================================================
# 4. TRAINING & VALIDATION SEPARATIONS
# ==============================================================================
TOTAL_TRAINING_EPISODES = 100
TARGET_NETWORK_UPDATE_FREQ = 5 # Synchronize Target Net every N episodes
DATA_TRAIN_TEST_SPLIT = 0.80   # Use the first 80% of data for training


# ==============================================================================
# 5. BROKER LIVE API GATEWAY CONNECTIONS
# ==============================================================================
# Use local environment variables for production security (e.g., 'export APCA_API_KEY_ID=...')
BROKER_API_KEY = os.getenv("APCA_API_KEY_ID", "MOCK_DEVELOPMENT_KEY_ID")
BROKER_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "MOCK_DEVELOPMENT_SECRET_KEY")

# Default connection routing to the mock Alpaca sandbox paper endpoint
BROKER_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://alpaca.markets")

# ==============================================================================
# MULTI-ASSET & LIVE PRODUCTION PIPELINES
# ==============================================================================
ASSET_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
DB_PATH = "production_trading_vault.db"

# WebSocket Streams (Alpaca Paper Data Source Example)
WS_STREAM_URL = "wss://stream.data.alpaca.markets/v2/iex"

# ==============================================================================
# HIERARCHICAL ALLOCATION & ASYNCHRONOUS NETWORK PARAMETERS
# ==============================================================================
ASSET_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
STATE_DIMENSION = 10
ACTION_DIMENSION = 3

# Capital Limits
MAX_CAPITAL_PER_ASSET_PCT = 0.30  # No single stock can consume more than 30% of portfolio equity
MIN_RESERVE_CASH_PCT = 0.10       # Always maintain at least 10% cash for risk safety

# API Network Endpoints (Alpaca Sandbox Example)
BROKER_BASE_URL = "https://alpaca.markets"


