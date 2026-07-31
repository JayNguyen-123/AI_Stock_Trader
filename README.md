# 📈 Enterprise-Grade Multi-Asset Reinforcement Learning Trading Engine

An advanced, high-frequency, multi-asset algorithmic trading system utilizing **Deep Q-Networks (DQN)** built on PyTorch. This architecture decouples alpha generation (asset-specific specialist agents) from capital preservation using a hierarchical network model, an asynchronous non-blocking connection layer, and real-time microstructure simulation.

---

## 🏗️ System Architecture & Framework Matrix

The codebase transitions from basic grid-world configurations to an enterprise setup by incorporating state-of-the-art value-based reinforcement learning enhancements:

*   **Double DQN (DDQN)**: Decouples action selection from valuation to minimize optimization value overestimations.
*   **Dueling Network Topology**: Splits neural networks into distinct state-value V(s) and action-advantage A(s, a) channels for stable training updates.
*   **Noisy Linear Layers**: Integrates factorized Gaussian parameter noise directly into the neural network layers, making traditional, erratic ε-greedy exploration obsolete.
*   **Prioritized Experience Replay (PER)**: Uses a proportional binary `SumTree` array data structure to prioritize sampling high-loss transitions (\(O(\log N)\) updates).
*   **Hierarchical Allocation Critic**: Intercepts buy and sell suggestions from independent asset agents to enforce global portfolio diversification caps and cash reserve thresholds before placing orders.
*   **Asynchronous REST & WebSocket Engine**: Combines non-blocking `websockets` data ingestion channels with concurrent `aiohttp` execution threads to minimize latency.

---

## 📂 Project Directory Structure

```text
ai-stock-trader/
│
├── config/
│   └── settings.py          # Allocation limits, asset lists, and network endpoints
│
├── src/
│   ├── __init__.py          # Package exposure hooks for single-line imports
│   ├── agent.py             # ProductionDDQNAgent class (one per asset specialist)
│   ├── allocator.py         # MasterAllocationCritic logic (hierarchical budget filter)
│   ├── buffer.py            # SumTree and PrioritizedReplayBuffer structural layers
│   ├── environment.py       # RealisticTradingEnv simulation container (for background testing)
│   ├── models.py            # DuelingNoisyDQN and NoisyLinear neural topologies
│   ├── risk_manager.py      # RiskManagementOverlay (independent defensive supervisor)
│   └── tracker.py           # LiveAssetFeatureTracker (streaming tech indicator compiler)
│
├── utils/
│   ├── __init__.py
│   ├── async_api_bridge.py  # AsyncBrokerTradingBridge execution engine via aiohttp
│   ├── checkpoints.py       # ModelCheckpointManager weight storage pipelines
│   ├── db_logger.py         # Thread-safe SQLite logger tracking live telemetry
│   └── websocket_client.py  # AsyncMarketStreamClient event-driven listener
│
├── checkpoints/
│   ├── best_trading_model.pt # Automatically generated high-score model checkpoint
│   └── checkpoint_latest.pt # Automatically generated training recovery file
│
├── live_run.py              # Asynchronous event-driven master trading orchestrator
├── main.py                  # Standard trainer script tracking live plotting curves
├── backtest.py              # Deterministic out-of-sample institutional test suite
└── requirements.txt         # Package dependency resolution definitions
```

---

## 🛠️ Data Flow Pipeline Mappings

The engine coordinates asynchronous live tracking and execution following this operational topology:

```text
[ Live Exchange WebSocket Stream ]
               │
               ▼ (Asynchronous Ticks)
   [ LiveAssetFeatureTracker ] ──► (Calculates rolling RSI, MACD, SMA metrics)
               │
               ▼ (10-Dimensional Vector State Map)
   [ ProductionDDQNAgent (Basket) ] ──► (Generates raw buy/sell advantage values)
               │
               ▼ (Action Proposals)
   [ MasterAllocationCritic ] ──► (Enforces portfolio concentration and cash limits)
               │
               ▼ (Risk-Approved Orders)
 [ AsyncBrokerTradingBridge ] ──► [ Dispatches Parallel API Orders via aiohttp ]
               │
               └─► [ Thread-Safe SQLite Database Ledger Log Outputs ]
```

---

## 🚀 Execution & Operations Guide

Follow these sequential workflow commands to install dependencies, train models, backtest strategies, and deploy the system:

### 1. Environment Provisioning & Installation
Ensure you have Python 3.10+ installed. Spin up a virtual environment and load the project dependency profile:
```bash
# Clone the repository and navigate to the project directory
cd ai-stock-trader

# Create and activate your virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install the technical package matrix
pip install -r requirements.txt
```

### 2. Local Training Loop Execution
Run the standalone model trainer to start parameter optimization. This process displays an active `matplotlib` dashboard that plots real-time portfolio net worth metrics alongside network loss values without lagging the main training threads:
```bash
python main.py
```
*Note: If your training loop is disconnected or interrupted, running the command again will automatically pick up from the last saved state using `checkpoints/checkpoint_latest.pt`.*

### 3. Out-of-Sample Performance Backtesting
Evaluate your saved model weights against completely unseen historical data. This script puts the neural networks into strict `.eval()` mode, scales parameter exploration noise to zero, and calculates key institutional risk metrics:
```bash
python backtest.py
```
This backtest generates a performance breakdown reporting **Annualized Sharpe Ratios**, **Maximum System Drawdowns**, and trade allocation matrices, along with an execution chart mapping buy and sell triggers onto the underlying asset curve.

### 4. Production Asynchronous Deployment
To run the fully asynchronous, event-driven multi-asset system connected to streaming WebSocket market channels and live execution layers, run the production orchestrator:
```bash
python live_run.py
```

To view or audit trade executions and real-time telemetry metrics saved by the database logging layer, open a parallel terminal shell and query the SQLite database:
```bash
sqlite3 production_trading_vault.db "SELECT * FROM order_ledger ORDER BY timestamp DESC LIMIT 10;"
```

---

## 🔒 Production Risk & Safety Controls

To ensure capital preservation during live deployments, the system enforces three independent guardrails:
1. **Asset Exposure Caps**: The `MasterAllocationCritic` blocks any buy recommendation that pushes a single asset's exposure past **30%** of total portfolio equity.
2. **Cash Reserve Minimums**: The trading engine maintains a mandatory **10% cash cushion**, dropping any incoming trade requests if it risks falling below this liquidity floor.
3. **Defensive Loss Dampeners**: The `RiskManagementOverlay` tracks consecutive losses. If the system hits an unexpected market regime and records 4 consecutive losing trades, it activates a circuit breaker that vetoes aggressive buys until the system stabilizes or is manually reset.
