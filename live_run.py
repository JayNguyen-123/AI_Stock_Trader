import asyncio
import uuid
import numpy as np

from config.settings import ASSET_UNIVERSE, STATE_DIMENSION, ACTION_DIMENSION
from src.agent import ProductionDDQNAgent
from src.tracker import LiveAssetFeatureTracker
from src.allocator import MasterAllocationCritic
from utils.async_api_bridge import AsyncBrokerTradingBridge
from utils.db_logger import ThreadSafeDatabaseLogger

# Instantiate modular components
tracker = LiveAssetFeatureTracker()
db_logger = ThreadSafeDatabaseLogger()
api_bridge = AsyncBrokerTradingBridge()
allocator = MasterAllocationCritic(ASSET_UNIVERSE)

# Initialize dedicated specialist agent neural networks for each asset in the portfolio
specialist_networks = {symbol: ProductionDDQNAgent(STATE_DIMENSION, ACTION_DIMENSION) for symbol in ASSET_UNIVERSE}

# Live tracking cache positions
portfolio_cash = 10000.0
holdings = {symbol: 0 for symbol in ASSET_UNIVERSE}
current_prices = {symbol: 100.0 for symbol in ASSET_UNIVERSE} # Fallback seed price metrics

async def evaluate_portfolio_network_step():
    """Master network evaluation step. Runs agent lookups and executes approved actions asynchronously."""
    global portfolio_cash
    
    # Calculate current portfolio valuation profiles
    holdings_equity = sum(holdings[s] * current_prices[s] for s in ASSET_UNIVERSE)
    net_worth = portfolio_cash + holdings_equity
    
    raw_proposals = {}
    
    # 1. Gather trade proposals from individual asset networks
    for symbol in ASSET_UNIVERSE:
        state_features = tracker.extract_state_features(symbol, portfolio_cash, holdings[symbol], net_worth)
        if state_features is not None:
            # Query the asset's specific specialist neural network
            raw_proposals[symbol] = specialist_networks[symbol].select_action(state_features)
        else:
            raw_proposals[symbol] = 0 # Default to Hold if indicator buffers are underfilled
            
    # 2. Run proposals through the Master Allocation Critic to enforce portfolio risk checks
    filtered_decisions = allocator.evaluate_allocation_requests(
        raw_proposals, portfolio_cash, current_prices, holdings, net_worth
    )
    
    # 3. Deploy approved trades concurrently using a non-blocking asynchronous task loop
    async_tasks = []
    for symbol, action in filtered_decisions.items():
        if action == 1:   # Approved BUY Order Execution
            async_tasks.append(process_execution_flow(symbol, "BUY", current_prices[symbol]))
        elif action == 2: # Approved SELL Order Execution
            async_tasks.append(process_execution_flow(symbol, "SELL", current_prices[symbol]))
            
    if async_tasks:
        await asyncio.gather(*async_tasks)


async def process_execution_flow(symbol, side, execution_price):
    """Handles async order routing, balances position metrics, and logs transaction updates"""
    global portfolio_cash
    
    # Dispatch non-blocking order request to the broker
    receipt = await api_bridge.dispatch_market_order(symbol, side, quantity=1)
    
    if receipt:
        # Update accounting metrics on verified executions
        if side == "BUY":
            holdings[symbol] += 1
            portfolio_cash -= execution_price
        elif side == "SELL":
            holdings[symbol] -= 1
            portfolio_cash += execution_price
            
        # Log transaction to disk storage layers asynchronously
        db_logger.log_order(str(uuid.uuid4())[:8], symbol, side, 1, execution_price)


def websocket_data_ingestion_callback(symbol, price, volume):
    """Callback function triggered on incoming live streaming data packets"""
    current_prices[symbol] = price
    tracker.append_tick(symbol, price, volume)
    
    # Schedule a portfolio review on the event loop as data packets arrive
    asyncio.create_task(evaluate_portfolio_network_step())


async def master_production_loop():
    """Main lifecycle thread driver that connects async tasks and maintains system uptime"""
    await api_bridge.initialize_session()
    print("[SYSTEM BOOT] Multi-agent allocation channels operational. Ingesting network packets...")
    
    try:
        # Mocking active engine loop mechanics for testing environments
        while True:
            # Simulate streaming network ticks coming in across the asset universe
            for symbol in ASSET_UNIVERSE:
                mock_price_tick = current_prices[symbol] + np.random.normal(0, 0.2)
                websocket_data_ingestion_callback(symbol, max(1.0, mock_price_tick), volume=500)
                
            await asyncio.sleep(1.0) # Check system intervals every 1 second
    except asyncio.CancelledError:
        pass
    finally:
        await api_bridge.close_session()
        print("[SHUTDOWN] Network sessions disconnected cleanly.")


if __name__ == "__main__":
    # Launch your asynchronous multi-agent deployment runtime
    try:
        asyncio.run(master_production_loop())
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Execution interrupted by terminal prompt request.")
