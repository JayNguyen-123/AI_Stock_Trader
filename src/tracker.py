import collections
import numpy as np
import ta
import pandas as pd

class LiveAssetFeatureTracker:
    """Compiles streaming stock ticks into raw multi-dimensional agent observations"""
    def __init__(self, buffer_size=100):
        self.price_history = collections.defaultdict(list)
        self.volume_history = collections.defaultdict(list)
        self.buffer_size = buffer_size

    def append_tick(self, symbol, price, volume):
        self.price_history[symbol].append(price)
        self.volume_history[symbol].append(volume)
        
        # Enforce rolling window memory bounds
        if len(self.price_history[symbol]) > self.buffer_size:
            self.price_history[symbol].pop(0)
            self.volume_history[symbol].pop(0)

    def extract_state_features(self, symbol, portfolio_balance, shares_held, total_net_worth):
        prices = self.price_history[symbol]
        volumes = self.volume_history[symbol]
        
        # Guard: Wait until the buffer has enough data to reliably calculate technical features
        if len(prices) < 20:
            return None
            
        df = pd.DataFrame({'close': prices, 'volume': volumes})
        
        # Calculate localized tech metrics on the running slice
        rsi = float(ta.momentum.rsi(df['close'], window=14).iloc[-1])
        macd = float(ta.trend.macd(df['close']).fillna(0).iloc[-1])
        sma = float(ta.trend.sma_indicator(df['close'], window=15).fillna(prices[-1]).iloc[-1])
        
        # 10-Dimensional Vector State Matching Agent Parameter Formats
        market_features = [prices[-1], float(volumes[-1]), rsi, macd, sma]
        agent_features = [portfolio_balance / 10000.0, float(shares_held), total_net_worth / 10000.0, 1.0, prices[-1]]
        
        return np.array(market_features + agent_features, dtype=np.float32)
