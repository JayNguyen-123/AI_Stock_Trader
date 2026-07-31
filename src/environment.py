import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config.settings import (
    INITIAL_PORTFOLIO_BALANCE,
    TRANSACTION_FEE_RATE,
    MARKET_SLIPPAGE_RATE,
    CRITICAL_DRAWDOWN_LIMIT
)


class RealisticTradingEnv(gym.Env):
    """
    A custom market microstructure simulation environment that applies
    proportional transaction costs and fills order pricing with slippage penalties.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(RealisticTradingEnv, self).__init__()
        # Ensure index matching configurations
        self.df = df.reset_index(drop=True)
        
        # Standard Gymnasium space definitions
        # Action space mapping: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space tracking (5 data metrics + 5 agent metrics = 10 float states)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Resets the portfolio state tracking matrices to baseline configuration states"""
        super().reset(seed=seed)
        
        self.balance = INITIAL_PORTFOLIO_BALANCE
        self.shares_held = 0
        self.current_step = 0
        self.net_worth = INITIAL_PORTFOLIO_BALANCE
        
        return self._get_observation(), {}

    def _get_observation(self):
        """Constructs and scales the structural state observation array"""
        # Extract underlying market dataframe technical indicator metrics
        market_features = self.df.iloc[self.current_step][['close', 'volume', 'rsi', 'macd', 'sma']].values
        
        # Extract localized agent performance status indicators
        agent_status = np.array([
            self.balance / INITIAL_PORTFOLIO_BALANCE,
            float(self.shares_held),
            self.net_worth / INITIAL_PORTFOLIO_BALANCE,
            float(self.current_step) / len(self.df),
            self.df.iloc[self.current_step]['close']
        ])
        
        # Flatten structural sub-arrays together
        return np.concatenate([market_features, agent_status]).astype(np.float32)

    def step(self, action):
        """Executes a trade action, processes transaction fees, and steps forward in time"""
        raw_price = self.df.iloc[self.current_step]['close']
        prev_net_worth = self.net_worth
        
        # ======================================================================
        # MARKET MICROSTRUCTURE POSITION CHECK AND ACCOUNTING
        # ======================================================================
        if action == 1:   # BUY Order Pipeline Execution
            # Apply execution slippage (the agent fills at a slightly higher price)
            execution_price = raw_price * (1.0 + MARKET_SLIPPAGE_RATE)
            transaction_cost = execution_price * TRANSACTION_FEE_RATE
            total_cost = execution_price + transaction_cost
            
            if self.balance >= total_cost:
                self.shares_held += 1
                self.balance -= total_cost
                
        elif action == 2: # SELL Order Pipeline Execution
            if self.shares_held > 0:
                # Apply execution slippage (the agent fills at a slightly lower price)
                execution_price = raw_price * (1.0 - MARKET_SLIPPAGE_RATE)
                transaction_cost = execution_price * TRANSACTION_FEE_RATE
                
                self.shares_held -= 1
                self.balance += (execution_price - transaction_cost)
                
        # Advance temporal index step across data array frame bounds
        self.current_step += 1
        
        # Update current step account equity valuation metrics
        current_market_price = self.df.iloc[self.current_step]['close']
        self.net_worth = self.balance + (self.shares_held * current_market_price)
        
        # ======================================================================
        # REWARD SIGNAL DESIGN & GATEWAY TERMINATIONS
        # ======================================================================
        # Reward corresponds to the immediate local portfolio equity delta step variance
        reward = self.net_worth - prev_net_worth
        
        # Stop flag checks
        terminated = self.current_step >= (len(self.df) - 1)
        
        # Early truncation gate triggered if risk drawdown limits are breached
        truncated = self.net_worth < (INITIAL_PORTFOLIO_BALANCE * CRITICAL_DRAWDOWN_LIMIT)
        
        return self._get_observation(), reward, terminated, truncated, {}
