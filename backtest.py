import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.settings import (
    STATE_DIMENSION,
    ACTION_DIMENSION,
    DATA_TRAIN_TEST_SPLIT,
    BEST_MODEL_FILE
)
from src.agent import ProductionDDQNAgent
from src.environment import RealisticTradingEnv
from utils.checkpoints import ModelCheckpointManager
from main import generate_synthetic_market_data
from src import RiskManagementOverlay


class OutOfSampleBacktester:
    """Evaluates an agent's trading strategy against unseen historical market data slices"""
    def __init__(self, env):
        self.env = env

    def run_backtest(self, agent):
        """Executes a deterministic performance sweep with neural net backpropagation frozen"""
        # Set network to evaluation mode (freezes dropout, normalization, and noisy layers)
        agent.policy_net.eval()

        # Initialize the independent risk supervisor module
        risk_guard = RiskManagementOverlay(Max_Position_Size=4, Volatility_Lookback=10)
        
        state, _ = self.env.reset()
        portfolio_history = []
        action_history = []
        close_prices = []
        
        print("[Backtester] Initiating out-of-sample trading simulation with Risk Overlay Active...")
    
        
        while True:
           # 1. Query raw target action from the RL model weights
           raw_agent_action = agent.select_action(state)
        
           # 2. Extract current contextual variables from the simulation environment
           current_holdings = self.env.shares_held
           current_balance = self.env.balance
           current_net_worth = self.env.net_worth
        
           # 3. Intercept and screen action via Risk Management Rules
           compliant_action = risk_guard.inspect_and_filter_action(
                agent_action=raw_agent_action,
                current_holdings=current_holdings,
                current_balance=current_balance,
                net_worth=current_net_worth,
                price_history=close_prices
           )
        
           current_close = self.env.df.iloc[self.env.current_step]['close']
           close_prices.append(current_close)
        
           # 4. Step environment forward using the safe, filtered action
           state, reward, terminated, truncated, _ = self.env.step(compliant_action)
        
           # 5. Feed trade outcomes back to the risk layer to track consecutive losses
           risk_guard.update_risk_metrics(step_reward=reward)
        
           portfolio_history.append(self.env.net_worth)
           action_history.append(compliant_action)
        
           if terminated or truncated:
              break
            
        self._calculate_performance_metrics(portfolio_history, action_history)
        self._render_backtest_chart(portfolio_history, close_prices, action_history)
    
        return portfolio_history

    def _calculate_performance_metrics(self, history, actions):
        """Computes institutional risk-adjusted return vectors"""
        history = np.array(history)
        
        # Calculate periodic step returns
        returns = np.diff(history) / (history[:-1] + 1e-9)
        
        initial_worth = history[0]
        final_worth = history[-1]
        total_return = ((final_worth - initial_worth) / initial_worth) * 100
        
        # Calculate Annualized Sharpe Ratio (assuming 252 active market trading days per year)
        mean_return = np.mean(returns) if len(returns) > 0 else 0.0
        std_return = np.std(returns) if len(returns) > 0 else 1.0
        sharpe_ratio = (mean_return / (std_return + 1e-9)) * np.sqrt(252)
        
        # Calculate Peak-to-Trough Maximum Drawdown
        peaks = np.maximum.accumulate(history)
        drawdowns = (history - peaks) / (peaks + 1e-9)
        max_drawdown = np.min(drawdowns) * 100
        
        print("\n" + "="*50 + "\n INSTITUTIONAL OUT-OF-SAMPLE BACKTEST REPORT \n" + "="*50)
        print(f"Initial Portfolio Capital:  ${initial_worth:,.2f}")
        print(f"Final Portfolio Valuation:  ${final_worth:,.2f}")
        print(f"Absolute Strategy Return:   {total_return:.2f}%")
        print(f"Annualized Sharpe Ratio:    {sharpe_ratio:.4f}")
        print(f"Maximum Portfolio Drawdown: {max_drawdown:.2f}%")
        print("-" * 50)
        print(f"Trade Distribution Matrix:  Buys={actions.count(1)} | Sells={actions.count(2)} | Holds={actions.count(0)}")
        print("="*50 + "\n")

    def _render_backtest_chart(self, portfolio_history, close_prices, actions):
        """Generates a performance visual comparing portfolio growth directly to the underlying asset"""
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Equity Curve Acceleration Profile
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(portfolio_history, label='Agent Net Worth (Equity Curve)', color='darkgreen', linewidth=2)
        plt.title('Out-of-Sample Evaluation Portfolio Performance')
        plt.ylabel('Account Value ($)')
        plt.legend(loc='upper left')
        plt.grid(True)
        
        # Subplot 2: Trade Execution Points mapped onto the underlying asset curve
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        plt.plot(close_prices, label='Asset Close Price', color='blue', alpha=0.6)
        
        # Scatter execution shapes across timeline index maps
        buy_indices = [i for i, x in enumerate(actions) if x == 1]
        sell_indices = [i for i, x in enumerate(actions) if x == 2]
        
        plt.scatter(buy_indices, [close_prices[i] for i in buy_indices], color='green', marker='^', alpha=1.0, label='BUY Order')
        plt.scatter(sell_indices, [close_prices[i] for i in sell_indices], color='red', marker='v', alpha=1.0, label='SELL Order')
        
        plt.ylabel('Asset Price ($)')
        plt.xlabel('Sequential Testing Intervals (Steps)')
        plt.legend(loc='upper left')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


def main():
    # 1. Synthesize market data using the exact same generation seed engine parameters
    raw_market_data = generate_synthetic_market_data(records=500)
    split_index = int(len(raw_market_data) * DATA_TRAIN_TEST_SPLIT)
    
    # Isolate the final validation block that was completely hidden from the agent during training
    validation_df = raw_market_data.iloc[split_index:].copy()
    print(f"[Engine] Out-of-sample evaluation dataset loaded containing {len(validation_df)} frames.")

    # 2. Re-instantiate environment and agent layers
    val_env = RealisticTradingEnv(validation_df)
    agent = ProductionDDQNAgent(state_dim=STATE_DIMENSION, action_dim=ACTION_DIMENSION)
    checkpoint_mgr = ModelCheckpointManager()
    
    # 3. Pull our historical benchmark high-score parameter file from the workspace disks
    load_status = checkpoint_mgr.load_checkpoint(agent, BEST_MODEL_FILE)
    
    if load_status == 0:
        print("[CRITICAL FAULT] Backtest aborted. Could not locate a valid 'best_trading_model.pt' file.")
        return

    # 4. Trigger evaluation sweep
    tester = OutOfSampleBacktester(val_env)
    tester.run_backtest(agent)


if __name__ == "__main__":
    main()
