import numpy as np
from config.settings import INITIAL_PORTFOLIO_BALANCE, CRITICAL_DRAWDOWN_LIMIT

class RiskManagementOverlay:
    """
    An independent risk gate that monitors agent outputs, enforces position constraints,
    and manages systemic circuit breakers to protect trading capital.
    """
    def __init__(self, Max_Position_Size=5, Max_Daily_Loss_Pct=0.05, Volatility_Lookback=10):
        self.max_position_size = max_position_size
        self.max_daily_loss_pct = max_daily_loss_pct
        self.volatility_lookback = volatility_lookback
        
        # Risk State Tracking
        self.circuit_breaker_tripped = False
        self.consecutive_losses = 0
        self.loss_streak_threshold = 4  # Veto agent after N straight losses to allow recalibration

    def inspect_and_filter_action(self, agent_action, current_holdings, current_balance, net_worth, price_history):
        """
        Intercepts agent actions and applies defensive compliance filters.
        Action Map: 0 = Hold, 1 = Buy, 2 = Sell
        Returns: Filtered Action (int)
        """
        # 1. Systemic Risk Gate: Circuit Breaker Verification
        if self.circuit_breaker_tripped:
            print("[RISK OVERLAY VETO] Circuit breaker is TRIPPED. Overriding action to HOLD.")
            return 0  # Force Hold

        # 2. Capital Preservation Gate: Lifetime Drawdown Limit
        hard_floor = INITIAL_PORTFOLIO_BALANCE * CRITICAL_DRAWDOWN_LIMIT
        if net_worth <= hard_floor:
            self.circuit_breaker_tripped = True
            print(f"[CRITICAL CIRCUIT BREAKER] Portfolio value (${net_worth:.2f}) breached maximum drawdown floor (${hard_floor:.2f}). System Halted.")
            return 0

        # 3. Market Stress Gate: Historical Volatility Filter
        if len(price_history) >= self.volatility_lookback:
            recent_prices = price_history[-self.volatility_lookback:]
            returns = np.diff(recent_prices) / recent_prices[:-1]
            realized_vol = np.std(returns)
            
            # Veto aggressive buys if market volatility spikes over a 5% threshold
            if realized_vol > 0.05 and agent_action == 1:
                print(f"[RISK OVERLAY VETO] Market volatility spike detected ({realized_vol:.4f}). Vetoing BUY action to HOLD.")
                return 0

        # 4. Exposure Protection Gate: Position Concentration Constraints
        if agent_action == 1 and current_holdings >= self.max_position_size:
            print(f"[RISK OVERLAY VETO] Position size limit reached ({current_holdings}/{self.max_position_size} shares). Vetoing BUY action to HOLD.")
            return 0

        # 5. Adaptive Guard: Loss Streak Dampener
        if self.consecutive_losses >= self.loss_streak_threshold and agent_action == 1:
            print(f"[RISK OVERLAY VETO] Loss streak active ({self.consecutive_losses} losses). Defensively overriding BUY action to HOLD.")
            return 0

        return agent_action

    def update_risk_metrics(self, step_reward):
        """Tracks rolling performance anomalies to adaptively manage gating parameters"""
        if step_reward < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0  # Reset on a profitable trade
