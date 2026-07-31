import numpy as np
from config.settings import MAX_CAPITAL_PER_ASSET_PCT, MIN_RESERVE_CASH_PCT

class MasterAllocationCritic:
    """
    Hierarchical supervisor network that reviews trades from individual asset networks 
    and checks capital boundaries before approving executions.
    """
    def __init__(self, asset_universe):
        self.asset_universe = asset_universe

    def evaluate_allocation_requests(self, raw_proposals, portfolio_cash, current_prices, holdings, net_worth):
        """
        Reviews and filters action choices from individual asset agents.
        Proposals: Dict mapping [symbol] -> Action Intent (0: Hold, 1: Buy, 2: Sell)
        Returns: Approved Actions Dict [symbol] -> Verified Action (int)
        """
        approved_actions = {symbol: 0 for symbol in self.asset_universe}
        
        # Calculate current equity values across active positions
        asset_valuations = {s: holdings[s] * current_prices[s] for s in self.asset_universe}
        
        # Phase 1: Approve Liquidation/Sell Orders immediately to free up capital spaces
        for symbol, action in raw_proposals.items():
            if action == 2: # Sell Request
                if holdings[symbol] > 0:
                    approved_actions[symbol] = 2
                    
        # Calculate expected remaining cash balances after approved liquidations
        estimated_cash = portfolio_cash + sum(current_prices[s] for s, act in approved_actions.items() if act == 2)
        min_cash_floor = net_worth * MIN_RESERVE_CASH_PCT

        # Phase 2: Screen Buy Requests against portfolio exposure limits
        for symbol, action in raw_proposals.items():
            if action == 1: # Buy Request
                cost = current_prices[symbol]
                current_allocation_pct = asset_valuations[symbol] / net_worth
                
                # Risk Check A: Ensure the account maintains its minimum cash reserve
                cash_check = (estimated_cash - cost) >= min_cash_floor
                # Risk Check B: Prevent over-allocation into a single stock
                concentration_check = (current_allocation_pct + (cost / net_worth)) <= MAX_CAPITAL_PER_ASSET_PCT
                
                if cash_check and concentration_check:
                    approved_actions[symbol] = 1
                    estimated_cash -= cost # Deduct allocated capital from working cash pool
                else:
                    print(f"[ALLOCATOR VETO] Blocked BUY request for {symbol}. Reason: "
                          f"Cash Check Pass={cash_check} | Concentration Check Pass={concentration_check}")
                    
        return approved_actions
