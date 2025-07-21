import pandas as pd
import numpy as np
import warnings

class ExecutionSimulator:
    """
    Simulates realistic trade execution with costs, slippage, and other market frictions
    """
    
    def __init__(self, 
                 commission_rate=0.001,  # 0.1% commission per trade
                 bid_ask_spread=0.0005,   # 0.05% bid-ask spread
                 slippage_factor=0.0001,  # 0.01% slippage
                 min_commission=1.0):     # Minimum commission per trade
        self.commission_rate = commission_rate
        self.bid_ask_spread = bid_ask_spread
        self.slippage_factor = slippage_factor
        self.min_commission = min_commission
    
    def adjust_returns(self, df):
        """
        Adjust strategy returns for realistic execution costs
        
        Args:
            df: DataFrame with trading signals and returns
            
        Returns:
            DataFrame with adjusted returns including execution costs
        """
        # Make a copy to avoid modifying original data
        result_df = df.copy()
        
        # Ensure required columns exist
        if 'strategy' not in result_df.columns:
            raise ValueError("DataFrame must contain 'strategy' column with strategy returns")
        
        # Create net_strategy column if it doesn't exist
        if 'net_strategy' not in result_df.columns:
            result_df['net_strategy'] = result_df['strategy'].copy()
        
        # Detect position changes to calculate transaction costs
        if 'position' in result_df.columns:
            result_df = self._apply_transaction_costs(result_df)
        else:
            # If no position column, assume transaction costs are minimal
            # Just apply a small cost reduction
            result_df['net_strategy'] = result_df['strategy'] * (1 - self.commission_rate/2)
        
        # Apply slippage and bid-ask spread
        result_df = self._apply_slippage(result_df)
        
        # Recalculate equity curve with net returns
        if 'equity_curve' in result_df.columns:
            initial_capital = result_df['equity_curve'].iloc[0] / (1 + result_df['strategy'].iloc[0]) if len(result_df) > 0 else 100000
        else:
            initial_capital = 100000
            
        result_df['equity_curve'] = initial_capital * (1 + result_df['net_strategy']).cumprod()
        
        # Add execution statistics
        result_df = self._add_execution_stats(result_df)
        
        return result_df
    
    def _apply_transaction_costs(self, df):
        """Apply transaction costs when positions change"""
        # Calculate position changes
        position_changes = df['position'].diff().fillna(0)
        
        # Identify trades (when position changes)
        trade_mask = (position_changes != 0)
        
        # Calculate transaction costs
        transaction_costs = pd.Series(0.0, index=df.index)
        
        for i, is_trade in enumerate(trade_mask):
            if is_trade and i > 0:  # Skip first row
                # Calculate cost based on position size change
                position_change = abs(position_changes.iloc[i])
                
                if position_change > 0:
                    # Commission cost
                    if 'price' in df.columns:
                        trade_value = position_change * df['price'].iloc[i]
                        commission = max(trade_value * self.commission_rate, self.min_commission)
                    else:
                        # Use a percentage of returns as proxy
                        commission = abs(df['strategy'].iloc[i]) * self.commission_rate * 100
                    
                    # Convert commission to return impact
                    if 'price' in df.columns and df['price'].iloc[i] > 0:
                        cost_as_return = commission / (position_change * df['price'].iloc[i])
                    else:
                        cost_as_return = self.commission_rate
                    
                    transaction_costs.iloc[i] = cost_as_return
        
        # Apply transaction costs to net strategy returns
        df['net_strategy'] = df['strategy'] - transaction_costs
        
        # Store transaction costs for analysis
        df['transaction_costs'] = transaction_costs
        
        return df
    
    def _apply_slippage(self, df):
        """Apply slippage and bid-ask spread costs"""
        # Apply slippage proportional to return magnitude
        if 'position' in df.columns:
            position_changes = df['position'].diff().fillna(0)
            trade_mask = (position_changes != 0)
            
            slippage_costs = pd.Series(0.0, index=df.index)
            slippage_costs[trade_mask] = self.bid_ask_spread + self.slippage_factor
            
            # Apply slippage costs
            df['net_strategy'] = df['net_strategy'] - slippage_costs
            
            # Store slippage costs
            df['slippage_costs'] = slippage_costs
        else:
            # Apply minimal slippage for strategies without explicit positions
            df['net_strategy'] = df['net_strategy'] * (1 - self.bid_ask_spread)
            df['slippage_costs'] = abs(df['strategy']) * self.bid_ask_spread
        
        return df
    
    def _add_execution_stats(self, df):
        """Add execution performance statistics"""
        # Calculate total execution costs
        total_transaction_costs = df.get('transaction_costs', 0).sum()
        total_slippage_costs = df.get('slippage_costs', 0).sum()
        total_execution_costs = total_transaction_costs + total_slippage_costs
        
        # Add execution metrics as attributes (can be accessed later)
        if not hasattr(self, 'execution_stats'):
            self.execution_stats = {}
        
        self.execution_stats.update({
            'total_transaction_costs': total_transaction_costs,
            'total_slippage_costs': total_slippage_costs,
            'total_execution_costs': total_execution_costs,
            'avg_cost_per_trade': total_execution_costs / max(1, self._count_trades(df)),
            'cost_drag_annualized': total_execution_costs * 252 / len(df) if len(df) > 0 else 0
        })
        
        return df
    
    def _count_trades(self, df):
        """Count number of trades executed"""
        if 'position' in df.columns:
            position_changes = df['position'].diff().fillna(0)
            return (position_changes != 0).sum()
        else:
            # Estimate trades based on strategy returns
            return len(df[df['strategy'] != 0])
    
    def get_execution_summary(self):
        """Get summary of execution costs and statistics"""
        if hasattr(self, 'execution_stats'):
            return self.execution_stats
        else:
            return {"message": "No execution statistics available. Run adjust_returns() first."}
    
    def simulate_market_impact(self, df, impact_factor=0.0001):
        """
        Simulate market impact for large trades
        
        Args:
            df: DataFrame with trading data
            impact_factor: Factor to simulate market impact
        """
        if 'position' in df.columns:
            position_changes = df['position'].diff().fillna(0).abs()
            
            # Market impact proportional to position size
            market_impact = position_changes * impact_factor
            
            # Apply market impact to net strategy
            df['net_strategy'] = df['net_strategy'] - market_impact
            df['market_impact'] = market_impact
        
        return df
    
    def apply_realistic_fills(self, df, fill_probability=0.95):
        """
        Simulate realistic order fills (some orders may not get filled)
        
        Args:
            df: DataFrame with trading data
            fill_probability: Probability that an order gets filled
        """
        if 'position' in df.columns:
            # Generate random fill outcomes
            np.random.seed(42)  # For reproducibility
            fill_outcomes = np.random.random(len(df)) < fill_probability
            
            # Adjust positions for unfilled orders
            position_adjusted = df['position'].copy()
            position_changes = df['position'].diff().fillna(0)
            
            # Set position change to 0 for unfilled orders
            unfilled_mask = (position_changes != 0) & (~fill_outcomes)
            position_adjusted[unfilled_mask] = position_adjusted.shift(1)[unfilled_mask]
            
            # Recalculate strategy returns with adjusted positions
            if 'returns' in df.columns:
                df['strategy_adjusted'] = position_adjusted.shift(1) * df['returns']
                df['net_strategy'] = df['strategy_adjusted']
            
            df['position_adjusted'] = position_adjusted
            df['fill_success'] = fill_outcomes
        
        return df

# Enhanced execution simulator for pairs trading
class PairsExecutionSimulator(ExecutionSimulator):
    """
    Specialized execution simulator for pairs trading strategies
    """
    
    def __init__(self, 
                 commission_rate=0.001,
                 bid_ask_spread=0.0005,
                 slippage_factor=0.0001,
                 min_commission=1.0,
                 leg_correlation=0.95):  # Correlation between pair legs
        super().__init__(commission_rate, bid_ask_spread, slippage_factor, min_commission)
        self.leg_correlation = leg_correlation
    
    def adjust_returns(self, df):
        """Adjust returns for pairs trading with leg-specific costs"""
        result_df = df.copy()
        
        # Ensure required columns exist
        if 'strategy' not in result_df.columns:
            raise ValueError("DataFrame must contain 'strategy' column")
        
        if 'net_strategy' not in result_df.columns:
            result_df['net_strategy'] = result_df['strategy'].copy()
        
        # Apply pairs-specific costs
        result_df = self._apply_pairs_costs(result_df)
        
        # Apply general execution costs
        result_df = super()._apply_slippage(result_df)
        
        # Recalculate equity curve
        if 'equity_curve' in result_df.columns:
            initial_capital = 100000  # Default
        else:
            initial_capital = 100000
            
        result_df['equity_curve'] = initial_capital * (1 + result_df['net_strategy']).cumprod()
        
        return result_df
    
    def _apply_pairs_costs(self, df):
        """Apply costs specific to pairs trading"""
        # Double the transaction costs since we're trading two legs
        if 'position' in df.columns:
            position_changes = df['position'].diff().fillna(0)
            trade_mask = (position_changes != 0)
            
            # Pairs trading involves two legs, so double the costs
            pairs_transaction_costs = pd.Series(0.0, index=df.index)
            pairs_transaction_costs[trade_mask] = self.commission_rate * 2  # Two legs
            
            df['net_strategy'] = df['strategy'] - pairs_transaction_costs
            df['pairs_transaction_costs'] = pairs_transaction_costs
        else:
            # Apply minimal cost adjustment for pairs
            df['net_strategy'] = df['strategy'] * (1 - self.commission_rate * 2)
        
        return df

# Factory function to get appropriate simulator
def get_execution_simulator(strategy_type="single", **kwargs):
    """
    Factory function to get appropriate execution simulator
    
    Args:
        strategy_type: Type of strategy ("single", "pairs", "portfolio")
        **kwargs: Additional parameters for the simulator
    
    Returns:
        Appropriate ExecutionSimulator instance
    """
    if strategy_type.lower() == "pairs":
        return PairsExecutionSimulator(**kwargs)
    else:
        return ExecutionSimulator(**kwargs)




