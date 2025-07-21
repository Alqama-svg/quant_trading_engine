#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

class PairsTradingStrategy:
    def __init__(self, window=30, z_entry=1.0, z_exit=0.0):
        self.window = window
        self.z_entry = z_entry
        self.z_exit = z_exit
    
    def generate_signals(self, df1, df2):
        """
        Generate trading signals for pairs trading
        
        Args:
            df1: DataFrame with 'price' column for first asset
            df2: DataFrame with 'price' column for second asset
            
        Returns:
            DataFrame with trading signals and strategy returns
        """
        
        price1 = df1['price']
        price2 = df2['price']
        
        # Align the data by index (important!)
        aligned_data = pd.concat([price1, price2], axis=1, join='inner').dropna()
        price1_aligned = aligned_data.iloc[:, 0]
        price2_aligned = aligned_data.iloc[:, 1]
        
        # Calculate spread (you might want to use hedge ratio instead of simple difference)
        spread = price1_aligned - price2_aligned
        
        # Calculate rolling statistics
        rolling_mean = spread.rolling(window=self.window, min_periods=1).mean()
        rolling_std = spread.rolling(window=self.window, min_periods=1).std()
        
        # Calculate z-score
        z_score = (spread - rolling_mean) / rolling_std
        z_score = z_score.fillna(0)  # Handle NaN values
        
        # Generate trading positions
        # 1 = Long spread (buy asset 1, sell asset 2)
        # -1 = Short spread (sell asset 1, buy asset 2)
        # 0 = No position
        position = pd.Series(index=z_score.index, data=0.0)
        
        # Entry signals (FIXED: Added missing # for comment)
        position[z_score > self.z_entry] = -1  # Short the spread when z-score is high
        position[z_score < -self.z_entry] = 1   # Long the spread when z-score is low
        
        # Exit signals
        exit_mask = (np.abs(z_score) < self.z_exit)
        position[exit_mask] = 0
        
        # Forward fill positions to maintain state
        position = position.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        # Calculate returns
        spread_returns = spread.pct_change().fillna(0)
        strategy_returns = position.shift(1) * spread_returns
        
        # Create results DataFrame (FIXED: Added missing columns)
        results_df = pd.DataFrame({
            'price1': price1_aligned,
            'price2': price2_aligned,
            'spread': spread,
            'z_score': z_score,  # FIXED: Changed from 'zscore' to 'z_score'
            'mean': rolling_mean,  # ADDED: Missing column
            'std': rolling_std,    # ADDED: Missing column
            'position': position,
            'returns': spread_returns,
            'strategy': strategy_returns,
            'net_strategy': strategy_returns  # Required by StrategyAnalytics
        })
        
        # Calculate equity curve
        results_df['equity_curve'] = 100000 * (1 + results_df['strategy']).cumprod()
        
        return results_df.dropna()
    
    def run_backtest(self, ticker1, ticker2, start='2020-01-01', end='2024-12-31', initial_cash=100_000):
        """Run complete backtest for pairs trading strategy"""
        try:
            # Download data with warning suppression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                df1_raw = yf.download(ticker1, start=start, end=end, auto_adjust=True, progress=False)
                df2_raw = yf.download(ticker2, start=start, end=end, auto_adjust=True, progress=False)
            
            # Check if data was downloaded successfully
            if df1_raw.empty or df2_raw.empty:
                raise ValueError(f"No data available for {ticker1} or {ticker2}")
            
            # Extract close prices and rename columns
            df1 = df1_raw[['Close']].copy()
            df2 = df2_raw[['Close']].copy()
            df1.columns = ['price']
            df2.columns = ['price']
            
            # Remove any missing values
            df1.dropna(inplace=True)
            df2.dropna(inplace=True)
            
            # Align the dataframes to have the same dates
            common_dates = df1.index.intersection(df2.index)
            df1 = df1.loc[common_dates]
            df2 = df2.loc[common_dates]
            
            if len(common_dates) == 0:
                raise ValueError(f"No common trading dates found for {ticker1} and {ticker2}")
            
            # Generate signals and calculate performance
            df = self.generate_signals(df1, df2)
            
            # Scale equity curve by initial cash
            df['equity_curve'] = initial_cash * (1 + df['strategy']).cumprod()
            
            return df
            
        except Exception as e:
            print(f"Error in pairs trading backtest: {e}")
            return pd.DataFrame()
    
    def plot_performance(self, df, ticker1, ticker2):
        """Plot backtest performance results"""
        if df.empty:
            st.error("No data to plot")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot 1: Equity Curve
        axes[0].plot(df.index, df['equity_curve'], label='Pairs Trading Strategy', color='blue', linewidth=2)
        axes[0].set_title(f"Pairs Trading Strategy: {ticker1} vs {ticker2}")
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Z-Score and Trading Signals (FIXED: Use correct column name)
        axes[1].plot(df.index, df['z_score'], label='Z-Score', color='black', linewidth=1)
        axes[1].axhline(y=self.z_entry, color='red', linestyle='--', alpha=0.7, label=f'Entry Threshold ({self.z_entry})')
        axes[1].axhline(y=-self.z_entry, color='red', linestyle='--', alpha=0.7, label=f'Entry Threshold ({-self.z_entry})')
        axes[1].axhline(y=self.z_exit, color='green', linestyle='--', alpha=0.7, label=f'Exit Threshold ({self.z_exit})')
        axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        axes[1].set_ylabel("Z-Score")
        axes[1].set_title("Z-Score and Trading Thresholds")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Price Spread (FIXED: Use correct column names)
        axes[2].plot(df.index, df['spread'], label='Price Spread', color='purple', linewidth=1)
        axes[2].plot(df.index, df['mean'], label='Rolling Mean', color='orange', linewidth=1)
        axes[2].fill_between(df.index, 
                           df['mean'] - 2*df['std'], 
                           df['mean'] + 2*df['std'], 
                           alpha=0.2, color='gray', label='±2 Std Dev')
        axes[2].set_ylabel("Price Spread")
        axes[2].set_xlabel("Date")
        axes[2].set_title("Price Spread and Statistical Bands")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def calculate_performance_metrics(self, df):
        """Calculate key performance metrics for the strategy"""
        if df.empty or 'strategy' not in df.columns:
            return {}
        
        strategy_returns = df['strategy'].dropna()
        
        if len(strategy_returns) == 0:
            return {}
        
        # Calculate metrics
        total_return = df['equity_curve'].iloc[-1] / df['equity_curve'].iloc[0] - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        cumulative = df['equity_curve']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (strategy_returns > 0).sum()
        total_trades = len(strategy_returns[strategy_returns != 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'Total Return': f"{total_return:.2%}",
            'Annualized Return': f"{annualized_return:.2%}",
            'Volatility': f"{volatility:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.3f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Win Rate': f"{win_rate:.2%}",
            'Total Trades': total_trades
        }

# Example usage function for testing
def test_pairs_trading():
    """Test function to verify the pairs trading strategy works"""
    strategy = PairsTradingStrategy(window=20, z_entry=1.5, z_exit=0.5)
    
    # Test with example tickers
    df = strategy.run_backtest('AAPL', 'MSFT', start='2023-01-01', end='2024-01-01')
    
    if not df.empty:
        print("Backtest completed successfully!")
        print(f"Final portfolio value: ${df['equity_curve'].iloc[-1]:,.2f}")
        
        # Calculate and display performance metrics
        metrics = strategy.calculate_performance_metrics(df)
        print("\nPerformance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
    else:
        print("Backtest failed!")

if __name__ == "__main__":
    test_pairs_trading()