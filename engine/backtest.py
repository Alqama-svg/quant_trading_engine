#!/usr/bin/env python
# coding: utf-8

# In[25]:

import yfinance as yf
import pandas as pd
import warnings  # ADD THIS MISSING IMPORT

class BacktestEngine:
    def __init__(self, ticker, start_date, end_date, strategy, data=None, initial_cash=100_000):
        # FIXED: Removed duplicate 'strategy' parameter
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.data = data  # Store the optional data parameter
        self.results = None
        
        # FIXED: Download data only if not provided
        if self.data is None:
            self._download_data()
        
        # FIXED: Prepare data and run backtest
        self._prepare_data()

    def _download_data(self):
        """Download data with warning suppression"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.data = yf.download(
                self.ticker, 
                start=self.start_date, 
                end=self.end_date, 
                auto_adjust=True, 
                progress=False
            )

    def _prepare_data(self):
        """Prepare data for backtesting"""
        # Use existing data if provided, otherwise download
        if self.data is None or self.data.empty:
            self._download_data()
        
        # Make a copy to avoid modifying original data
        df = self.data.copy()
        
        # Handle different column structures
        if 'Adj Close' in df.columns:
            df = df[['Adj Close']].copy()
            df.columns = ['price']
        elif 'Close' in df.columns:
            df = df[['Close']].copy()
            df.columns = ['price']
        else:
            raise ValueError(f"No suitable price column found in data for {self.ticker}")

        # Clean data
        df.dropna(inplace=True)
        
        if df.empty:
            raise ValueError(f"No data available for {self.ticker} in the specified date range")

        # Generate trading signals using the strategy
        df = self.strategy.generate_signals(df)
        
        # Calculate returns and strategy performance
        df['returns'] = df['price'].pct_change()
        df['strategy'] = df['position'].shift(1) * df['returns']
        df.dropna(inplace=True)

        # Calculate equity curves
        df['equity_curve'] = self.initial_cash * (1 + df['strategy']).cumprod()
        df['buy_hold'] = self.initial_cash * (1 + df['returns']).cumprod()
        
        # Store results
        self.results = df

    def _run_backtest(self):
        """Run the backtest - this method was called but not defined"""
        # The backtesting logic is now in _prepare_data
        # This method can be used for additional post-processing if needed
        if self.results is not None:
            return self.results
        else:
            raise ValueError("No results available. Data preparation may have failed.")

    def get_performance_metrics(self):
        """Calculate and return key performance metrics"""
        if self.results is None:
            return None
        
        strategy_returns = self.results['strategy']
        buy_hold_returns = self.results['returns']
        
        metrics = {
            'Total Strategy Return': (self.results['equity_curve'].iloc[-1] / self.initial_cash - 1) * 100,
            'Total Buy & Hold Return': (self.results['buy_hold'].iloc[-1] / self.initial_cash - 1) * 100,
            'Strategy Volatility': strategy_returns.std() * (252 ** 0.5) * 100,
            'Strategy Sharpe Ratio': (strategy_returns.mean() / strategy_returns.std()) * (252 ** 0.5) if strategy_returns.std() > 0 else 0,
            'Max Drawdown': self._calculate_max_drawdown(self.results['equity_curve']),
            'Win Rate': (strategy_returns > 0).sum() / len(strategy_returns) * 100,
            'Number of Trades': self._count_trades()
        }
        
        return metrics

    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown"""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min() * 100

    def _count_trades(self):
        """Count the number of trades"""
        if 'position' not in self.results.columns:
            return 0
        position_changes = self.results['position'].diff().fillna(0)
        return (position_changes != 0).sum()

    def plot_performance(self):
        """Plot strategy performance"""
        if self.results is not None:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot equity curves
            ax1.plot(self.results.index, self.results['equity_curve'], 
                    label='Strategy', linewidth=2)
            ax1.plot(self.results.index, self.results['buy_hold'], 
                    label='Buy & Hold', linewidth=2)
            ax1.set_title(f"{self.ticker} Strategy Performance")
            ax1.set_ylabel("Portfolio Value ($)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot positions
            ax2.plot(self.results.index, self.results['price'], 
                    label='Price', alpha=0.7)
            
            # Highlight buy/sell signals if available
            if 'position' in self.results.columns:
                buy_signals = self.results[self.results['position'] == 1]
                sell_signals = self.results[self.results['position'] == 0]
                
                ax2.scatter(buy_signals.index, buy_signals['price'], 
                           color='green', marker='^', s=100, label='Buy Signal')
                ax2.scatter(sell_signals.index, sell_signals['price'], 
                           color='red', marker='v', s=100, label='Sell Signal')
            
            ax2.set_title(f"{self.ticker} Price and Trading Signals")
            ax2.set_ylabel("Price ($)")
            ax2.set_xlabel("Date")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        else:
            print("No results to plot. Run backtest first.")
            return None

    def export_results(self, filename=None):
        """Export results to CSV"""
        if self.results is not None:
            if filename is None:
                filename = f"{self.ticker}_backtest_results.csv"
            self.results.to_csv(filename)
            print(f"Results exported to {filename}")
        else:
            print("No results to export.")






