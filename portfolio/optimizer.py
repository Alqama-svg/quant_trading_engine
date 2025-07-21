#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt
import warnings

def download_and_prepare_data(tickers, start_date='2020-01-01', end_date='2024-12-31'):
    """Download data and prepare returns DataFrame properly"""
    returns_data = {}
    
    for ticker in tickers:
        try:
            # Suppress FutureWarning about auto_adjust
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
            
            if df.empty:
                # FIXED: Added proper indentation here
                raise ValueError(f"No data retrieved for {ticker}")
                
            # Get Close prices and calculate returns
            close_prices = df['Close']
            returns = close_prices.pct_change().dropna()
            returns_data[ticker] = returns
            
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            continue
    
    if not returns_data:
        raise ValueError("No valid data downloaded for any ticker")
    
    # Create DataFrame with proper column names
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()
    
    return returns_df

def optimize_portfolio(returns, risk_aversion=1.0):
    """
    Mean variance optimizer using CVXPY with fallback solvers
    """
    try:
        mu = returns.mean().values  # Expected returns
        cov = returns.cov().values  # Covariance matrix
        n = len(mu)
        
        # Decision variables
        w = cp.Variable(n)
        
        # Objective function: maximize utility (return - risk penalty)
        objective = cp.Maximize(mu.T @ w - risk_aversion * cp.quad_form(w, cov))
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # weights sum to 1
            w >= 0,          # long-only (no short selling)
            w <= 0.5         # max 50% in any single asset
        ]
        
        # Create and solve problem
        problem = cp.Problem(objective, constraints)
        
        # Try different solvers
        solvers_to_try = [cp.ECOS, cp.SCS, cp.CVXOPT]
        
        for solver in solvers_to_try:
            try:
                problem.solve(solver=solver, verbose=False)
                if problem.status == cp.OPTIMAL:
                    weights = pd.Series(w.value, index=returns.columns)
                    return weights
            except Exception as e:
                print(f"Solver {solver} failed: {e}")
                continue
        
        # If optimization fails, return equal weights
        print("Optimization failed, returning equal weights")
        return equal_weight_portfolio(returns)
        
    except Exception as e:
        print(f"Portfolio optimization error: {e}")
        return equal_weight_portfolio(returns)

def equal_weight_portfolio(returns):
    """Create equal weight portfolio"""
    n = returns.shape[1]
    return pd.Series([1/n] * n, index=returns.columns)

# Wrapper functions for Streamlit app
def get_optimized_weights(returns_df, risk_aversion=0.5):
    """Wrapper function for use in Streamlit app"""
    return optimize_portfolio(returns_df, risk_aversion)

def get_equal_weights(returns_df):
    """Wrapper function for use in Streamlit app"""
    return equal_weight_portfolio(returns_df)

# Performance metrics calculation function
def calculate_portfolio_metrics(returns):
    """Calculate portfolio performance metrics"""
    total_return = (1 + returns).cumprod().iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    max_drawdown = ((1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1).min()
    
    return {
        'Total Return': f"{total_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.3f}",
        'Max Drawdown': f"{max_drawdown:.2%}"
    }

# Main execution (for testing purposes)
if __name__ == "__main__":
    # Define tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
    
    # Download and prepare data
    try:
        returns_df = download_and_prepare_data(tickers)
        print("Returns DataFrame shape:", returns_df.shape)
        print("Returns DataFrame head:")
        print(returns_df.head())
        
        # Optimize portfolio
        opt_weights = optimize_portfolio(returns_df, risk_aversion=0.5)
        eq_weights = equal_weight_portfolio(returns_df)
        
        print("\nOptimized Weights:")
        print(opt_weights)
        print("\nEqual Weights:")
        print(eq_weights)
        
        # Calculate portfolio returns
        opt_portfolio_returns = (returns_df * opt_weights).sum(axis=1)
        eq_portfolio_returns = (returns_df * eq_weights).sum(axis=1)
        
        # Calculate equity curves
        initial_capital = 100_000
        opt_equity_curve = initial_capital * (1 + opt_portfolio_returns).cumprod()
        eq_equity_curve = initial_capital * (1 + eq_portfolio_returns).cumprod()
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        plt.plot(opt_equity_curve.index, opt_equity_curve.values, 
                label='Optimized Portfolio', linewidth=2)
        plt.plot(eq_equity_curve.index, eq_equity_curve.values, 
                label='Equal Weight Portfolio', linewidth=2)
        plt.title("Portfolio Performance Comparison")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Display in Streamlit if running in Streamlit context
        try:
            st.pyplot(plt)
        except:
            plt.show()
        
        # Performance metrics
        print("\nOptimized Portfolio Metrics:")
        opt_metrics = calculate_portfolio_metrics(opt_portfolio_returns)
        for metric, value in opt_metrics.items():
            print(f"{metric}: {value}")
        
        print("\nEqual Weight Portfolio Metrics:")
        eq_metrics = calculate_portfolio_metrics(eq_portfolio_returns)
        for metric, value in eq_metrics.items():
            print(f"{metric}: {value}")
            
    except Exception as e:
        print(f"Error in main execution: {e}")



