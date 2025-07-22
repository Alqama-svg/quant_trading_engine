import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Quantitative Trading Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import your trading engine modules
try:
    from engine.backtest import BacktestEngine
    from strategies.mean_reversion import MeanReversionStrategy
    from strategies.momentum import MomentumStrategy
    from strategies.pairs_trading import PairsTradingStrategy
    from analytics.performance import StrategyAnalytics
    from risk_management.stop_loss import apply_stop_loss
    from risk_management.position_sizing import apply_position_sizing
    from risk_management.drawdown_limits import apply_drawdown_limit
    from engine.execution import ExecutionSimulator
    from portfolio.optimizer import optimize_portfolio, equal_weight_portfolio
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please make sure all required modules are available in your repository.")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .strategy-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">🚀 Quantitative Trading Engine</h1>', unsafe_allow_html=True)
    
    # Add Railway deployment badge
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <span style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
            🚆 Deployed on Railway
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choose Strategy",
        ["🏠 Home", "📉 Mean Reversion", "📈 Momentum", "⚖️ Pairs Trading", "🎯 Portfolio Optimization", "🛡️ Risk Management"],
        index=0
    )
    
    # Add sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Quick Info")
    st.sidebar.info("""
    **Live Demo Features:**
    - Real-time strategy backtesting
    - Interactive parameter tuning
    - Performance analytics
    - Risk management tools
    - Portfolio optimization
    """)
    
    # Route to different pages
    if page == "🏠 Home":
        show_home()
    elif page == "📉 Mean Reversion":
        show_mean_reversion()
    elif page == "📈 Momentum":
        show_momentum()
    elif page == "⚖️ Pairs Trading":
        show_pairs_trading()
    elif page == "🎯 Portfolio Optimization":
        show_portfolio_optimization()
    elif page == "🛡️ Risk Management":
        show_risk_management()

def show_home():
    st.markdown("## 🎯 Welcome to the Quantitative Trading Engine")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="strategy-card">
        <h3>🔧 Advanced Features</h3>
        <ul>
        <li><strong>Backtesting Engine</strong>: Time-series simulation with realistic fills</li>
        <li><strong>Strategy Templates</strong>: Mean Reversion, Momentum, Pairs Trading</li>
        <li><strong>Risk Management</strong>: Stop-loss, position sizing, drawdown limits</li>
        <li><strong>Performance Analytics</strong>: Sharpe, Calmar, statistical testing</li>
        <li><strong>Portfolio Optimization</strong>: Mean-variance frontier optimization</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="strategy-card">
        <h3>📈 Available Strategies</h3>
        <ul>
        <li><strong>Mean Reversion</strong>: Z-score based</li>
        <li><strong>Momentum</strong>: MA crossover</li>
        <li><strong>Pairs Trading</strong>: Spread-based</li>
        <li><strong>Risk Controls</strong>: Multiple tools</li>
        <li><strong>Analytics</strong>: Comprehensive metrics</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Demo section
    st.markdown("## 🎮 Quick Demo - Mean Reversion Strategy")
    
    demo_col1, demo_col2, demo_col3 = st.columns(3)
    
    with demo_col1:
        demo_symbol = st.selectbox("📊 Select Stock", ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"], index=0)
    
    with demo_col2:
        demo_period = st.selectbox("📅 Time Period", ["6mo", "1y", "2y"], index=1)
    
    with demo_col3:
        demo_window = st.slider("🪟 Rolling Window", 10, 50, 20)
    
    if st.button("🚀 Run Quick Demo", key="demo_button"):
        run_demo_strategy(demo_symbol, demo_period, demo_window)

def run_demo_strategy(symbol, period, window):
    """Run a quick demo of mean reversion strategy"""
    try:
        with st.spinner(f"🔄 Running demo strategy on {symbol}..."):
            # Download sample data
            data = yf.download(symbol, period=period, progress=False)
            
            if data.empty:
                st.error(f"❌ No data found for symbol {symbol}")
                return
            
            # Simple mean reversion calculation
            data['sma'] = data['Close'].rolling(window=window).mean()
            data['std'] = data['Close'].rolling(window=window).std()
            data['z_score'] = (data['Close'] - data['sma']) / data['std']
            
            # Generate signals
            data['signal'] = 0
            data.loc[data['z_score'] > 1.5, 'signal'] = -1  # Sell when high
            data.loc[data['z_score'] < -1.5, 'signal'] = 1   # Buy when low
            data.loc[abs(data['z_score']) < 0.5, 'signal'] = 0  # Exit
            
            # Calculate returns
            data['returns'] = data['Close'].pct_change()
            data['strategy_returns'] = data['signal'].shift(1) * data['returns']
            data['equity'] = (1 + data['strategy_returns']).cumprod()
            data['benchmark'] = (1 + data['returns']).cumprod()
            
            # Remove NaN values
            data = data.dropna()
            
            st.success("✅ Demo Strategy Complete!")
            
            # Display metrics
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                total_return = (data['equity'].iloc[-1] - 1) * 100
                st.metric("📈 Total Return", f"{total_return:.2f}%")
            
            with metrics_col2:
                benchmark_return = (data['benchmark'].iloc[-1] - 1) * 100
                st.metric("📊 Buy & Hold", f"{benchmark_return:.2f}%")
            
            with metrics_col3:
                if data['strategy_returns'].std() != 0:
                    sharpe = (data['strategy_returns'].mean() / data['strategy_returns'].std()) * np.sqrt(252)
                    st.metric("⚡ Sharpe Ratio", f"{sharpe:.2f}")
                else:
                    st.metric("⚡ Sharpe Ratio", "N/A")
            
            with metrics_col4:
                max_dd = ((data['equity'].expanding().max() - data['equity']) / data['equity'].expanding().max()).max() * 100
                st.metric("📉 Max Drawdown", f"-{max_dd:.2f}%")
            
            # Plot results
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Equity curves
            ax1.plot(data.index, data['equity'], label='Strategy', linewidth=2, color='#667eea')
            ax1.plot(data.index, data['benchmark'], label='Buy & Hold', linewidth=2, color='#ff7f0e')
            ax1.set_title(f'{symbol} - Strategy vs Buy & Hold Performance', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Portfolio Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Z-score
            ax2.plot(data.index, data['z_score'], color='purple', alpha=0.7)
            ax2.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='Sell Signal')
            ax2.axhline(y=-1.5, color='green', linestyle='--', alpha=0.7, label='Buy Signal')
            ax2.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label='Exit Signals')
            ax2.axhline(y=-0.5, color='orange', linestyle=':', alpha=0.7)
            ax2.set_title('Z-Score and Trading Signals', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Z-Score')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Strategy summary
            st.markdown("### 📋 Strategy Summary")
            st.markdown(f"""
            **Strategy**: Mean Reversion using {window}-day rolling window
            **Logic**: 
            - 🔴 **Sell** when Z-score > 1.5 (price too high)
            - 🟢 **Buy** when Z-score < -1.5 (price too low)  
            - 🟡 **Exit** when |Z-score| < 0.5 (price normalizing)
            
            **Performance**: Strategy returned **{total_return:.2f}%** vs Buy & Hold **{benchmark_return:.2f}%**
            """)
            
    except Exception as e:
        st.error(f"❌ Demo failed: {str(e)}")
        st.info("💡 This might be due to data access limitations. The full engine has more robust error handling.")

def show_mean_reversion():
    st.markdown("## 📉 Mean Reversion Strategy")
    st.info("🚧 Advanced mean reversion backtesting with custom parameters - Coming soon in full version!")
    
    # Basic interface
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Features:**
        - Custom Z-score thresholds
        - Multiple entry/exit rules
        - Risk management integration
        - Performance analytics
        """)
    
    with col2:
        st.markdown("""
        **Parameters:**
        - Rolling window size
        - Z-score entry/exit levels
        - Stop-loss percentage
        - Position sizing rules
        """)

def show_momentum():
    st.markdown("## 📈 Momentum Strategy")
    st.info("🚧 Moving average crossover and trend-following strategies - Coming soon!")

def show_pairs_trading():
    st.markdown("## ⚖️ Pairs Trading Strategy")
    st.info("🚧 Spread-based pairs trading with cointegration analysis - Coming soon!")

def show_portfolio_optimization():
    st.markdown("## 🎯 Portfolio Optimization")
    st.info("🚧 Mean-variance optimization and portfolio allocation - Coming soon!")

def show_risk_management():
    st.markdown("## 🛡️ Risk Management")
    st.info("🚧 Stop-loss, position sizing, and drawdown management tools - Coming soon!")

if __name__ == "__main__":
    main()