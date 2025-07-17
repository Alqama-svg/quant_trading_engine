# Quantitative Trading Strategy Engine 🚀


A comprehensive, modular, and extensible backtesting framework for developing, validating, and deploying quantitative trading strategies. This engine supports multiple trading approaches, built-in risk controls, performance analytics, statistical testing, and portfolio optimization — all tied together through an intuitive, reusable API.


---


## 🔍 Highlights


- **Backtest core** is designed with a time-series loop architecture: simulate fills, positions, returns, and equity curves.
- **Three distinct strategy templates** ready out of the box:
  - Mean Reversion (z-score)
  - Momentum (moving average crossover)
  - Pairs Trading (spread-based mean reversion)
- **Smart risk management** tools including stop-loss, position sizing by capital, and drawdown constraints.
- **Performance analytics suite**: Sharpe, Calmar Ratio, Max Drawdown, Annualized Return/Volatility.
- **Statistical testing**: Includes both t-test and bootstrap resampling to assess the strategy’s robustness.
- **Portfolio optimization** via mean-variance frontier (CVXPY), plus comparison with equal-weight implementation.
- **Standalone execution** with `main.py`, and interactive Jupyter notebooks for development & demonstration.
- **Scalable**: Easily extendable with new strategies, optimizers, or risk modules.


---


## 📁 Repository Structure


```plaintext
quant_trading_engine/
├── analytics/
│   ├── performance.py           # Performance metrics
│   └── statistical_tests.py     # t-test, bootstrap analysis
├── engine/
│   ├── backtest.py              # Core BacktestEngine class
│   └── execution.py             # ExecutionSimulator for slippage/fees
├── portfolio/
│   └── optimizer.py             # CVXPY-based optimizer, equal weight
├── risk_management/
│   ├── stop_loss.py
│   ├── position_sizing.py
│   └── drawdown_limits.py
├── strategies/
│   ├── mean_reversion.py
│   ├── momentum.py
│   └── pairs_trading.py
├── utils/
│   └── data_loader.py           # Optional API-based market data loader
├── notebooks/                   # Jupyter notebooks for prototyping/analysis
│   └── main.ipynb              # Demo orchestrator notebook
├── tests/                       # (Optional) Unit tests
├── main.py                      # Full pipeline: strategy → risk → analytics → portfolio
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
├── LICENSE                      # MIT License (recommended)
└── .gitignore                   # Ignore caches, pyc, checkpoint files
```

---


## 🚀 Features & Usage


1. ### Backtest Engine
    - Easy-to-use: Plug any `Strategy` subclass into `BacktestEngine()`

    - Computes daily positions, returns, and equity curve

    - Tracks buy-and-hold benchmark alongside strategy for comparison

2. ### Strategy Implementations
    - Mean Reversion: Z-score entry/exits for single assets

    - Momentum: Trend-following logic via moving averages

    - Pairs Trading: Spread-based strategy for hedged pairs (e.g., KO/PEP)
  
```bash
from strategies.mean_reversion import MeanReversionStrategy
strategy = MeanReversionStrategy(window=20, z_entry=1.5, z_exit=0.5)
```
3. ### Risk Management Modules
     **Apply these in sny backtest pipeline**
    
    - `apply_stop_loss(df, stop_loss_pct=0.02)`
    - `apply_position_sizing(df, risk_pct=0.02)`
    - `apply_drawdown_limit(df, max_drawdown=0.1)`
  
5. ### Execution Simulation
      **Simulates per-trade cost and slippage:**

```bash
exec_sim = ExecutionSimulator(slippage=0.0005, fee=0.0005)
df = exec_sim.adjust_returns(df)
```
6. ### Performance & Statistical Analysis
      **Use `StrategyAnalytics` to compute:**

    - Sharpe Ratio, Calmar Ratio, Max Drawdown
    - Annualized Return & Volatility
    - Significance testing via t-test and bootstrap resampling

```bash
analytics = StrategyAnalytics(df, return_col='net_strategy')
print(analytics.summary())
print("Significant?", analytics.is_significant())
```
7. ### Portfolio Optimization
      **Combine strategy returns across tickers, optimize via mean-variance, compare to equal-weight:**

```bash
from portfolio.optimizer import optimize_portfolio, equal_weight_portfolio
w_opt = optimize_portfolio(returns_df, risk_aversion=0.5)
w_eq = equal_weight_portfolio(returns_df)
```

---


## 📈 Example Summary (`main.py`)

### Pipeline runs


```bash
AAPL backtest → apply execution → analytics → risk modules →
Compare MR vs Momentum on MSFT →
Run KO/PEP pairs trading →
Optimize AAPL, MSFT, GOOGL portfolio → plot performance
```
### Plots, printed summaries, and charts are generated along the way.


---


## 🔧 Installation & Setup

```bash
git clone https://github.com/your_user/quant_trading_engine.git
cd quant_trading_engine
pip install -r requirements.txt
```
### Then run full pipeline:

```bash
python main.py
```
### Or explore interactively:

```bash
jupyter notebook main.ipynb
```

---


## 📚 Dependencies
    
    
  - pandas, numpy, matplotlib, seaborn
  - yfinance
  - cvxpy
  - scipy, statsmodels


---


## 🧠 Key Concepts

  - **Backtesting Engine:** Core event loop that feeds historical prices to strategies, records positions, returns, and equity curve.

  - **Strategy Pattern:** Each strategy (mean reversion, momentum, pairs) inherits from a common interface and defines its own signal logic.

  - **Risk Management:** Modular application of capital control rules before return calculation.

  - **Analytics:** Computation of all key metrics and tests to evaluate whether strategy performance is statistically valid.

  - **Portfolio Optimization:** Using mean and covariance of returns to compute optimal weights that maximize return-to-risk tradeoff.


---


## 💼 Use Cases & Extensions

  - Academic Research: Test alpha-generating hypotheses rigorously.

  - Quant Interviews: Demonstrate data pipelines, metrics, and optimization cases.

  - Client Reporting: Use generated charts and metrics in presentations.

  - Production Pivot: Hook into live data feeds, cloud schedulers, and UI dashboards.


---


## 🧩 License & Contributions

  - This project is released under the MIT License. See LICENSE for full terms.

  - Contributions, forks, and pull requests are encouraged, If you add a feature, fix an issue, or refactor, let me know!


---


## ✉️ Get in Touch

Feel free to raise an issue, ask a question, or collaborate.


---


**Empower your quant research‑to‑deployment workflow with this engine!
Build, backtest, analyze, and operate trading strategies — confidently and systematically.**


---


