//! Trading and backtesting module

mod backtest;
mod metrics;
mod portfolio;

pub use backtest::{Backtest, BacktestConfig, BacktestResult};
pub use metrics::{calculate_sharpe, calculate_max_drawdown, calculate_calmar, TradingMetrics};
pub use portfolio::Portfolio;
