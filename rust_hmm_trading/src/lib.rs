//! # HMM Trading
//!
//! Hidden Markov Model based regime detection for cryptocurrency trading.
//!
//! This library provides:
//! - Bybit API client for fetching market data
//! - Feature engineering for regime detection
//! - Gaussian HMM implementation with Viterbi, Forward-Backward, and Baum-Welch algorithms
//! - Regime detection and interpretation
//! - Trading strategies for Bull, Bear, and Sideways regimes
//! - Backtesting framework
//!
//! ## Quick Example
//!
//! ```rust,no_run
//! use hmm_trading::api::BybitClient;
//! use hmm_trading::models::GaussianHMM;
//! use hmm_trading::regime::RegimeDetector;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch data
//!     let client = BybitClient::new();
//!     let candles = client.get_klines("BTCUSDT", "4h", 1000).await?;
//!
//!     // Build features
//!     let features = hmm_trading::data::build_features(&candles)?;
//!
//!     // Train HMM
//!     let mut hmm = GaussianHMM::new(3); // 3 regimes
//!     hmm.fit(&features, 100)?;
//!
//!     // Detect current regime
//!     let detector = RegimeDetector::new(hmm);
//!     let regime = detector.current_regime(&features)?;
//!
//!     println!("Current regime: {:?}", regime);
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod data;
pub mod models;
pub mod regime;
pub mod strategies;
pub mod trading;

// Re-exports for convenience
pub use api::BybitClient;
pub use data::{Candle, Dataset};
pub use models::GaussianHMM;
pub use regime::{Regime, RegimeDetector};
pub use strategies::{Strategy, StrategyType};
pub use trading::{Backtest, BacktestResult};
