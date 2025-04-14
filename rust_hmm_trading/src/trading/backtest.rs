//! Backtesting engine

use super::metrics::TradingMetrics;
use super::portfolio::Portfolio;
use crate::data::{Dataset, Features};
use crate::models::GaussianHMM;
use crate::regime::{Regime, RegimeDetector};
use crate::strategies::StrategySwitcher;

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Transaction cost rate
    pub transaction_cost: f64,
    /// Annualization factor (e.g., 2190 for 4h candles)
    pub annualization: f64,
    /// Risk-free rate (annual)
    pub risk_free_rate: f64,
    /// Maximum position size as fraction of equity
    pub max_position: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            transaction_cost: 0.001,      // 0.1%
            annualization: 2190.0,        // 4h candles per year
            risk_free_rate: 0.04,         // 4%
            max_position: 1.0,            // 100% max
        }
    }
}

/// Backtest result
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Performance metrics
    pub metrics: TradingMetrics,
    /// Equity curve
    pub equity_curve: Vec<f64>,
    /// Returns
    pub returns: Vec<f64>,
    /// Regime at each point
    pub regimes: Vec<Regime>,
    /// Position at each point
    pub positions: Vec<f64>,
    /// Number of strategy switches
    pub num_switches: usize,
}

impl BacktestResult {
    /// Print summary
    pub fn print_summary(&self) {
        println!("\n=== Backtest Results ===\n");
        self.metrics.print_summary();
        println!("\nStrategy Switches: {}", self.num_switches);
        println!("Data Points:       {}", self.equity_curve.len());
    }
}

/// Backtesting engine
pub struct Backtest {
    config: BacktestConfig,
}

impl Backtest {
    /// Create new backtest with default config
    pub fn new() -> Self {
        Self {
            config: BacktestConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest with HMM regime detection
    pub fn run(
        &self,
        dataset: &Dataset,
        features: &Features,
        hmm: &GaussianHMM,
    ) -> anyhow::Result<BacktestResult> {
        let n = dataset.candles.len();
        let n_features = features.n_samples();

        if n < 100 {
            anyhow::bail!("Need at least 100 data points for backtest");
        }

        // Initialize components
        let mut portfolio = Portfolio::new(self.config.initial_capital)
            .with_transaction_cost(self.config.transaction_cost);
        let mut switcher = StrategySwitcher::new();
        let detector = RegimeDetector::new(hmm.clone());

        // Detect regimes for all data
        let regime_states = detector.detect_smoothed(features)?;

        // Align candles with features (features start later due to lookback)
        let offset = n - n_features;

        let mut regimes = Vec::with_capacity(n_features);
        let mut positions = Vec::with_capacity(n_features);

        // Run backtest
        for i in 0..n_features {
            let candle_idx = offset + i;
            let candle = &dataset.candles[candle_idx];
            let history = &dataset.candles[offset.max(1)..candle_idx + 1];
            let regime_state = &regime_states[i];

            // Update strategy based on regime
            switcher.update_regime(regime_state, candle.timestamp);

            // Get target position
            let target = switcher.target_position(candle, history);
            let target_clamped = target.clamp(-self.config.max_position, self.config.max_position);

            // Update position
            portfolio.target_fraction(target_clamped, candle.close);

            // Update portfolio with new price
            portfolio.update(candle.close);

            // Record state
            regimes.push(regime_state.regime);
            positions.push(portfolio.position_fraction(candle.close));
        }

        // Calculate metrics
        let metrics = TradingMetrics::calculate(
            &portfolio.returns_history,
            &portfolio.equity_history,
            &portfolio.trade_returns,
            self.config.annualization,
            self.config.risk_free_rate,
        );

        Ok(BacktestResult {
            metrics,
            equity_curve: portfolio.equity_history,
            returns: portfolio.returns_history,
            regimes,
            positions,
            num_switches: switcher.num_switches(),
        })
    }

    /// Run buy-and-hold benchmark
    pub fn run_buy_hold(&self, dataset: &Dataset) -> anyhow::Result<BacktestResult> {
        let n = dataset.candles.len();

        if n < 2 {
            anyhow::bail!("Need at least 2 data points");
        }

        let mut portfolio = Portfolio::new(self.config.initial_capital)
            .with_transaction_cost(self.config.transaction_cost);

        // Buy at start
        let initial_price = dataset.candles[0].close;
        portfolio.target_fraction(1.0, initial_price);

        let mut regimes = Vec::with_capacity(n);
        let mut positions = Vec::with_capacity(n);

        // Hold through entire period
        for candle in &dataset.candles {
            portfolio.update(candle.close);
            regimes.push(Regime::Unknown(0));
            positions.push(1.0);
        }

        // Sell at end
        let final_price = dataset.candles.last().unwrap().close;
        portfolio.close_position(final_price);

        let metrics = TradingMetrics::calculate(
            &portfolio.returns_history,
            &portfolio.equity_history,
            &portfolio.trade_returns,
            self.config.annualization,
            self.config.risk_free_rate,
        );

        Ok(BacktestResult {
            metrics,
            equity_curve: portfolio.equity_history,
            returns: portfolio.returns_history,
            regimes,
            positions,
            num_switches: 0,
        })
    }
}

impl Default for Backtest {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Candle;

    fn sample_dataset() -> Dataset {
        let candles: Vec<Candle> = (0..200)
            .map(|i| {
                let trend = (i as f64 * 0.05).sin() * 10.0;
                let price = 100.0 + trend + (i as f64 * 0.1);
                Candle {
                    timestamp: 1000 + i as u64 * 3600,
                    open: price * 0.99,
                    high: price * 1.02,
                    low: price * 0.98,
                    close: price,
                    volume: 1000.0,
                    turnover: price * 1000.0,
                }
            })
            .collect();

        Dataset::new(candles, "BTCUSDT", "4h")
    }

    #[test]
    fn test_backtest_creation() {
        let backtest = Backtest::new();
        assert_eq!(backtest.config.initial_capital, 10000.0);
    }

    #[test]
    fn test_buy_hold() {
        let dataset = sample_dataset();
        let backtest = Backtest::new();
        let result = backtest.run_buy_hold(&dataset).unwrap();

        assert!(!result.equity_curve.is_empty());
        assert_eq!(result.num_switches, 0);
    }
}
