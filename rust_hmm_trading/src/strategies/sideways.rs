//! Sideways regime strategy: mean-reversion, range trading

use super::base::{rsi, sma, volatility, Signal, Strategy, StrategyType};
use crate::data::Candle;

/// Sideways regime strategy
///
/// Characteristics:
/// - Mean-reversion based
/// - Buy at support, sell at resistance
/// - Moderate allocation
/// - Range-bound trading
pub struct SidewaysStrategy {
    /// RSI period
    rsi_period: usize,
    /// RSI oversold level
    rsi_oversold: f64,
    /// RSI overbought level
    rsi_overbought: f64,
    /// Lookback for range calculation
    range_lookback: usize,
    /// Target allocation in sideways regime
    target_allocation: f64,
}

impl Default for SidewaysStrategy {
    fn default() -> Self {
        Self {
            rsi_period: 14,
            rsi_oversold: 30.0,
            rsi_overbought: 70.0,
            range_lookback: 20,
            target_allocation: 0.5, // 50% in sideways
        }
    }
}

impl SidewaysStrategy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_rsi_levels(mut self, oversold: f64, overbought: f64) -> Self {
        self.rsi_oversold = oversold;
        self.rsi_overbought = overbought;
        self
    }

    pub fn with_range_lookback(mut self, lookback: usize) -> Self {
        self.range_lookback = lookback;
        self
    }

    pub fn with_target_allocation(mut self, allocation: f64) -> Self {
        self.target_allocation = allocation.clamp(0.0, 1.0);
        self
    }

    /// Calculate range position (0 = at low, 1 = at high)
    fn range_position(&self, candle: &Candle, history: &[Candle]) -> Option<f64> {
        if history.len() < self.range_lookback {
            return None;
        }

        let recent = &history[history.len() - self.range_lookback..];
        let high = recent.iter().map(|c| c.high).fold(f64::MIN, f64::max);
        let low = recent.iter().map(|c| c.low).fold(f64::MAX, f64::min);

        if high - low > 1e-10 {
            Some((candle.close - low) / (high - low))
        } else {
            Some(0.5)
        }
    }
}

impl Strategy for SidewaysStrategy {
    fn name(&self) -> &str {
        "Sideways Mean-Reversion"
    }

    fn strategy_type(&self) -> StrategyType {
        StrategyType::Sideways
    }

    fn generate_signal(&self, candle: &Candle, history: &[Candle]) -> Signal {
        let prices: Vec<f64> = history.iter().map(|c| c.close).collect();

        if prices.len() < self.range_lookback {
            return Signal::Hold;
        }

        let rsi_val = rsi(&prices, self.rsi_period).unwrap_or(50.0);
        let range_pos = self.range_position(candle, history).unwrap_or(0.5);
        let sma_20 = sma(&prices, 20).unwrap_or(candle.close);

        // Mean-reversion signals
        let distance_from_mean = (candle.close - sma_20) / sma_20;

        // At lower end of range + oversold = buy
        if rsi_val < self.rsi_oversold && range_pos < 0.3 {
            Signal::StrongBuy
        } else if rsi_val < 40.0 && range_pos < 0.4 {
            Signal::Buy
        }
        // At upper end of range + overbought = sell
        else if rsi_val > self.rsi_overbought && range_pos > 0.7 {
            Signal::StrongSell
        } else if rsi_val > 60.0 && range_pos > 0.6 {
            Signal::Sell
        }
        // Far from mean = mean-reversion signal
        else if distance_from_mean < -0.03 {
            Signal::Buy
        } else if distance_from_mean > 0.03 {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn target_position(&self, candle: &Candle, history: &[Candle]) -> f64 {
        let prices: Vec<f64> = history.iter().map(|c| c.close).collect();

        if prices.len() < self.range_lookback {
            return self.target_allocation * 0.5;
        }

        let rsi_val = rsi(&prices, self.rsi_period).unwrap_or(50.0);
        let range_pos = self.range_position(candle, history).unwrap_or(0.5);

        // Position sizing based on distance from mean
        let mut allocation = self.target_allocation;

        // At extremes, increase position (mean-reversion)
        if rsi_val < 30.0 || rsi_val > 70.0 {
            allocation *= 1.2;
        }

        // Scale by range position (contrarian)
        if range_pos < 0.3 {
            allocation *= 1.0 + (0.3 - range_pos);
        } else if range_pos > 0.7 {
            allocation *= 1.0 - (range_pos - 0.7);
        }

        allocation.clamp(0.0, 1.0)
    }

    fn stop_loss(&self, entry_price: f64, current_price: f64) -> Option<f64> {
        // Medium stop in sideways (5%)
        let stop = entry_price * 0.95;
        if current_price < stop {
            Some(stop)
        } else {
            None
        }
    }

    fn take_profit(&self, entry_price: f64, current_price: f64) -> Option<f64> {
        // Take profit at range boundaries (7%)
        let target = entry_price * 1.07;
        if current_price > target {
            Some(target)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_candle(close: f64) -> Candle {
        Candle {
            timestamp: 1000,
            open: close,
            high: close * 1.01,
            low: close * 0.99,
            close,
            volume: 1000.0,
            turnover: close * 1000.0,
        }
    }

    #[test]
    fn test_sideways_strategy_creation() {
        let strategy = SidewaysStrategy::new();
        assert_eq!(strategy.name(), "Sideways Mean-Reversion");
        assert_eq!(strategy.strategy_type(), StrategyType::Sideways);
    }

    #[test]
    fn test_sideways_range_position() {
        let strategy = SidewaysStrategy::new();

        // Create ranging history
        let history: Vec<Candle> = (0..30)
            .map(|i| {
                let price = 100.0 + 5.0 * (i as f64 * 0.3).sin();
                sample_candle(price)
            })
            .collect();

        let at_bottom = sample_candle(95.0);
        let at_top = sample_candle(105.0);

        let bottom_pos = strategy.range_position(&at_bottom, &history);
        let top_pos = strategy.range_position(&at_top, &history);

        assert!(bottom_pos.unwrap() < top_pos.unwrap());
    }
}
