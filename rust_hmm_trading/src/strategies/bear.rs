//! Bear regime strategy: defensive, low allocation, hedging

use super::base::{ema, rsi, sma, volatility, Signal, Strategy, StrategyType};
use crate::data::Candle;

/// Bear regime strategy
///
/// Characteristics:
/// - Defensive positioning
/// - Low or zero equity allocation
/// - Cash/bonds preference
/// - Sell rallies
/// - Wide stops (market is volatile)
pub struct BearStrategy {
    /// Fast EMA period
    fast_ema: usize,
    /// Slow EMA period
    slow_ema: usize,
    /// RSI overbought level (sell signal)
    rsi_overbought: f64,
    /// Target allocation in bear regime (low)
    target_allocation: f64,
    /// Whether to allow short positions
    allow_short: bool,
}

impl Default for BearStrategy {
    fn default() -> Self {
        Self {
            fast_ema: 8,
            slow_ema: 21,
            rsi_overbought: 60.0, // Lower threshold in bear
            target_allocation: 0.2, // Only 20% in bear
            allow_short: false,
        }
    }
}

impl BearStrategy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_ema_periods(mut self, fast: usize, slow: usize) -> Self {
        self.fast_ema = fast;
        self.slow_ema = slow;
        self
    }

    pub fn with_target_allocation(mut self, allocation: f64) -> Self {
        self.target_allocation = allocation.clamp(0.0, 1.0);
        self
    }

    pub fn with_shorting(mut self, allow: bool) -> Self {
        self.allow_short = allow;
        self
    }
}

impl Strategy for BearStrategy {
    fn name(&self) -> &str {
        "Bear Defensive"
    }

    fn strategy_type(&self) -> StrategyType {
        StrategyType::Bear
    }

    fn generate_signal(&self, candle: &Candle, history: &[Candle]) -> Signal {
        let prices: Vec<f64> = history.iter().map(|c| c.close).collect();

        if prices.len() < self.slow_ema + 1 {
            return Signal::Sell; // Default to defensive in bear
        }

        let fast = ema(&prices, self.fast_ema).unwrap_or(candle.close);
        let slow = ema(&prices, self.slow_ema).unwrap_or(candle.close);
        let rsi_val = rsi(&prices, 14).unwrap_or(50.0);
        let vol = volatility(&prices, 20).unwrap_or(0.03);

        // Trend is down if fast EMA < slow EMA
        let trend_down = fast < slow;

        // Check for relief rally
        let recent_return = if prices.len() >= 5 {
            (candle.close - prices[prices.len() - 5]) / prices[prices.len() - 5]
        } else {
            0.0
        };

        if trend_down {
            // In downtrend - stay defensive
            if rsi_val > self.rsi_overbought {
                // Relief rally getting exhausted
                Signal::StrongSell
            } else if recent_return > 0.05 {
                // Bear market rally - sell into strength
                Signal::Sell
            } else {
                // Stay out
                Signal::Hold
            }
        } else {
            // Potential trend reversal
            if rsi_val < 30.0 && recent_return > 0.0 {
                // Oversold bounce with momentum
                Signal::Buy
            } else if vol > 0.04 {
                // High volatility - stay cautious
                Signal::Hold
            } else {
                Signal::Hold
            }
        }
    }

    fn target_position(&self, candle: &Candle, history: &[Candle]) -> f64 {
        let prices: Vec<f64> = history.iter().map(|c| c.close).collect();

        if prices.len() < self.slow_ema {
            return 0.0; // Stay out in bear by default
        }

        let fast = ema(&prices, self.fast_ema).unwrap_or(candle.close);
        let slow = ema(&prices, self.slow_ema).unwrap_or(candle.close);
        let vol = volatility(&prices, 20).unwrap_or(0.03);

        let mut allocation = self.target_allocation;

        // Reduce further if strong downtrend
        if fast < slow * 0.98 {
            allocation *= 0.5;
        }

        // Reduce for high volatility
        if vol > 0.03 {
            allocation *= 0.5;
        }

        // Allow short position if enabled and in strong downtrend
        if self.allow_short && fast < slow * 0.95 {
            return -allocation; // Negative = short
        }

        allocation
    }

    fn stop_loss(&self, entry_price: f64, current_price: f64) -> Option<f64> {
        // Wider stop in bear (8%) - market is volatile
        let stop = entry_price * 0.92;
        if current_price < stop {
            Some(stop)
        } else {
            None
        }
    }

    fn take_profit(&self, entry_price: f64, current_price: f64) -> Option<f64> {
        // Quick profit taking in bear (5%)
        let target = entry_price * 1.05;
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
            open: close * 1.01,
            high: close * 1.02,
            low: close * 0.99,
            close,
            volume: 1000.0,
            turnover: close * 1000.0,
        }
    }

    #[test]
    fn test_bear_strategy_creation() {
        let strategy = BearStrategy::new();
        assert_eq!(strategy.name(), "Bear Defensive");
        assert_eq!(strategy.strategy_type(), StrategyType::Bear);
    }

    #[test]
    fn test_bear_low_allocation() {
        let strategy = BearStrategy::new();

        // Create downtrend history
        let history: Vec<Candle> = (0..50)
            .map(|i| sample_candle(100.0 - i as f64 * 0.5))
            .collect();
        let current = sample_candle(75.0);

        let allocation = strategy.target_position(&current, &history);
        // Should have low allocation in bear
        assert!(allocation <= strategy.target_allocation);
    }
}
