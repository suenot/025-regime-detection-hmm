//! Bull regime strategy: momentum-based, aggressive positioning

use super::base::{ema, rsi, sma, volatility, Signal, Strategy, StrategyType};
use crate::data::Candle;

/// Bull regime strategy
///
/// Characteristics:
/// - Follow the trend (momentum)
/// - High equity allocation
/// - Buy on pullbacks
/// - Tight stops on reversals
pub struct BullStrategy {
    /// Fast EMA period
    fast_ema: usize,
    /// Slow EMA period
    slow_ema: usize,
    /// RSI oversold level (buy signal)
    rsi_oversold: f64,
    /// Target allocation in bull regime
    target_allocation: f64,
}

impl Default for BullStrategy {
    fn default() -> Self {
        Self {
            fast_ema: 12,
            slow_ema: 26,
            rsi_oversold: 40.0,
            target_allocation: 1.0, // 100% in bull
        }
    }
}

impl BullStrategy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_ema_periods(mut self, fast: usize, slow: usize) -> Self {
        self.fast_ema = fast;
        self.slow_ema = slow;
        self
    }

    pub fn with_rsi_oversold(mut self, level: f64) -> Self {
        self.rsi_oversold = level;
        self
    }

    pub fn with_target_allocation(mut self, allocation: f64) -> Self {
        self.target_allocation = allocation.clamp(0.0, 1.0);
        self
    }
}

impl Strategy for BullStrategy {
    fn name(&self) -> &str {
        "Bull Momentum"
    }

    fn strategy_type(&self) -> StrategyType {
        StrategyType::Bull
    }

    fn generate_signal(&self, candle: &Candle, history: &[Candle]) -> Signal {
        let prices: Vec<f64> = history.iter().map(|c| c.close).collect();

        if prices.len() < self.slow_ema + 1 {
            return Signal::Hold;
        }

        let fast = ema(&prices, self.fast_ema).unwrap_or(candle.close);
        let slow = ema(&prices, self.slow_ema).unwrap_or(candle.close);
        let rsi_val = rsi(&prices, 14).unwrap_or(50.0);

        // Trend is up if fast EMA > slow EMA
        let trend_up = fast > slow;

        // Current price momentum
        let momentum = (candle.close - slow) / slow;

        if trend_up {
            // In uptrend
            if rsi_val < self.rsi_oversold {
                // Oversold in uptrend = buy opportunity
                Signal::StrongBuy
            } else if momentum > 0.02 {
                // Strong momentum
                Signal::Buy
            } else {
                Signal::Hold
            }
        } else {
            // Trend weakening
            if momentum < -0.03 {
                // Significant pullback
                Signal::Sell
            } else if rsi_val > 70.0 {
                // Overbought and trend weakening
                Signal::Sell
            } else {
                Signal::Hold
            }
        }
    }

    fn target_position(&self, candle: &Candle, history: &[Candle]) -> f64 {
        let prices: Vec<f64> = history.iter().map(|c| c.close).collect();

        if prices.len() < self.slow_ema {
            return self.target_allocation * 0.5;
        }

        let fast = ema(&prices, self.fast_ema).unwrap_or(candle.close);
        let slow = ema(&prices, self.slow_ema).unwrap_or(candle.close);
        let vol = volatility(&prices, 20).unwrap_or(0.02);

        // Base allocation
        let mut allocation = self.target_allocation;

        // Reduce if trend weakening
        if fast < slow {
            allocation *= 0.7;
        }

        // Adjust for volatility
        let vol_adj = self.position_size(vol, 1.0);
        allocation *= vol_adj.min(1.0);

        allocation
    }

    fn stop_loss(&self, entry_price: f64, current_price: f64) -> Option<f64> {
        // Tighter stop in bull (3%) - protect profits
        let stop = entry_price * 0.97;
        if current_price < stop {
            Some(stop)
        } else {
            None
        }
    }

    fn take_profit(&self, entry_price: f64, current_price: f64) -> Option<f64> {
        // Let winners run in bull market - no take profit by default
        // Use trailing stop instead
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_candle(close: f64) -> Candle {
        Candle {
            timestamp: 1000,
            open: close * 0.99,
            high: close * 1.01,
            low: close * 0.98,
            close,
            volume: 1000.0,
            turnover: close * 1000.0,
        }
    }

    #[test]
    fn test_bull_strategy_creation() {
        let strategy = BullStrategy::new();
        assert_eq!(strategy.name(), "Bull Momentum");
        assert_eq!(strategy.strategy_type(), StrategyType::Bull);
    }

    #[test]
    fn test_bull_signal_in_uptrend() {
        let strategy = BullStrategy::new();

        // Create uptrend history
        let history: Vec<Candle> = (0..50)
            .map(|i| sample_candle(100.0 + i as f64 * 0.5))
            .collect();
        let current = sample_candle(125.0);

        let signal = strategy.generate_signal(&current, &history);
        // Should be buy or hold in uptrend
        assert!(!signal.is_sell());
    }
}
