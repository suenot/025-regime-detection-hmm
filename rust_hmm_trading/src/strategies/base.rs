//! Base strategy types and traits

use crate::data::Candle;
use crate::regime::Regime;

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    /// Strong buy signal
    StrongBuy,
    /// Normal buy signal
    Buy,
    /// Hold current position
    Hold,
    /// Normal sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl Signal {
    /// Convert signal to position adjustment (-1.0 to 1.0)
    pub fn to_adjustment(&self) -> f64 {
        match self {
            Signal::StrongBuy => 1.0,
            Signal::Buy => 0.5,
            Signal::Hold => 0.0,
            Signal::Sell => -0.5,
            Signal::StrongSell => -1.0,
        }
    }

    /// Whether this is a buy signal
    pub fn is_buy(&self) -> bool {
        matches!(self, Signal::StrongBuy | Signal::Buy)
    }

    /// Whether this is a sell signal
    pub fn is_sell(&self) -> bool {
        matches!(self, Signal::Sell | Signal::StrongSell)
    }
}

/// Current position
#[derive(Debug, Clone)]
pub struct Position {
    /// Size (positive = long, negative = short, 0 = flat)
    pub size: f64,
    /// Entry price
    pub entry_price: f64,
    /// Entry timestamp
    pub entry_time: u64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            size: 0.0,
            entry_price: 0.0,
            entry_time: 0,
            unrealized_pnl: 0.0,
        }
    }
}

impl Position {
    /// Create flat position
    pub fn flat() -> Self {
        Self::default()
    }

    /// Check if position is flat
    pub fn is_flat(&self) -> bool {
        self.size.abs() < 1e-10
    }

    /// Check if position is long
    pub fn is_long(&self) -> bool {
        self.size > 1e-10
    }

    /// Check if position is short
    pub fn is_short(&self) -> bool {
        self.size < -1e-10
    }

    /// Update unrealized PnL
    pub fn update_pnl(&mut self, current_price: f64) {
        if !self.is_flat() {
            self.unrealized_pnl = self.size * (current_price - self.entry_price);
        }
    }
}

/// Strategy type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyType {
    /// Strategy for bull regime
    Bull,
    /// Strategy for bear regime
    Bear,
    /// Strategy for sideways regime
    Sideways,
    /// Combined adaptive strategy
    Adaptive,
}

impl From<Regime> for StrategyType {
    fn from(regime: Regime) -> Self {
        match regime {
            Regime::Bull => StrategyType::Bull,
            Regime::Bear => StrategyType::Bear,
            Regime::Sideways => StrategyType::Sideways,
            Regime::Unknown(_) => StrategyType::Sideways, // Default to sideways
        }
    }
}

/// Base strategy trait
pub trait Strategy: Send + Sync {
    /// Strategy name
    fn name(&self) -> &str;

    /// Strategy type
    fn strategy_type(&self) -> StrategyType;

    /// Generate signal based on current candle and history
    fn generate_signal(&self, candle: &Candle, history: &[Candle]) -> Signal;

    /// Get target position size (0.0 to 1.0 of portfolio)
    fn target_position(&self, candle: &Candle, history: &[Candle]) -> f64;

    /// Get stop loss level (optional)
    fn stop_loss(&self, entry_price: f64, current_price: f64) -> Option<f64> {
        // Default: 5% stop loss
        let stop = entry_price * 0.95;
        if current_price < stop {
            Some(stop)
        } else {
            None
        }
    }

    /// Get take profit level (optional)
    fn take_profit(&self, entry_price: f64, current_price: f64) -> Option<f64> {
        // Default: 10% take profit
        let target = entry_price * 1.10;
        if current_price > target {
            Some(target)
        } else {
            None
        }
    }

    /// Calculate position size based on volatility
    fn position_size(&self, volatility: f64, base_size: f64) -> f64 {
        // Inverse volatility sizing
        let target_vol = 0.02; // 2% target volatility
        if volatility > 1e-10 {
            (target_vol / volatility).min(2.0) * base_size
        } else {
            base_size
        }
    }
}

/// Calculate simple moving average
pub fn sma(prices: &[f64], period: usize) -> Option<f64> {
    if prices.len() < period {
        return None;
    }
    let sum: f64 = prices[prices.len() - period..].iter().sum();
    Some(sum / period as f64)
}

/// Calculate exponential moving average
pub fn ema(prices: &[f64], period: usize) -> Option<f64> {
    if prices.is_empty() {
        return None;
    }
    let k = 2.0 / (period as f64 + 1.0);
    let mut ema_val = prices[0];
    for &price in prices.iter().skip(1) {
        ema_val = price * k + ema_val * (1.0 - k);
    }
    Some(ema_val)
}

/// Calculate RSI
pub fn rsi(prices: &[f64], period: usize) -> Option<f64> {
    if prices.len() < period + 1 {
        return None;
    }

    let changes: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();
    let recent = &changes[changes.len() - period..];

    let gains: f64 = recent.iter().filter(|&&x| x > 0.0).sum();
    let losses: f64 = recent.iter().filter(|&&x| x < 0.0).map(|x| x.abs()).sum();

    let avg_gain = gains / period as f64;
    let avg_loss = losses / period as f64;

    if avg_loss < 1e-10 {
        return Some(100.0);
    }

    let rs = avg_gain / avg_loss;
    Some(100.0 - 100.0 / (1.0 + rs))
}

/// Calculate volatility (standard deviation of returns)
pub fn volatility(prices: &[f64], period: usize) -> Option<f64> {
    if prices.len() < period + 1 {
        return None;
    }

    let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
    let recent = &returns[returns.len() - period..];

    let mean = recent.iter().sum::<f64>() / period as f64;
    let variance = recent.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / period as f64;

    Some(variance.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_to_adjustment() {
        assert_eq!(Signal::StrongBuy.to_adjustment(), 1.0);
        assert_eq!(Signal::Hold.to_adjustment(), 0.0);
        assert_eq!(Signal::StrongSell.to_adjustment(), -1.0);
    }

    #[test]
    fn test_position_flat() {
        let pos = Position::flat();
        assert!(pos.is_flat());
        assert!(!pos.is_long());
        assert!(!pos.is_short());
    }

    #[test]
    fn test_sma() {
        let prices = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let sma_3 = sma(&prices, 3).unwrap();
        assert!((sma_3 - 13.0).abs() < 1e-10);
    }

    #[test]
    fn test_rsi() {
        // Uptrend should give high RSI
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let rsi_val = rsi(&prices, 14).unwrap();
        assert!(rsi_val > 50.0);
    }
}
