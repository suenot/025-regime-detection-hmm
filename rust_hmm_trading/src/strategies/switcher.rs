//! Strategy switcher: selects strategy based on current regime

use super::base::{Signal, Strategy, StrategyType};
use super::{BearStrategy, BullStrategy, SidewaysStrategy};
use crate::data::Candle;
use crate::regime::{Regime, RegimeDetector, RegimeState};

/// Strategy switcher that selects strategy based on regime
pub struct StrategySwitcher {
    /// Bull regime strategy
    bull_strategy: BullStrategy,
    /// Bear regime strategy
    bear_strategy: BearStrategy,
    /// Sideways regime strategy
    sideways_strategy: SidewaysStrategy,
    /// Current active strategy type
    current_strategy: StrategyType,
    /// Time spent in current strategy
    time_in_strategy: usize,
    /// Minimum time before switching
    min_switch_time: usize,
    /// History of strategy switches
    switch_history: Vec<(u64, StrategyType)>,
}

impl Default for StrategySwitcher {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategySwitcher {
    /// Create new strategy switcher with default strategies
    pub fn new() -> Self {
        Self {
            bull_strategy: BullStrategy::new(),
            bear_strategy: BearStrategy::new(),
            sideways_strategy: SidewaysStrategy::new(),
            current_strategy: StrategyType::Sideways, // Start defensive
            time_in_strategy: 0,
            min_switch_time: 5,
            switch_history: Vec::new(),
        }
    }

    /// Set custom bull strategy
    pub fn with_bull_strategy(mut self, strategy: BullStrategy) -> Self {
        self.bull_strategy = strategy;
        self
    }

    /// Set custom bear strategy
    pub fn with_bear_strategy(mut self, strategy: BearStrategy) -> Self {
        self.bear_strategy = strategy;
        self
    }

    /// Set custom sideways strategy
    pub fn with_sideways_strategy(mut self, strategy: SidewaysStrategy) -> Self {
        self.sideways_strategy = strategy;
        self
    }

    /// Set minimum switch time
    pub fn with_min_switch_time(mut self, periods: usize) -> Self {
        self.min_switch_time = periods;
        self
    }

    /// Get current active strategy type
    pub fn current_strategy_type(&self) -> StrategyType {
        self.current_strategy
    }

    /// Get current active strategy
    fn current_strategy_ref(&self) -> &dyn Strategy {
        match self.current_strategy {
            StrategyType::Bull => &self.bull_strategy,
            StrategyType::Bear => &self.bear_strategy,
            StrategyType::Sideways | StrategyType::Adaptive => &self.sideways_strategy,
        }
    }

    /// Update strategy based on regime state
    pub fn update_regime(&mut self, regime_state: &RegimeState, timestamp: u64) {
        self.time_in_strategy += 1;

        // Only switch if confident and have been in current strategy long enough
        if regime_state.is_confident && self.time_in_strategy >= self.min_switch_time {
            let new_strategy = StrategyType::from(regime_state.regime);

            if new_strategy != self.current_strategy {
                // Record switch
                self.switch_history.push((timestamp, new_strategy));

                // Switch strategy
                self.current_strategy = new_strategy;
                self.time_in_strategy = 0;

                tracing::info!(
                    "Switched strategy to {:?} (regime: {}, prob: {:.1}%)",
                    new_strategy,
                    regime_state.regime,
                    regime_state.probability * 100.0
                );
            }
        }
    }

    /// Generate signal using current strategy
    pub fn generate_signal(&self, candle: &Candle, history: &[Candle]) -> Signal {
        self.current_strategy_ref().generate_signal(candle, history)
    }

    /// Get target position using current strategy
    pub fn target_position(&self, candle: &Candle, history: &[Candle]) -> f64 {
        self.current_strategy_ref()
            .target_position(candle, history)
    }

    /// Get stop loss using current strategy
    pub fn stop_loss(&self, entry_price: f64, current_price: f64) -> Option<f64> {
        self.current_strategy_ref()
            .stop_loss(entry_price, current_price)
    }

    /// Get take profit using current strategy
    pub fn take_profit(&self, entry_price: f64, current_price: f64) -> Option<f64> {
        self.current_strategy_ref()
            .take_profit(entry_price, current_price)
    }

    /// Get switch history
    pub fn switch_history(&self) -> &[(u64, StrategyType)] {
        &self.switch_history
    }

    /// Get number of switches
    pub fn num_switches(&self) -> usize {
        self.switch_history.len()
    }

    /// Reset switcher state
    pub fn reset(&mut self) {
        self.current_strategy = StrategyType::Sideways;
        self.time_in_strategy = 0;
        self.switch_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn sample_regime_state(regime: Regime, prob: f64) -> RegimeState {
        RegimeState {
            regime,
            state_id: 0,
            probability: prob,
            all_probabilities: Array1::from_vec(vec![prob, 1.0 - prob]),
            is_confident: prob > 0.7,
        }
    }

    #[test]
    fn test_switcher_creation() {
        let switcher = StrategySwitcher::new();
        assert_eq!(switcher.current_strategy_type(), StrategyType::Sideways);
    }

    #[test]
    fn test_switcher_regime_update() {
        let mut switcher = StrategySwitcher::new().with_min_switch_time(1);

        // Start in sideways
        assert_eq!(switcher.current_strategy_type(), StrategyType::Sideways);

        // Update to bull with high confidence
        let bull_state = sample_regime_state(Regime::Bull, 0.85);
        switcher.update_regime(&bull_state, 1000);

        // After min_switch_time, should switch
        let bull_state = sample_regime_state(Regime::Bull, 0.85);
        switcher.update_regime(&bull_state, 2000);

        assert_eq!(switcher.current_strategy_type(), StrategyType::Bull);
    }

    #[test]
    fn test_switcher_no_switch_low_confidence() {
        let mut switcher = StrategySwitcher::new().with_min_switch_time(1);

        // Low confidence should not trigger switch
        let bear_state = sample_regime_state(Regime::Bear, 0.5);
        switcher.update_regime(&bear_state, 1000);
        switcher.update_regime(&bear_state, 2000);

        // Should stay in sideways
        assert_eq!(switcher.current_strategy_type(), StrategyType::Sideways);
    }
}
