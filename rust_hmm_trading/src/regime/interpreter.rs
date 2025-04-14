//! Regime interpretation based on statistical properties

use crate::data::Features;
use crate::models::GaussianHMM;
use ndarray::Array1;
use std::fmt;

/// Market regime types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Regime {
    /// Bullish regime: upward trend, low volatility
    Bull,
    /// Bearish regime: downward trend, high volatility
    Bear,
    /// Sideways regime: no clear trend, medium volatility
    Sideways,
    /// Unknown regime (before interpretation)
    Unknown(usize),
}

impl fmt::Display for Regime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Regime::Bull => write!(f, "Bull"),
            Regime::Bear => write!(f, "Bear"),
            Regime::Sideways => write!(f, "Sideways"),
            Regime::Unknown(id) => write!(f, "State_{}", id),
        }
    }
}

impl Regime {
    /// Get emoji representation
    pub fn emoji(&self) -> &'static str {
        match self {
            Regime::Bull => "🐂",
            Regime::Bear => "🐻",
            Regime::Sideways => "↔️",
            Regime::Unknown(_) => "❓",
        }
    }

    /// Get color code for terminal output
    pub fn color(&self) -> &'static str {
        match self {
            Regime::Bull => "green",
            Regime::Bear => "red",
            Regime::Sideways => "yellow",
            Regime::Unknown(_) => "white",
        }
    }
}

/// Detailed regime information
#[derive(Debug, Clone)]
pub struct RegimeInfo {
    /// Regime type
    pub regime: Regime,
    /// HMM state index
    pub state_id: usize,
    /// Posterior probability
    pub probability: f64,
    /// Average return in this regime (annualized)
    pub avg_return: f64,
    /// Average volatility (annualized)
    pub avg_volatility: f64,
    /// Average duration (in periods)
    pub avg_duration: f64,
    /// Transition probabilities to other regimes
    pub transitions: Vec<(Regime, f64)>,
}

impl fmt::Display for RegimeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} {} (State {})",
            self.regime.emoji(),
            self.regime,
            self.state_id
        )?;
        writeln!(f, "  Probability: {:.1}%", self.probability * 100.0)?;
        writeln!(f, "  Avg Return: {:.1}%", self.avg_return * 100.0)?;
        writeln!(f, "  Avg Volatility: {:.1}%", self.avg_volatility * 100.0)?;
        writeln!(f, "  Avg Duration: {:.1} periods", self.avg_duration)?;
        writeln!(f, "  Transitions:")?;
        for (regime, prob) in &self.transitions {
            writeln!(f, "    → {}: {:.1}%", regime, prob * 100.0)?;
        }
        Ok(())
    }
}

/// Regime interpreter: maps HMM states to market regimes
pub struct RegimeInterpreter {
    /// Mapping from state index to regime
    state_to_regime: Vec<Regime>,
    /// Regime statistics
    regime_stats: Vec<RegimeInfo>,
}

impl RegimeInterpreter {
    /// Create interpreter by analyzing HMM states
    pub fn from_hmm(
        hmm: &GaussianHMM,
        features: &Features,
        returns: &[f64],
    ) -> anyhow::Result<Self> {
        let n_states = hmm.n_states();

        // Get state assignments
        let states = hmm.predict(features)?;
        let probs = hmm.predict_proba(features)?;

        // Calculate statistics for each state
        let mut state_stats: Vec<(f64, f64, f64)> = vec![(0.0, 0.0, 0.0); n_states];
        let mut state_counts = vec![0usize; n_states];

        for (i, &state) in states.iter().enumerate() {
            if i < returns.len() {
                state_stats[state].0 += returns[i]; // sum of returns
                state_stats[state].1 += returns[i].powi(2); // sum of squared returns
                state_counts[state] += 1;
            }
        }

        // Compute mean return and volatility for each state
        let mut avg_returns = vec![0.0; n_states];
        let mut avg_volatilities = vec![0.0; n_states];

        for i in 0..n_states {
            if state_counts[i] > 0 {
                let n = state_counts[i] as f64;
                avg_returns[i] = state_stats[i].0 / n;
                let var = state_stats[i].1 / n - avg_returns[i].powi(2);
                avg_volatilities[i] = var.sqrt();
            }
        }

        // Map states to regimes based on return and volatility
        let state_to_regime = Self::classify_states(&avg_returns, &avg_volatilities);

        // Calculate average durations
        let avg_durations = Self::calculate_durations(&states, n_states);

        // Build regime info
        let transition_matrix = hmm.transition_matrix();
        let mut regime_stats = Vec::new();

        for state_id in 0..n_states {
            let regime = state_to_regime[state_id];

            // Calculate probability of being in this regime
            let prob = probs.column(state_id).mean().unwrap_or(0.0);

            // Annualize (assuming 4h candles, ~2190 per year)
            let annualization_factor = (2190.0_f64).sqrt();
            let ann_return = avg_returns[state_id] * 2190.0;
            let ann_vol = avg_volatilities[state_id] * annualization_factor;

            // Transitions
            let mut transitions = Vec::new();
            for j in 0..n_states {
                if j != state_id {
                    transitions.push((state_to_regime[j], transition_matrix[[state_id, j]]));
                }
            }

            regime_stats.push(RegimeInfo {
                regime,
                state_id,
                probability: prob,
                avg_return: ann_return,
                avg_volatility: ann_vol,
                avg_duration: avg_durations[state_id],
                transitions,
            });
        }

        Ok(Self {
            state_to_regime,
            regime_stats,
        })
    }

    /// Classify states into regimes based on return/volatility
    fn classify_states(returns: &[f64], volatilities: &[f64]) -> Vec<Regime> {
        let n = returns.len();
        let mut regimes = vec![Regime::Unknown(0); n];

        // Sort states by return
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            returns[b]
                .partial_cmp(&returns[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if n >= 3 {
            // 3-state model: Bull (highest return), Bear (lowest return), Sideways (middle)
            regimes[indices[0]] = Regime::Bull;
            regimes[indices[n - 1]] = Regime::Bear;
            for i in 1..n - 1 {
                regimes[indices[i]] = Regime::Sideways;
            }
        } else if n == 2 {
            // 2-state model: Bull (higher return), Bear (lower return)
            regimes[indices[0]] = Regime::Bull;
            regimes[indices[1]] = Regime::Bear;
        } else if n == 1 {
            regimes[0] = Regime::Sideways;
        }

        // Additional validation using volatility
        // Bear typically has highest volatility
        for i in 0..n {
            // If classified as Bull but has very high volatility, might be Bear
            if regimes[i] == Regime::Bull && volatilities[i] > volatilities.iter().sum::<f64>() / n as f64 * 1.5 {
                // Keep as is, but could be adjusted
            }
        }

        regimes
    }

    /// Calculate average duration of each regime
    fn calculate_durations(states: &[usize], n_states: usize) -> Vec<f64> {
        let mut durations = vec![Vec::new(); n_states];
        let mut current_state = states[0];
        let mut current_duration = 1;

        for &state in states.iter().skip(1) {
            if state == current_state {
                current_duration += 1;
            } else {
                durations[current_state].push(current_duration as f64);
                current_state = state;
                current_duration = 1;
            }
        }
        durations[current_state].push(current_duration as f64);

        durations
            .iter()
            .map(|d| {
                if d.is_empty() {
                    0.0
                } else {
                    d.iter().sum::<f64>() / d.len() as f64
                }
            })
            .collect()
    }

    /// Get regime for a state index
    pub fn get_regime(&self, state: usize) -> Regime {
        self.state_to_regime.get(state).copied().unwrap_or(Regime::Unknown(state))
    }

    /// Get regime info for a state
    pub fn get_regime_info(&self, state: usize) -> Option<&RegimeInfo> {
        self.regime_stats.iter().find(|r| r.state_id == state)
    }

    /// Get all regime statistics
    pub fn all_regimes(&self) -> &[RegimeInfo] {
        &self.regime_stats
    }

    /// Find the Bull regime info
    pub fn bull_regime(&self) -> Option<&RegimeInfo> {
        self.regime_stats.iter().find(|r| r.regime == Regime::Bull)
    }

    /// Find the Bear regime info
    pub fn bear_regime(&self) -> Option<&RegimeInfo> {
        self.regime_stats.iter().find(|r| r.regime == Regime::Bear)
    }

    /// Find the Sideways regime info
    pub fn sideways_regime(&self) -> Option<&RegimeInfo> {
        self.regime_stats.iter().find(|r| r.regime == Regime::Sideways)
    }

    /// Print regime summary
    pub fn print_summary(&self) {
        println!("\n=== Regime Summary ===\n");
        for info in &self.regime_stats {
            println!("{}", info);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_display() {
        assert_eq!(format!("{}", Regime::Bull), "Bull");
        assert_eq!(format!("{}", Regime::Bear), "Bear");
        assert_eq!(format!("{}", Regime::Sideways), "Sideways");
    }

    #[test]
    fn test_classify_states() {
        let returns = vec![0.1, -0.1, 0.0];
        let volatilities = vec![0.1, 0.3, 0.15];
        let regimes = RegimeInterpreter::classify_states(&returns, &volatilities);

        assert_eq!(regimes[0], Regime::Bull);
        assert_eq!(regimes[1], Regime::Bear);
        assert_eq!(regimes[2], Regime::Sideways);
    }
}
