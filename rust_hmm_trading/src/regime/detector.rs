//! Regime detector using trained HMM

use super::interpreter::{Regime, RegimeInfo, RegimeInterpreter};
use crate::data::Features;
use crate::models::GaussianHMM;
use ndarray::Array1;

/// Regime detector combining HMM and interpretation
pub struct RegimeDetector {
    /// Trained HMM model
    hmm: GaussianHMM,
    /// Regime interpreter (optional, set after first fit)
    interpreter: Option<RegimeInterpreter>,
    /// Smoothing window for regime probabilities
    smoothing_window: usize,
    /// Minimum probability threshold for regime switch
    switch_threshold: f64,
    /// Minimum periods in regime before allowing switch
    min_regime_duration: usize,
}

impl RegimeDetector {
    /// Create new detector from trained HMM
    pub fn new(hmm: GaussianHMM) -> Self {
        Self {
            hmm,
            interpreter: None,
            smoothing_window: 5,
            switch_threshold: 0.7,
            min_regime_duration: 5,
        }
    }

    /// Set smoothing window for probability averaging
    pub fn with_smoothing(mut self, window: usize) -> Self {
        self.smoothing_window = window;
        self
    }

    /// Set minimum probability threshold for regime switch
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.switch_threshold = threshold;
        self
    }

    /// Set minimum duration before allowing regime switch
    pub fn with_min_duration(mut self, duration: usize) -> Self {
        self.min_regime_duration = duration;
        self
    }

    /// Initialize interpreter with data
    pub fn fit_interpreter(&mut self, features: &Features, returns: &[f64]) -> anyhow::Result<()> {
        self.interpreter = Some(RegimeInterpreter::from_hmm(&self.hmm, features, returns)?);
        Ok(())
    }

    /// Get the interpreter
    pub fn interpreter(&self) -> Option<&RegimeInterpreter> {
        self.interpreter.as_ref()
    }

    /// Detect current regime from features
    pub fn current_regime(&self, features: &Features) -> anyhow::Result<RegimeState> {
        let probs = self.hmm.predict_proba(features)?;
        let n_samples = probs.nrows();

        if n_samples == 0 {
            anyhow::bail!("No samples to detect regime");
        }

        // Get last few probabilities for smoothing
        let start = n_samples.saturating_sub(self.smoothing_window);
        let mut avg_probs = Array1::zeros(self.hmm.n_states());

        for t in start..n_samples {
            for j in 0..self.hmm.n_states() {
                avg_probs[j] += probs[[t, j]];
            }
        }
        avg_probs /= (n_samples - start) as f64;

        // Find dominant regime
        let (best_state, best_prob) = avg_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, &p)| (i, p))
            .unwrap_or((0, 0.0));

        let regime = self
            .interpreter
            .as_ref()
            .map(|i| i.get_regime(best_state))
            .unwrap_or(Regime::Unknown(best_state));

        Ok(RegimeState {
            regime,
            state_id: best_state,
            probability: best_prob,
            all_probabilities: avg_probs,
            is_confident: best_prob >= self.switch_threshold,
        })
    }

    /// Detect regimes for full history
    pub fn detect_all(&self, features: &Features) -> anyhow::Result<Vec<RegimeState>> {
        let probs = self.hmm.predict_proba(features)?;
        let states = self.hmm.predict(features)?;
        let n_samples = probs.nrows();

        let mut results = Vec::with_capacity(n_samples);

        for t in 0..n_samples {
            let state = states[t];
            let prob = probs[[t, state]];
            let all_probs = probs.row(t).to_owned();

            let regime = self
                .interpreter
                .as_ref()
                .map(|i| i.get_regime(state))
                .unwrap_or(Regime::Unknown(state));

            results.push(RegimeState {
                regime,
                state_id: state,
                probability: prob,
                all_probabilities: all_probs,
                is_confident: prob >= self.switch_threshold,
            });
        }

        Ok(results)
    }

    /// Detect regimes with anti-whipsaw logic
    pub fn detect_smoothed(&self, features: &Features) -> anyhow::Result<Vec<RegimeState>> {
        let raw_states = self.detect_all(features)?;
        let n = raw_states.len();

        if n < self.min_regime_duration {
            return Ok(raw_states);
        }

        let mut smoothed = raw_states.clone();
        let mut current_regime = raw_states[0].regime;
        let mut time_in_regime = 0;

        for i in 0..n {
            let candidate = raw_states[i].regime;

            if candidate == current_regime {
                time_in_regime += 1;
            } else {
                // Check if we should switch
                if raw_states[i].is_confident && time_in_regime >= self.min_regime_duration {
                    // Look ahead to confirm regime change
                    let lookahead = (i + self.min_regime_duration).min(n);
                    let same_regime_count = (i..lookahead)
                        .filter(|&j| raw_states[j].regime == candidate)
                        .count();

                    if same_regime_count >= self.min_regime_duration / 2 {
                        current_regime = candidate;
                        time_in_regime = 1;
                    }
                }
            }

            // Update smoothed state
            smoothed[i] = RegimeState {
                regime: current_regime,
                state_id: smoothed[i].state_id,
                probability: smoothed[i].probability,
                all_probabilities: smoothed[i].all_probabilities.clone(),
                is_confident: smoothed[i].is_confident,
            };
        }

        Ok(smoothed)
    }

    /// Get HMM reference
    pub fn hmm(&self) -> &GaussianHMM {
        &self.hmm
    }

    /// Get transition matrix
    pub fn transition_matrix(&self) -> &ndarray::Array2<f64> {
        self.hmm.transition_matrix()
    }
}

/// Current regime state with probabilities
#[derive(Debug, Clone)]
pub struct RegimeState {
    /// Detected regime
    pub regime: Regime,
    /// HMM state index
    pub state_id: usize,
    /// Probability of this regime
    pub probability: f64,
    /// Probabilities of all regimes
    pub all_probabilities: Array1<f64>,
    /// Whether probability exceeds confidence threshold
    pub is_confident: bool,
}

impl std::fmt::Display for RegimeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} ({:.1}%{})",
            self.regime.emoji(),
            self.regime,
            self.probability * 100.0,
            if self.is_confident { " ✓" } else { "" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_state_display() {
        let state = RegimeState {
            regime: Regime::Bull,
            state_id: 0,
            probability: 0.85,
            all_probabilities: Array1::from_vec(vec![0.85, 0.10, 0.05]),
            is_confident: true,
        };

        let display = format!("{}", state);
        assert!(display.contains("Bull"));
        assert!(display.contains("85.0%"));
    }
}
