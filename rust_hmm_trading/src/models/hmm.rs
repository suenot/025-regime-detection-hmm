//! Gaussian Hidden Markov Model implementation

use super::algorithms::{baum_welch_step, forward_backward, viterbi};
use super::gaussian::MultivariateGaussian;
use crate::data::Features;
use ndarray::{Array1, Array2};
use rand::Rng;

/// HMM parameters
#[derive(Debug, Clone)]
pub struct HMMParams {
    /// Number of hidden states
    pub n_states: usize,
    /// Number of features
    pub n_features: usize,
    /// Initial state probabilities
    pub initial_probs: Array1<f64>,
    /// State transition matrix
    pub transition_matrix: Array2<f64>,
    /// Emission distributions (one per state)
    pub emissions: Vec<MultivariateGaussian>,
}

impl HMMParams {
    /// Create random initial parameters
    pub fn random(n_states: usize, n_features: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Random initial probabilities (uniform-ish)
        let mut initial_probs = Array1::zeros(n_states);
        for i in 0..n_states {
            initial_probs[i] = rng.gen::<f64>() + 0.1;
        }
        let sum = initial_probs.sum();
        initial_probs /= sum;

        // Random transition matrix with diagonal dominance
        let mut transition_matrix = Array2::zeros((n_states, n_states));
        for i in 0..n_states {
            for j in 0..n_states {
                if i == j {
                    transition_matrix[[i, j]] = 0.8 + rng.gen::<f64>() * 0.15;
                } else {
                    transition_matrix[[i, j]] = rng.gen::<f64>() * 0.1;
                }
            }
            // Normalize row
            let row_sum: f64 = transition_matrix.row(i).sum();
            for j in 0..n_states {
                transition_matrix[[i, j]] /= row_sum;
            }
        }

        // Random emissions
        let mut emissions = Vec::new();
        for i in 0..n_states {
            // Spread means across feature space
            let mut mean = Array1::zeros(n_features);
            for j in 0..n_features {
                mean[j] = (i as f64 - (n_states as f64 / 2.0)) * 0.5 + rng.gen::<f64>() * 0.2;
            }
            emissions.push(MultivariateGaussian::with_identity(mean));
        }

        Self {
            n_states,
            n_features,
            initial_probs,
            transition_matrix,
            emissions,
        }
    }
}

/// Gaussian Hidden Markov Model
#[derive(Debug, Clone)]
pub struct GaussianHMM {
    /// Model parameters
    pub params: HMMParams,
    /// Whether the model is trained
    pub is_fitted: bool,
    /// Training log-likelihood history
    pub log_likelihood_history: Vec<f64>,
    /// Convergence tolerance
    pub tol: f64,
}

impl GaussianHMM {
    /// Create new untrained HMM with given number of states
    pub fn new(n_states: usize) -> Self {
        Self {
            params: HMMParams {
                n_states,
                n_features: 0,
                initial_probs: Array1::zeros(0),
                transition_matrix: Array2::zeros((0, 0)),
                emissions: vec![],
            },
            is_fitted: false,
            log_likelihood_history: vec![],
            tol: 1e-4,
        }
    }

    /// Create with specified parameters
    pub fn with_params(params: HMMParams) -> Self {
        Self {
            params,
            is_fitted: true,
            log_likelihood_history: vec![],
            tol: 1e-4,
        }
    }

    /// Set convergence tolerance
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Number of states
    pub fn n_states(&self) -> usize {
        self.params.n_states
    }

    /// Fit the model to data using Baum-Welch (EM) algorithm
    ///
    /// # Arguments
    /// * `features` - Feature matrix from FeatureBuilder
    /// * `n_iter` - Maximum number of iterations
    ///
    /// # Returns
    /// Final log-likelihood
    pub fn fit(&mut self, features: &Features, n_iter: usize) -> anyhow::Result<f64> {
        let observations = &features.data;
        let n_features = observations.ncols();

        if observations.nrows() < 10 {
            anyhow::bail!("Need at least 10 observations for training");
        }

        // Initialize parameters randomly
        self.params = HMMParams::random(self.params.n_states, n_features);

        // K-means style initialization for emission means
        self.initialize_emissions_kmeans(observations);

        self.log_likelihood_history.clear();
        let mut prev_ll = f64::NEG_INFINITY;

        for iter in 0..n_iter {
            // Baum-Welch step
            let (new_initial, new_transition, gamma, log_ll) = baum_welch_step(
                observations,
                &self.params.initial_probs,
                &self.params.transition_matrix,
                &self.params.emissions,
            );

            // Update parameters
            self.params.initial_probs = new_initial;
            self.params.transition_matrix = new_transition;

            // Update emission distributions
            for j in 0..self.params.n_states {
                let weights = gamma.column(j).to_owned();
                self.params.emissions[j].update_weighted(observations, &weights);
            }

            self.log_likelihood_history.push(log_ll);

            // Check convergence
            if (log_ll - prev_ll).abs() < self.tol {
                tracing::info!("Converged after {} iterations", iter + 1);
                break;
            }

            prev_ll = log_ll;

            if (iter + 1) % 10 == 0 {
                tracing::debug!("Iteration {}: log-likelihood = {:.4}", iter + 1, log_ll);
            }
        }

        self.is_fitted = true;
        Ok(*self.log_likelihood_history.last().unwrap_or(&0.0))
    }

    /// Initialize emissions using simple k-means
    fn initialize_emissions_kmeans(&mut self, observations: &Array2<f64>) {
        let n = observations.nrows();
        let d = observations.ncols();
        let k = self.params.n_states;

        // Select k random points as initial centers
        let mut rng = rand::thread_rng();
        let mut centers: Vec<Array1<f64>> = Vec::new();

        for _ in 0..k {
            let idx = rng.gen_range(0..n);
            centers.push(observations.row(idx).to_owned());
        }

        // Run a few k-means iterations
        for _ in 0..10 {
            // Assign points to nearest center
            let mut assignments = vec![0; n];
            for i in 0..n {
                let mut best_dist = f64::MAX;
                for (j, center) in centers.iter().enumerate() {
                    let dist: f64 = observations
                        .row(i)
                        .iter()
                        .zip(center.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    if dist < best_dist {
                        best_dist = dist;
                        assignments[i] = j;
                    }
                }
            }

            // Update centers
            for j in 0..k {
                let mut new_center = Array1::zeros(d);
                let mut count = 0;
                for i in 0..n {
                    if assignments[i] == j {
                        new_center += &observations.row(i);
                        count += 1;
                    }
                }
                if count > 0 {
                    new_center /= count as f64;
                    centers[j] = new_center;
                }
            }
        }

        // Set emission means
        for (j, center) in centers.into_iter().enumerate() {
            self.params.emissions[j] = MultivariateGaussian::with_identity(center);
        }
    }

    /// Predict most likely state sequence (Viterbi)
    pub fn predict(&self, features: &Features) -> anyhow::Result<Vec<usize>> {
        if !self.is_fitted {
            anyhow::bail!("Model not fitted yet");
        }

        let (path, _) = viterbi(
            &features.data,
            &self.params.initial_probs,
            &self.params.transition_matrix,
            &self.params.emissions,
        );

        Ok(path)
    }

    /// Get posterior state probabilities
    pub fn predict_proba(&self, features: &Features) -> anyhow::Result<Array2<f64>> {
        if !self.is_fitted {
            anyhow::bail!("Model not fitted yet");
        }

        let (_, _, gamma, _) = forward_backward(
            &features.data,
            &self.params.initial_probs,
            &self.params.transition_matrix,
            &self.params.emissions,
        );

        Ok(gamma)
    }

    /// Get log-likelihood of observations
    pub fn score(&self, features: &Features) -> anyhow::Result<f64> {
        if !self.is_fitted {
            anyhow::bail!("Model not fitted yet");
        }

        let (_, _, _, log_ll) = forward_backward(
            &features.data,
            &self.params.initial_probs,
            &self.params.transition_matrix,
            &self.params.emissions,
        );

        Ok(log_ll)
    }

    /// Sample a sequence from the model
    pub fn sample(&self, length: usize) -> anyhow::Result<(Vec<usize>, Array2<f64>)> {
        if !self.is_fitted {
            anyhow::bail!("Model not fitted yet");
        }

        let mut rng = rand::thread_rng();
        let mut states = Vec::with_capacity(length);
        let mut observations =
            Array2::zeros((length, self.params.emissions[0].dim()));

        // Sample initial state
        let mut current_state = sample_discrete(&self.params.initial_probs, &mut rng);
        states.push(current_state);

        // Sample observation from emission
        let obs = self.params.emissions[current_state].sample();
        observations.row_mut(0).assign(&obs);

        // Sample rest of sequence
        for t in 1..length {
            // Sample next state from transition
            let trans_probs = self.params.transition_matrix.row(current_state);
            current_state = sample_discrete(&trans_probs.to_owned(), &mut rng);
            states.push(current_state);

            // Sample observation
            let obs = self.params.emissions[current_state].sample();
            observations.row_mut(t).assign(&obs);
        }

        Ok((states, observations))
    }

    /// Get the transition matrix
    pub fn transition_matrix(&self) -> &Array2<f64> {
        &self.params.transition_matrix
    }

    /// Get emission means for each state
    pub fn emission_means(&self) -> Vec<Array1<f64>> {
        self.params.emissions.iter().map(|e| e.mean.clone()).collect()
    }

    /// Calculate AIC (Akaike Information Criterion)
    pub fn aic(&self, features: &Features) -> anyhow::Result<f64> {
        let log_ll = self.score(features)?;
        let n_params = self.count_parameters();
        Ok(2.0 * n_params as f64 - 2.0 * log_ll)
    }

    /// Calculate BIC (Bayesian Information Criterion)
    pub fn bic(&self, features: &Features) -> anyhow::Result<f64> {
        let log_ll = self.score(features)?;
        let n_params = self.count_parameters();
        let n_samples = features.n_samples() as f64;
        Ok(n_params as f64 * n_samples.ln() - 2.0 * log_ll)
    }

    /// Count number of free parameters
    fn count_parameters(&self) -> usize {
        let n = self.params.n_states;
        let d = self.params.n_features;

        // Initial: n-1 (one is constrained)
        // Transition: n * (n-1)
        // Means: n * d
        // Covariances: n * d * (d+1) / 2 (symmetric)

        (n - 1) + n * (n - 1) + n * d + n * d * (d + 1) / 2
    }
}

/// Sample from discrete distribution
fn sample_discrete<R: Rng>(probs: &Array1<f64>, rng: &mut R) -> usize {
    let u: f64 = rng.gen();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u < cumsum {
            return i;
        }
    }
    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_hmm_creation() {
        let hmm = GaussianHMM::new(3);
        assert_eq!(hmm.n_states(), 3);
        assert!(!hmm.is_fitted);
    }

    #[test]
    fn test_hmm_params_random() {
        let params = HMMParams::random(3, 5);
        assert_eq!(params.n_states, 3);
        assert_eq!(params.n_features, 5);

        // Check initial probs sum to 1
        let sum: f64 = params.initial_probs.sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check transition rows sum to 1
        for i in 0..3 {
            let row_sum: f64 = params.transition_matrix.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }
}
