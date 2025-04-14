//! HMM algorithms: Viterbi, Forward-Backward, Baum-Welch

use super::gaussian::MultivariateGaussian;
use ndarray::{Array1, Array2};

/// Viterbi algorithm - finds most likely state sequence
///
/// # Arguments
/// * `observations` - Observation matrix (T x D)
/// * `initial_probs` - Initial state probabilities (N)
/// * `transition_matrix` - State transition probabilities (N x N)
/// * `emissions` - Emission distributions for each state
///
/// # Returns
/// Most likely state sequence and log probability
pub fn viterbi(
    observations: &Array2<f64>,
    initial_probs: &Array1<f64>,
    transition_matrix: &Array2<f64>,
    emissions: &[MultivariateGaussian],
) -> (Vec<usize>, f64) {
    let t = observations.nrows();
    let n = initial_probs.len();

    if t == 0 {
        return (vec![], 0.0);
    }

    // Log probabilities for numerical stability
    let log_initial: Array1<f64> = initial_probs.mapv(|p| (p + 1e-300).ln());
    let log_trans: Array2<f64> = transition_matrix.mapv(|p| (p + 1e-300).ln());

    // Delta: best path probability ending in state j at time t
    let mut delta = Array2::zeros((t, n));
    // Psi: backpointers for path reconstruction
    let mut psi = Array2::zeros((t, n));

    // Initialization (t = 0)
    let obs_0 = observations.row(0).to_owned();
    for j in 0..n {
        delta[[0, j]] = log_initial[j] + emissions[j].log_pdf(&obs_0);
    }

    // Recursion
    for t_idx in 1..t {
        let obs_t = observations.row(t_idx).to_owned();

        for j in 0..n {
            let mut best_val = f64::NEG_INFINITY;
            let mut best_state = 0;

            for i in 0..n {
                let val = delta[[t_idx - 1, i]] + log_trans[[i, j]];
                if val > best_val {
                    best_val = val;
                    best_state = i;
                }
            }

            delta[[t_idx, j]] = best_val + emissions[j].log_pdf(&obs_t);
            psi[[t_idx, j]] = best_state as f64;
        }
    }

    // Termination
    let mut best_final_state = 0;
    let mut best_final_prob = f64::NEG_INFINITY;
    for j in 0..n {
        if delta[[t - 1, j]] > best_final_prob {
            best_final_prob = delta[[t - 1, j]];
            best_final_state = j;
        }
    }

    // Backtracking
    let mut path = vec![0; t];
    path[t - 1] = best_final_state;
    for t_idx in (0..t - 1).rev() {
        path[t_idx] = psi[[t_idx + 1, path[t_idx + 1]]] as usize;
    }

    (path, best_final_prob)
}

/// Forward-Backward algorithm - computes state posterior probabilities
///
/// # Returns
/// (alpha, beta, gamma, log_likelihood)
/// - alpha: forward probabilities (T x N)
/// - beta: backward probabilities (T x N)
/// - gamma: posterior state probabilities (T x N)
/// - log_likelihood: log P(observations | model)
pub fn forward_backward(
    observations: &Array2<f64>,
    initial_probs: &Array1<f64>,
    transition_matrix: &Array2<f64>,
    emissions: &[MultivariateGaussian],
) -> (Array2<f64>, Array2<f64>, Array2<f64>, f64) {
    let t = observations.nrows();
    let n = initial_probs.len();

    if t == 0 {
        return (
            Array2::zeros((0, n)),
            Array2::zeros((0, n)),
            Array2::zeros((0, n)),
            0.0,
        );
    }

    // Compute emission probabilities for all observations
    let mut emission_probs = Array2::zeros((t, n));
    for t_idx in 0..t {
        let obs = observations.row(t_idx).to_owned();
        for j in 0..n {
            emission_probs[[t_idx, j]] = emissions[j].pdf(&obs);
        }
    }

    // Forward pass (with scaling for numerical stability)
    let mut alpha = Array2::zeros((t, n));
    let mut scale = Array1::zeros(t);

    // Initialization
    for j in 0..n {
        alpha[[0, j]] = initial_probs[j] * emission_probs[[0, j]];
    }
    scale[0] = alpha.row(0).sum();
    if scale[0] > 1e-300 {
        for j in 0..n {
            alpha[[0, j]] /= scale[0];
        }
    }

    // Recursion
    for t_idx in 1..t {
        for j in 0..n {
            let mut sum = 0.0;
            for i in 0..n {
                sum += alpha[[t_idx - 1, i]] * transition_matrix[[i, j]];
            }
            alpha[[t_idx, j]] = sum * emission_probs[[t_idx, j]];
        }

        scale[t_idx] = alpha.row(t_idx).sum();
        if scale[t_idx] > 1e-300 {
            for j in 0..n {
                alpha[[t_idx, j]] /= scale[t_idx];
            }
        }
    }

    // Log-likelihood
    let log_likelihood: f64 = scale.iter().map(|s| (s + 1e-300).ln()).sum();

    // Backward pass
    let mut beta = Array2::zeros((t, n));

    // Initialization
    for j in 0..n {
        beta[[t - 1, j]] = 1.0;
    }

    // Recursion
    for t_idx in (0..t - 1).rev() {
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += transition_matrix[[i, j]]
                    * emission_probs[[t_idx + 1, j]]
                    * beta[[t_idx + 1, j]];
            }
            beta[[t_idx, i]] = sum;
        }

        // Scale
        if scale[t_idx + 1] > 1e-300 {
            for i in 0..n {
                beta[[t_idx, i]] /= scale[t_idx + 1];
            }
        }
    }

    // Compute gamma (posterior probabilities)
    let mut gamma = Array2::zeros((t, n));
    for t_idx in 0..t {
        let mut sum = 0.0;
        for j in 0..n {
            gamma[[t_idx, j]] = alpha[[t_idx, j]] * beta[[t_idx, j]];
            sum += gamma[[t_idx, j]];
        }
        if sum > 1e-300 {
            for j in 0..n {
                gamma[[t_idx, j]] /= sum;
            }
        }
    }

    (alpha, beta, gamma, log_likelihood)
}

/// Single Baum-Welch (EM) step
///
/// # Returns
/// (new_initial, new_transition, expected_counts, log_likelihood)
pub fn baum_welch_step(
    observations: &Array2<f64>,
    initial_probs: &Array1<f64>,
    transition_matrix: &Array2<f64>,
    emissions: &[MultivariateGaussian],
) -> (Array1<f64>, Array2<f64>, Array2<f64>, f64) {
    let t = observations.nrows();
    let n = initial_probs.len();

    // E-step: compute forward-backward
    let (alpha, beta, gamma, log_likelihood) =
        forward_backward(observations, initial_probs, transition_matrix, emissions);

    // Compute xi: P(z_t = i, z_{t+1} = j | observations)
    let mut xi_sum = Array2::zeros((n, n));

    // Emission probabilities
    let mut emission_probs = Array2::zeros((t, n));
    for t_idx in 0..t {
        let obs = observations.row(t_idx).to_owned();
        for j in 0..n {
            emission_probs[[t_idx, j]] = emissions[j].pdf(&obs);
        }
    }

    for t_idx in 0..t - 1 {
        let mut normalizer = 0.0;

        for i in 0..n {
            for j in 0..n {
                let xi_ij = alpha[[t_idx, i]]
                    * transition_matrix[[i, j]]
                    * emission_probs[[t_idx + 1, j]]
                    * beta[[t_idx + 1, j]];
                xi_sum[[i, j]] += xi_ij;
                normalizer += xi_ij;
            }
        }
    }

    // M-step: update parameters

    // New initial probabilities
    let new_initial = gamma.row(0).to_owned();

    // New transition matrix
    let mut new_transition = Array2::zeros((n, n));
    for i in 0..n {
        let gamma_sum: f64 = (0..t - 1).map(|t_idx| gamma[[t_idx, i]]).sum();
        if gamma_sum > 1e-300 {
            for j in 0..n {
                new_transition[[i, j]] = xi_sum[[i, j]] / gamma_sum;
            }
        } else {
            // Keep uniform if no data
            for j in 0..n {
                new_transition[[i, j]] = 1.0 / n as f64;
            }
        }
    }

    // Normalize transition rows
    for i in 0..n {
        let row_sum: f64 = new_transition.row(i).sum();
        if row_sum > 1e-300 {
            for j in 0..n {
                new_transition[[i, j]] /= row_sum;
            }
        }
    }

    (new_initial, new_transition, gamma, log_likelihood)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn create_test_hmm() -> (Array1<f64>, Array2<f64>, Vec<MultivariateGaussian>) {
        // 2-state HMM
        let initial = array![0.6, 0.4];
        let transition = ndarray::arr2(&[[0.7, 0.3], [0.4, 0.6]]);

        let emissions = vec![
            MultivariateGaussian::new(array![0.0], Array2::eye(1)),
            MultivariateGaussian::new(array![3.0], Array2::eye(1)),
        ];

        (initial, transition, emissions)
    }

    #[test]
    fn test_viterbi() {
        let (initial, transition, emissions) = create_test_hmm();
        let obs = ndarray::arr2(&[[0.1], [0.2], [2.8], [3.1]]);

        let (path, log_prob) = viterbi(&obs, &initial, &transition, &emissions);

        assert_eq!(path.len(), 4);
        // Should detect state 0 for first two, state 1 for last two
        assert_eq!(path[0], 0);
        assert_eq!(path[3], 1);
        assert!(log_prob.is_finite());
    }

    #[test]
    fn test_forward_backward() {
        let (initial, transition, emissions) = create_test_hmm();
        let obs = ndarray::arr2(&[[0.1], [0.2], [2.8], [3.1]]);

        let (alpha, beta, gamma, log_ll) =
            forward_backward(&obs, &initial, &transition, &emissions);

        assert_eq!(alpha.nrows(), 4);
        assert_eq!(gamma.nrows(), 4);

        // Gamma rows should sum to 1
        for t in 0..4 {
            let sum: f64 = gamma.row(t).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        assert!(log_ll.is_finite());
    }
}
