//! HMM models module
//!
//! Provides Gaussian HMM implementation with Viterbi, Forward-Backward,
//! and Baum-Welch algorithms.

mod hmm;
mod gaussian;
mod algorithms;

pub use hmm::{GaussianHMM, HMMParams};
pub use gaussian::MultivariateGaussian;
pub use algorithms::{viterbi, forward_backward, baum_welch_step};
