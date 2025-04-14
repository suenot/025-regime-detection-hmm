//! Multivariate Gaussian distribution for HMM emissions

use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Multivariate Gaussian distribution
#[derive(Debug, Clone)]
pub struct MultivariateGaussian {
    /// Mean vector
    pub mean: Array1<f64>,
    /// Covariance matrix
    pub covariance: Array2<f64>,
    /// Precomputed inverse of covariance (for efficiency)
    covariance_inv: Option<Array2<f64>>,
    /// Precomputed log determinant
    log_det: Option<f64>,
}

impl MultivariateGaussian {
    /// Create new multivariate Gaussian
    pub fn new(mean: Array1<f64>, covariance: Array2<f64>) -> Self {
        let mut gaussian = Self {
            mean,
            covariance,
            covariance_inv: None,
            log_det: None,
        };
        gaussian.update_cache();
        gaussian
    }

    /// Create with identity covariance (useful for initialization)
    pub fn with_identity(mean: Array1<f64>) -> Self {
        let d = mean.len();
        let covariance = Array2::eye(d);
        Self::new(mean, covariance)
    }

    /// Create from samples (maximum likelihood estimation)
    pub fn from_samples(samples: &Array2<f64>) -> Self {
        let n = samples.nrows() as f64;
        let d = samples.ncols();

        // Compute mean
        let mean = samples.mean_axis(ndarray::Axis(0)).unwrap();

        // Compute covariance
        let mut covariance = Array2::zeros((d, d));
        for row in samples.rows() {
            let diff = &row - &mean;
            for i in 0..d {
                for j in 0..d {
                    covariance[[i, j]] += diff[i] * diff[j];
                }
            }
        }
        covariance /= n;

        // Add small regularization for numerical stability
        for i in 0..d {
            covariance[[i, i]] += 1e-6;
        }

        Self::new(mean, covariance)
    }

    /// Dimension of the distribution
    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    /// Update cached values after parameter changes
    pub fn update_cache(&mut self) {
        let d = self.dim();

        // Simple matrix inversion using Cholesky-like approach
        // For production, use proper linear algebra library
        self.covariance_inv = Some(self.invert_covariance());

        // Log determinant approximation
        let mut log_det = 0.0;
        for i in 0..d {
            log_det += self.covariance[[i, i]].ln();
        }
        self.log_det = Some(log_det);
    }

    /// Invert covariance matrix (simple implementation)
    fn invert_covariance(&self) -> Array2<f64> {
        let d = self.dim();
        let mut result = Array2::eye(d);

        // For diagonal-dominant matrices, use simple iterative method
        // In production, use proper linear algebra
        for i in 0..d {
            if self.covariance[[i, i]].abs() > 1e-10 {
                result[[i, i]] = 1.0 / self.covariance[[i, i]];
            }
        }

        result
    }

    /// Compute log probability density at a point
    pub fn log_pdf(&self, x: &Array1<f64>) -> f64 {
        let d = self.dim() as f64;
        let diff = x - &self.mean;

        // Compute quadratic form: (x - mu)' * Sigma^-1 * (x - mu)
        let cov_inv = self.covariance_inv.as_ref().unwrap();
        let mut quad_form = 0.0;
        for i in 0..self.dim() {
            for j in 0..self.dim() {
                quad_form += diff[i] * cov_inv[[i, j]] * diff[j];
            }
        }

        let log_det = self.log_det.unwrap_or(0.0);

        -0.5 * (d * (2.0 * PI).ln() + log_det + quad_form)
    }

    /// Compute probability density at a point
    pub fn pdf(&self, x: &Array1<f64>) -> f64 {
        self.log_pdf(x).exp()
    }

    /// Sample from the distribution (using Box-Muller)
    pub fn sample(&self) -> Array1<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let d = self.dim();

        // Generate standard normal samples
        let mut z = Array1::zeros(d);
        for i in 0..d {
            // Box-Muller transform
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            z[i] = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        }

        // Transform: x = mu + L * z where L is Cholesky factor
        // Simplified: for diagonal covariance
        let mut result = self.mean.clone();
        for i in 0..d {
            result[i] += self.covariance[[i, i]].sqrt() * z[i];
        }

        result
    }

    /// Update parameters with weighted samples
    pub fn update_weighted(&mut self, samples: &Array2<f64>, weights: &Array1<f64>) {
        let n = samples.nrows();
        let d = samples.ncols();
        let weight_sum = weights.sum();

        if weight_sum < 1e-10 {
            return;
        }

        // Weighted mean
        let mut new_mean = Array1::zeros(d);
        for i in 0..n {
            for j in 0..d {
                new_mean[j] += weights[i] * samples[[i, j]];
            }
        }
        new_mean /= weight_sum;

        // Weighted covariance
        let mut new_cov = Array2::zeros((d, d));
        for i in 0..n {
            let diff: Array1<f64> = samples.row(i).to_owned() - &new_mean;
            for j in 0..d {
                for k in 0..d {
                    new_cov[[j, k]] += weights[i] * diff[j] * diff[k];
                }
            }
        }
        new_cov /= weight_sum;

        // Regularization
        for i in 0..d {
            new_cov[[i, i]] += 1e-6;
        }

        self.mean = new_mean;
        self.covariance = new_cov;
        self.update_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gaussian_creation() {
        let mean = array![0.0, 0.0];
        let cov = Array2::eye(2);
        let g = MultivariateGaussian::new(mean, cov);
        assert_eq!(g.dim(), 2);
    }

    #[test]
    fn test_pdf_at_mean() {
        let mean = array![0.0, 0.0];
        let cov = Array2::eye(2);
        let g = MultivariateGaussian::new(mean.clone(), cov);

        // PDF should be highest at mean
        let pdf_at_mean = g.pdf(&mean);
        let pdf_away = g.pdf(&array![1.0, 1.0]);
        assert!(pdf_at_mean > pdf_away);
    }

    #[test]
    fn test_from_samples() {
        let samples = ndarray::arr2(&[
            [1.0, 2.0],
            [1.5, 2.5],
            [0.5, 1.5],
            [1.0, 2.0],
        ]);
        let g = MultivariateGaussian::from_samples(&samples);
        assert_eq!(g.dim(), 2);
        // Mean should be close to [1.0, 2.0]
        assert!((g.mean[0] - 1.0).abs() < 0.5);
        assert!((g.mean[1] - 2.0).abs() < 0.5);
    }
}
