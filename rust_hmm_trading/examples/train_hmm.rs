//! Example: Train HMM for regime detection
//!
//! This example demonstrates how to train a Gaussian HMM
//! for market regime detection using cryptocurrency data.
//!
//! Run with: cargo run --example train_hmm

use hmm_trading::api::BybitClient;
use hmm_trading::data::{Dataset, FeatureBuilder};
use hmm_trading::models::GaussianHMM;
use hmm_trading::regime::RegimeInterpreter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== HMM Training Example ===\n");

    // Fetch data
    println!("Fetching data from Bybit...");
    let client = BybitClient::new();
    let candles = client.get_klines("BTCUSDT", "4h", 1000).await?;

    let dataset = Dataset::new(candles, "BTCUSDT", "4h");
    println!("Loaded {} candles\n", dataset.len());

    // Build features
    println!("Building features...");
    let feature_builder = FeatureBuilder::new()
        .with_return_window(20)
        .with_volatility_window(20)
        .with_rsi_period(14);

    let features = feature_builder.build(&dataset)?;
    println!(
        "Built {} features for {} observations\n",
        features.n_features(),
        features.n_samples()
    );

    // Print feature names
    println!("Features:");
    for name in &features.names {
        println!("  - {}", name);
    }
    println!();

    // Train HMM with 3 states (Bull, Bear, Sideways)
    println!("Training 3-state Gaussian HMM...");
    let n_states = 3;
    let n_iter = 100;

    let mut hmm = GaussianHMM::new(n_states).with_tol(1e-4);
    let log_likelihood = hmm.fit(&features, n_iter)?;

    println!("\nTraining complete!");
    println!("  Final log-likelihood: {:.4}", log_likelihood);
    println!(
        "  Iterations: {}",
        hmm.log_likelihood_history.len()
    );

    // Show transition matrix
    println!("\n=== Transition Matrix ===");
    let trans = hmm.transition_matrix();
    println!("         State 0   State 1   State 2");
    for i in 0..n_states {
        print!("State {} ", i);
        for j in 0..n_states {
            print!("  {:.4}  ", trans[[i, j]]);
        }
        println!();
    }

    // Interpret regimes
    println!("\n=== Regime Interpretation ===");
    let returns = dataset.log_returns();
    let interpreter = RegimeInterpreter::from_hmm(&hmm, &features, &returns)?;

    interpreter.print_summary();

    // Show regime sequence for last 20 periods
    println!("\n=== Recent Regime Sequence ===");
    let states = hmm.predict(&features)?;
    let probs = hmm.predict_proba(&features)?;

    let start = states.len().saturating_sub(20);
    for i in start..states.len() {
        let state = states[i];
        let regime = interpreter.get_regime(state);
        let prob = probs[[i, state]];

        let bar_len = (prob * 20.0) as usize;
        let bar: String = "█".repeat(bar_len) + &"░".repeat(20 - bar_len);

        println!(
            "  {} {} {:?} ({:.1}%) [{}]",
            regime.emoji(),
            if i == states.len() - 1 {
                "→"
            } else {
                " "
            },
            regime,
            prob * 100.0,
            bar
        );
    }

    // Calculate AIC and BIC for model selection
    println!("\n=== Model Selection Criteria ===");
    let aic = hmm.aic(&features)?;
    let bic = hmm.bic(&features)?;
    println!("  AIC: {:.2}", aic);
    println!("  BIC: {:.2}", bic);
    println!(
        "  (Lower is better - use to compare different n_states)"
    );

    Ok(())
}
