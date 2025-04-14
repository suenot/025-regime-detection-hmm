//! Example: Real-time regime detection
//!
//! This example demonstrates how to detect the current market regime
//! and make trading decisions based on it.
//!
//! Run with: cargo run --example detect_regimes

use hmm_trading::api::BybitClient;
use hmm_trading::data::{Dataset, FeatureBuilder};
use hmm_trading::models::GaussianHMM;
use hmm_trading::regime::{Regime, RegimeDetector, RegimeInterpreter};
use hmm_trading::strategies::{Strategy, StrategySwitcher};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Real-time Regime Detection ===\n");

    // Symbols to analyze
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let interval = "4h";

    let client = BybitClient::new();

    for symbol in symbols {
        println!("Analyzing {}...", symbol);

        // Fetch data
        let candles = client.get_klines(symbol, interval, 500).await?;
        let dataset = Dataset::new(candles, symbol, interval);

        // Build features
        let features = FeatureBuilder::default().build(&dataset)?;

        // Train HMM
        let mut hmm = GaussianHMM::new(3);
        hmm.fit(&features, 100)?;

        // Create detector with interpreter
        let returns = dataset.log_returns();
        let mut detector = RegimeDetector::new(hmm)
            .with_smoothing(5)
            .with_threshold(0.7)
            .with_min_duration(5);

        detector.fit_interpreter(&features, &returns)?;

        // Detect current regime
        let current = detector.current_regime(&features)?;

        // Get recommended action from strategy
        let mut switcher = StrategySwitcher::new();
        switcher.update_regime(&current, 0);

        let last_candle = dataset.candles.last().unwrap();
        let history = &dataset.candles[dataset.candles.len().saturating_sub(50)..];
        let target_position = switcher.target_position(last_candle, history);

        // Print results
        println!("\n  {} {}", current.regime.emoji(), symbol);
        println!("  ─────────────────────────────");
        println!("  Regime:     {:?}", current.regime);
        println!("  Confidence: {:.1}%", current.probability * 100.0);
        println!("  Strategy:   {:?}", switcher.current_strategy_type());
        println!("  Position:   {:.0}%", target_position * 100.0);

        // Show probability distribution
        println!("  Probabilities:");
        if let Some(interpreter) = detector.interpreter() {
            for (i, &prob) in current.all_probabilities.iter().enumerate() {
                let regime = interpreter.get_regime(i);
                let bar_len = (prob * 20.0) as usize;
                let bar: String = "█".repeat(bar_len);
                println!("    {:8} {:5.1}% {}", format!("{:?}", regime), prob * 100.0, bar);
            }
        }

        // Trading recommendation
        println!("  Recommendation:");
        match current.regime {
            Regime::Bull => {
                println!("    → Stay long / accumulate on dips");
                println!("    → Use momentum strategies");
            }
            Regime::Bear => {
                println!("    → Reduce exposure / stay cash");
                println!("    → Sell rallies");
            }
            Regime::Sideways => {
                println!("    → Trade ranges / mean-reversion");
                println!("    → Wait for breakout confirmation");
            }
            Regime::Unknown(_) => {
                println!("    → Unclear regime, stay cautious");
            }
        }

        println!();
    }

    // Show transition probabilities for BTC
    println!("=== BTC Transition Probabilities ===\n");

    let btc_candles = client.get_klines("BTCUSDT", interval, 500).await?;
    let btc_dataset = Dataset::new(btc_candles, "BTCUSDT", interval);
    let btc_features = FeatureBuilder::default().build(&btc_dataset)?;

    let mut btc_hmm = GaussianHMM::new(3);
    btc_hmm.fit(&btc_features, 100)?;

    let returns = btc_dataset.log_returns();
    let interpreter = RegimeInterpreter::from_hmm(&btc_hmm, &btc_features, &returns)?;

    let trans = btc_hmm.transition_matrix();
    println!("From \\ To    Bull     Bear   Sideways");
    for i in 0..3 {
        let from_regime = interpreter.get_regime(i);
        print!("{:8}  ", format!("{:?}", from_regime));
        for j in 0..3 {
            print!(" {:5.1}% ", trans[[i, j]] * 100.0);
        }
        println!();
    }

    println!("\nInterpretation:");
    println!("  - Diagonal values show regime 'stickiness'");
    println!("  - Higher diagonal = longer average regime duration");
    println!("  - Off-diagonal values show transition probabilities");

    Ok(())
}
