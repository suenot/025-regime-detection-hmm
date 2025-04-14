//! Example: Full backtest with regime-switching strategy
//!
//! This example demonstrates a complete backtest comparing
//! the HMM regime-switching strategy against buy-and-hold.
//!
//! Run with: cargo run --example full_backtest

use hmm_trading::api::BybitClient;
use hmm_trading::data::{Dataset, FeatureBuilder};
use hmm_trading::models::GaussianHMM;
use hmm_trading::trading::{Backtest, BacktestConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║        HMM Regime-Switching Strategy Backtest            ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Configuration
    let symbol = "BTCUSDT";
    let interval = "4h";
    let initial_capital = 10000.0;
    let n_states = 3;

    println!("Configuration:");
    println!("  Symbol:          {}", symbol);
    println!("  Interval:        {}", interval);
    println!("  Initial Capital: ${:.2}", initial_capital);
    println!("  HMM States:      {}", n_states);
    println!();

    // Fetch data
    println!("Fetching historical data...");
    let client = BybitClient::new();
    let candles = client.get_klines(symbol, interval, 1000).await?;

    let dataset = Dataset::new(candles, symbol, interval);
    println!("  Loaded {} candles", dataset.len());

    // Build features
    println!("Building features...");
    let features = FeatureBuilder::default().build(&dataset)?;
    println!(
        "  {} features × {} samples",
        features.n_features(),
        features.n_samples()
    );

    // Train HMM
    println!("Training HMM...");
    let mut hmm = GaussianHMM::new(n_states);
    let log_ll = hmm.fit(&features, 100)?;
    println!("  Log-likelihood: {:.4}", log_ll);

    // Configure backtest
    let config = BacktestConfig {
        initial_capital,
        transaction_cost: 0.001, // 0.1%
        annualization: 2190.0,   // 4h candles per year
        risk_free_rate: 0.04,    // 4%
        max_position: 1.0,
    };

    let backtest = Backtest::with_config(config);

    // Run HMM strategy
    println!("\nRunning HMM strategy backtest...");
    let hmm_result = backtest.run(&dataset, &features, &hmm)?;

    // Run buy-and-hold
    println!("Running buy-and-hold benchmark...");
    let bh_result = backtest.run_buy_hold(&dataset)?;

    // Print results
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║                    BACKTEST RESULTS                       ║");
    println!("╠══════════════════════════════════════════════════════════╣");

    // HMM Strategy Results
    println!("║ HMM REGIME-SWITCHING STRATEGY                            ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║  Total Return:       {:>10.2}%                          ║",
        hmm_result.metrics.total_return * 100.0
    );
    println!(
        "║  Annual Return:      {:>10.2}%                          ║",
        hmm_result.metrics.annual_return * 100.0
    );
    println!(
        "║  Annual Volatility:  {:>10.2}%                          ║",
        hmm_result.metrics.annual_volatility * 100.0
    );
    println!(
        "║  Sharpe Ratio:       {:>10.2}                           ║",
        hmm_result.metrics.sharpe_ratio
    );
    println!(
        "║  Sortino Ratio:      {:>10.2}                           ║",
        hmm_result.metrics.sortino_ratio
    );
    println!(
        "║  Max Drawdown:       {:>10.2}%                          ║",
        hmm_result.metrics.max_drawdown * 100.0
    );
    println!(
        "║  Calmar Ratio:       {:>10.2}                           ║",
        hmm_result.metrics.calmar_ratio
    );
    println!(
        "║  Win Rate:           {:>10.1}%                          ║",
        hmm_result.metrics.win_rate * 100.0
    );
    println!(
        "║  Number of Trades:   {:>10}                            ║",
        hmm_result.metrics.num_trades
    );
    println!(
        "║  Strategy Switches:  {:>10}                            ║",
        hmm_result.num_switches
    );

    // Buy-and-Hold Results
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ BUY-AND-HOLD BENCHMARK                                   ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║  Total Return:       {:>10.2}%                          ║",
        bh_result.metrics.total_return * 100.0
    );
    println!(
        "║  Annual Return:      {:>10.2}%                          ║",
        bh_result.metrics.annual_return * 100.0
    );
    println!(
        "║  Annual Volatility:  {:>10.2}%                          ║",
        bh_result.metrics.annual_volatility * 100.0
    );
    println!(
        "║  Sharpe Ratio:       {:>10.2}                           ║",
        bh_result.metrics.sharpe_ratio
    );
    println!(
        "║  Max Drawdown:       {:>10.2}%                          ║",
        bh_result.metrics.max_drawdown * 100.0
    );

    // Comparison
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ COMPARISON                                               ║");
    println!("╠══════════════════════════════════════════════════════════╣");

    let alpha = hmm_result.metrics.total_return - bh_result.metrics.total_return;
    let sharpe_diff = hmm_result.metrics.sharpe_ratio - bh_result.metrics.sharpe_ratio;
    let dd_diff = bh_result.metrics.max_drawdown - hmm_result.metrics.max_drawdown;

    let alpha_str = if alpha >= 0.0 {
        format!("+{:.2}%", alpha * 100.0)
    } else {
        format!("{:.2}%", alpha * 100.0)
    };
    let sharpe_str = if sharpe_diff >= 0.0 {
        format!("+{:.2}", sharpe_diff)
    } else {
        format!("{:.2}", sharpe_diff)
    };
    let dd_str = if dd_diff >= 0.0 {
        format!("+{:.2}%", dd_diff * 100.0)
    } else {
        format!("{:.2}%", dd_diff * 100.0)
    };

    println!("║  Alpha vs B&H:       {:>10}                          ║", alpha_str);
    println!(
        "║  Sharpe Improvement: {:>10}                          ║",
        sharpe_str
    );
    println!(
        "║  Drawdown Reduction: {:>10}                          ║",
        dd_str
    );

    println!("╚══════════════════════════════════════════════════════════╝");

    // Regime distribution
    println!("\n=== Regime Distribution ===\n");
    let mut regime_counts = [0usize; 4]; // Bull, Bear, Sideways, Unknown
    for regime in &hmm_result.regimes {
        match regime {
            hmm_trading::regime::Regime::Bull => regime_counts[0] += 1,
            hmm_trading::regime::Regime::Bear => regime_counts[1] += 1,
            hmm_trading::regime::Regime::Sideways => regime_counts[2] += 1,
            hmm_trading::regime::Regime::Unknown(_) => regime_counts[3] += 1,
        }
    }

    let total = hmm_result.regimes.len() as f64;
    println!(
        "  Bull:     {:4} periods ({:5.1}%)",
        regime_counts[0],
        regime_counts[0] as f64 / total * 100.0
    );
    println!(
        "  Bear:     {:4} periods ({:5.1}%)",
        regime_counts[1],
        regime_counts[1] as f64 / total * 100.0
    );
    println!(
        "  Sideways: {:4} periods ({:5.1}%)",
        regime_counts[2],
        regime_counts[2] as f64 / total * 100.0
    );

    // Equity curve summary
    println!("\n=== Equity Curve Summary ===\n");
    let hmm_final = hmm_result.equity_curve.last().unwrap_or(&initial_capital);
    let bh_final = bh_result.equity_curve.last().unwrap_or(&initial_capital);

    println!("  HMM Final Equity:   ${:.2}", hmm_final);
    println!("  B&H Final Equity:   ${:.2}", bh_final);
    println!(
        "  Difference:         ${:.2}",
        hmm_final - bh_final
    );

    println!("\n=== Conclusion ===\n");
    if hmm_result.metrics.sharpe_ratio > bh_result.metrics.sharpe_ratio {
        println!("  ✓ HMM strategy achieved better risk-adjusted returns");
    } else {
        println!("  ✗ Buy-and-hold had better risk-adjusted returns in this period");
    }

    if hmm_result.metrics.max_drawdown < bh_result.metrics.max_drawdown {
        println!("  ✓ HMM strategy reduced maximum drawdown");
    } else {
        println!("  ✗ Buy-and-hold had lower drawdown in this period");
    }

    Ok(())
}
