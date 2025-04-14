//! Example: Fetch cryptocurrency data from Bybit
//!
//! This example demonstrates how to use the Bybit API client
//! to fetch historical OHLCV data.
//!
//! Run with: cargo run --example fetch_data

use hmm_trading::api::BybitClient;
use hmm_trading::data::Dataset;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Bybit Data Fetcher ===\n");

    // Create client
    let client = BybitClient::new();

    // Fetch BTC/USDT 4-hour candles
    let symbol = "BTCUSDT";
    let interval = "4h";
    let limit = 500;

    println!("Fetching {} {} candles for {}...", limit, interval, symbol);
    let candles = client.get_klines(symbol, interval, limit).await?;

    println!("Fetched {} candles\n", candles.len());

    // Create dataset
    let dataset = Dataset::new(candles, symbol, interval);

    // Show basic statistics
    let closes = dataset.closes();
    let returns = dataset.log_returns();

    let min_price = closes.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_price = closes.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let avg_price = closes.iter().sum::<f64>() / closes.len() as f64;

    let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let volatility = (returns.iter().map(|r| (r - avg_return).powi(2)).sum::<f64>()
        / returns.len() as f64)
        .sqrt();

    println!("=== Price Statistics ===");
    println!("  Min:  ${:.2}", min_price);
    println!("  Max:  ${:.2}", max_price);
    println!("  Avg:  ${:.2}", avg_price);
    println!("  Current: ${:.2}", closes.last().unwrap_or(&0.0));

    println!("\n=== Return Statistics ===");
    println!("  Avg Return:  {:.4}%", avg_return * 100.0);
    println!("  Volatility:  {:.4}%", volatility * 100.0);
    println!(
        "  Annualized Vol: {:.2}%",
        volatility * (2190.0_f64).sqrt() * 100.0
    );

    // Show last 10 candles
    println!("\n=== Last 10 Candles ===");
    for candle in dataset.candles.iter().rev().take(10).rev() {
        let change_pct = (candle.close - candle.open) / candle.open * 100.0;
        let arrow = if change_pct >= 0.0 { "↑" } else { "↓" };
        println!(
            "  {} | O: ${:.2} | H: ${:.2} | L: ${:.2} | C: ${:.2} | {} {:.2}%",
            candle.timestamp, candle.open, candle.high, candle.low, candle.close, arrow, change_pct
        );
    }

    // Save to CSV
    let output_path = "btcusdt_4h.csv";
    dataset.to_csv(output_path)?;
    println!("\nData saved to {}", output_path);

    Ok(())
}
