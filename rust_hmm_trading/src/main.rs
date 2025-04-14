//! HMM Trading CLI
//!
//! Command-line interface for regime detection and backtesting

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;
use hmm_trading::{
    api::BybitClient,
    data::{Dataset, FeatureBuilder},
    models::GaussianHMM,
    regime::RegimeDetector,
    trading::{Backtest, BacktestConfig},
};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "hmm_trading")]
#[command(about = "HMM-based regime detection for cryptocurrency trading")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch market data from Bybit
    Fetch {
        /// Trading symbol (e.g., BTCUSDT)
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Interval (e.g., 1h, 4h, 1d)
        #[arg(short, long, default_value = "4h")]
        interval: String,

        /// Number of candles to fetch
        #[arg(short, long, default_value = "1000")]
        limit: usize,

        /// Output file path
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Train HMM model on data
    Train {
        /// Input CSV file
        #[arg(short, long)]
        input: String,

        /// Number of states (regimes)
        #[arg(short = 'n', long, default_value = "3")]
        n_states: usize,

        /// Maximum training iterations
        #[arg(long, default_value = "100")]
        n_iter: usize,
    },

    /// Detect current regime
    Detect {
        /// Trading symbol
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Interval
        #[arg(short, long, default_value = "4h")]
        interval: String,

        /// Number of states
        #[arg(short = 'n', long, default_value = "3")]
        n_states: usize,
    },

    /// Run backtest
    Backtest {
        /// Trading symbol
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Interval
        #[arg(short, long, default_value = "4h")]
        interval: String,

        /// Initial capital
        #[arg(long, default_value = "10000")]
        capital: f64,

        /// Number of states
        #[arg(short = 'n', long, default_value = "3")]
        n_states: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("hmm_trading=info".parse()?))
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Fetch {
            symbol,
            interval,
            limit,
            output,
        } => {
            fetch_data(&symbol, &interval, limit, output.as_deref()).await?;
        }
        Commands::Train {
            input,
            n_states,
            n_iter,
        } => {
            train_model(&input, n_states, n_iter)?;
        }
        Commands::Detect {
            symbol,
            interval,
            n_states,
        } => {
            detect_regime(&symbol, &interval, n_states).await?;
        }
        Commands::Backtest {
            symbol,
            interval,
            capital,
            n_states,
        } => {
            run_backtest(&symbol, &interval, capital, n_states).await?;
        }
    }

    Ok(())
}

async fn fetch_data(symbol: &str, interval: &str, limit: usize, output: Option<&str>) -> Result<()> {
    println!("{}", "Fetching data from Bybit...".cyan());

    let client = BybitClient::new();
    let candles = client.get_klines(symbol, interval, limit).await?;

    println!(
        "{}",
        format!("Fetched {} candles for {}", candles.len(), symbol).green()
    );

    if let Some(path) = output {
        let dataset = Dataset::new(candles.clone(), symbol, interval);
        dataset.to_csv(path)?;
        println!("{}", format!("Saved to {}", path).green());
    }

    // Show last few candles
    println!("\nLast 5 candles:");
    for candle in candles.iter().rev().take(5).rev() {
        let change = (candle.close - candle.open) / candle.open * 100.0;
        let change_str = if change >= 0.0 {
            format!("+{:.2}%", change).green()
        } else {
            format!("{:.2}%", change).red()
        };
        println!(
            "  {} | O: {:.2} H: {:.2} L: {:.2} C: {:.2} | {}",
            candle.timestamp, candle.open, candle.high, candle.low, candle.close, change_str
        );
    }

    Ok(())
}

fn train_model(input: &str, n_states: usize, n_iter: usize) -> Result<()> {
    println!("{}", "Loading data...".cyan());

    let dataset = Dataset::from_csv(input, "UNKNOWN", "unknown")?;
    println!("Loaded {} candles", dataset.len());

    println!("{}", "Building features...".cyan());
    let features = FeatureBuilder::default().build(&dataset)?;
    println!(
        "Built {} features for {} samples",
        features.n_features(),
        features.n_samples()
    );

    println!(
        "{}",
        format!("Training {}-state HMM (max {} iterations)...", n_states, n_iter).cyan()
    );

    let mut hmm = GaussianHMM::new(n_states);
    let log_ll = hmm.fit(&features, n_iter)?;

    println!(
        "{}",
        format!("Training complete! Log-likelihood: {:.4}", log_ll).green()
    );

    // Show transition matrix
    println!("\nTransition Matrix:");
    let trans = hmm.transition_matrix();
    for i in 0..n_states {
        print!("  State {}: ", i);
        for j in 0..n_states {
            print!("{:.2}  ", trans[[i, j]]);
        }
        println!();
    }

    Ok(())
}

async fn detect_regime(symbol: &str, interval: &str, n_states: usize) -> Result<()> {
    println!("{}", "Fetching recent data...".cyan());

    let client = BybitClient::new();
    let candles = client.get_klines(symbol, interval, 500).await?;
    let dataset = Dataset::new(candles, symbol, interval);

    println!("{}", "Building features...".cyan());
    let features = FeatureBuilder::default().build(&dataset)?;

    println!("{}", "Training HMM...".cyan());
    let mut hmm = GaussianHMM::new(n_states);
    hmm.fit(&features, 100)?;

    println!("{}", "Detecting regimes...".cyan());
    let returns = dataset.log_returns();
    let mut detector = RegimeDetector::new(hmm);
    detector.fit_interpreter(&features, &returns)?;

    let current = detector.current_regime(&features)?;

    println!("\n{}", "=== Current Regime ===".bold());
    println!(
        "  {} {}",
        current.regime.emoji(),
        current.regime.to_string().bold()
    );
    println!("  Probability: {:.1}%", current.probability * 100.0);
    println!(
        "  Confident: {}",
        if current.is_confident { "Yes" } else { "No" }
    );

    // Show regime statistics
    if let Some(interpreter) = detector.interpreter() {
        interpreter.print_summary();
    }

    Ok(())
}

async fn run_backtest(symbol: &str, interval: &str, capital: f64, n_states: usize) -> Result<()> {
    println!("{}", "Fetching historical data...".cyan());

    let client = BybitClient::new();
    let candles = client.get_klines(symbol, interval, 1000).await?;
    let dataset = Dataset::new(candles, symbol, interval);

    println!("{}", "Building features...".cyan());
    let features = FeatureBuilder::default().build(&dataset)?;

    println!("{}", "Training HMM...".cyan());
    let mut hmm = GaussianHMM::new(n_states);
    hmm.fit(&features, 100)?;

    println!("{}", "Running backtest...".cyan());
    let config = BacktestConfig {
        initial_capital: capital,
        ..Default::default()
    };

    let backtest = Backtest::with_config(config.clone());

    // Run HMM strategy
    let hmm_result = backtest.run(&dataset, &features, &hmm)?;

    // Run buy-and-hold benchmark
    let bh_result = backtest.run_buy_hold(&dataset)?;

    println!("\n{}", "=== HMM Strategy Results ===".bold().green());
    hmm_result.print_summary();

    println!("\n{}", "=== Buy & Hold Benchmark ===".bold().yellow());
    bh_result.print_summary();

    // Comparison
    println!("\n{}", "=== Comparison ===".bold());
    let alpha = hmm_result.metrics.total_return - bh_result.metrics.total_return;
    let alpha_str = if alpha >= 0.0 {
        format!("+{:.2}%", alpha * 100.0).green()
    } else {
        format!("{:.2}%", alpha * 100.0).red()
    };
    println!("  Alpha vs Buy&Hold: {}", alpha_str);
    println!(
        "  Sharpe Improvement: {:.2}",
        hmm_result.metrics.sharpe_ratio - bh_result.metrics.sharpe_ratio
    );
    println!(
        "  Drawdown Improvement: {:.2}%",
        (bh_result.metrics.max_drawdown - hmm_result.metrics.max_drawdown) * 100.0
    );

    Ok(())
}
