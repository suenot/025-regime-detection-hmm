# HMM Trading - Rust Implementation

Rust implementation of Hidden Markov Model (HMM) based regime detection for cryptocurrency trading.

## Features

- **Bybit API Client**: Fetch historical OHLCV data from Bybit exchange
- **Feature Engineering**: Technical indicators for regime detection (returns, volatility, RSI, MACD, etc.)
- **Gaussian HMM**: Full implementation with Viterbi, Forward-Backward, and Baum-Welch algorithms
- **Regime Detection**: Automatic classification into Bull, Bear, and Sideways regimes
- **Trading Strategies**: Regime-specific strategies with position sizing
- **Backtesting**: Complete backtesting framework with performance metrics

## Quick Start

### Installation

```bash
cd rust_hmm_trading
cargo build --release
```

### CLI Usage

```bash
# Fetch data from Bybit
cargo run -- fetch -s BTCUSDT -i 4h -l 1000 -o data.csv

# Train HMM model
cargo run -- train -i data.csv -n 3 --n-iter 100

# Detect current regime
cargo run -- detect -s BTCUSDT -i 4h -n 3

# Run backtest
cargo run -- backtest -s BTCUSDT -i 4h --capital 10000 -n 3
```

### Examples

```bash
# Fetch cryptocurrency data
cargo run --example fetch_data

# Train HMM and analyze regimes
cargo run --example train_hmm

# Real-time regime detection
cargo run --example detect_regimes

# Full backtest comparison
cargo run --example full_backtest
```

## Project Structure

```
rust_hmm_trading/
├── Cargo.toml              # Dependencies
├── README.md               # This file
├── src/
│   ├── lib.rs              # Main library module
│   ├── main.rs             # CLI application
│   ├── api/                # Bybit API client
│   │   ├── mod.rs
│   │   ├── bybit.rs        # API implementation
│   │   └── error.rs        # Error types
│   ├── data/               # Data structures
│   │   ├── mod.rs
│   │   ├── types.rs        # Candle, Dataset
│   │   └── features.rs     # Feature engineering
│   ├── models/             # HMM implementation
│   │   ├── mod.rs
│   │   ├── hmm.rs          # GaussianHMM
│   │   ├── gaussian.rs     # Multivariate Gaussian
│   │   └── algorithms.rs   # Viterbi, Forward-Backward, Baum-Welch
│   ├── regime/             # Regime detection
│   │   ├── mod.rs
│   │   ├── detector.rs     # RegimeDetector
│   │   └── interpreter.rs  # Regime interpretation
│   ├── strategies/         # Trading strategies
│   │   ├── mod.rs
│   │   ├── base.rs         # Strategy trait
│   │   ├── bull.rs         # Bull strategy
│   │   ├── bear.rs         # Bear strategy
│   │   ├── sideways.rs     # Sideways strategy
│   │   └── switcher.rs     # Strategy switcher
│   └── trading/            # Backtesting
│       ├── mod.rs
│       ├── backtest.rs     # Backtest engine
│       ├── metrics.rs      # Performance metrics
│       └── portfolio.rs    # Portfolio management
└── examples/
    ├── fetch_data.rs       # Data fetching example
    ├── train_hmm.rs        # HMM training example
    ├── detect_regimes.rs   # Regime detection example
    └── full_backtest.rs    # Complete backtest example
```

## API Reference

### BybitClient

```rust
use hmm_trading::api::BybitClient;

let client = BybitClient::new();
let candles = client.get_klines("BTCUSDT", "4h", 1000).await?;
```

### GaussianHMM

```rust
use hmm_trading::models::GaussianHMM;
use hmm_trading::data::FeatureBuilder;

// Build features
let features = FeatureBuilder::default().build(&dataset)?;

// Train HMM
let mut hmm = GaussianHMM::new(3); // 3 states
hmm.fit(&features, 100)?;

// Predict states
let states = hmm.predict(&features)?;
let probs = hmm.predict_proba(&features)?;
```

### RegimeDetector

```rust
use hmm_trading::regime::RegimeDetector;

let mut detector = RegimeDetector::new(hmm)
    .with_smoothing(5)
    .with_threshold(0.7);

detector.fit_interpreter(&features, &returns)?;
let current = detector.current_regime(&features)?;

println!("Current regime: {:?}", current.regime);
```

### Backtest

```rust
use hmm_trading::trading::{Backtest, BacktestConfig};

let config = BacktestConfig {
    initial_capital: 10000.0,
    transaction_cost: 0.001,
    ..Default::default()
};

let backtest = Backtest::with_config(config);
let result = backtest.run(&dataset, &features, &hmm)?;

println!("Sharpe: {:.2}", result.metrics.sharpe_ratio);
println!("Max DD: {:.2}%", result.metrics.max_drawdown * 100.0);
```

## Regime Strategies

### Bull Strategy
- Momentum-based approach
- High equity allocation (100%)
- Buy on pullbacks
- Tight stops on reversals

### Bear Strategy
- Defensive positioning
- Low equity allocation (20%)
- Sell rallies
- Cash/bonds preference

### Sideways Strategy
- Mean-reversion based
- Moderate allocation (50%)
- Range trading
- Buy at support, sell at resistance

## Performance Metrics

- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return / Max Drawdown
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

## Dependencies

- `reqwest` - HTTP client for API
- `tokio` - Async runtime
- `ndarray` - Linear algebra
- `statrs` - Statistical functions
- `serde` - Serialization
- `clap` - CLI parsing
- `tracing` - Logging

## License

MIT
