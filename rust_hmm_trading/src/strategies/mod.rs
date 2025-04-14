//! Trading strategies module
//!
//! Provides regime-specific trading strategies and a strategy switcher.

mod base;
mod bull;
mod bear;
mod sideways;
mod switcher;

pub use base::{Strategy, StrategyType, Signal, Position};
pub use bull::BullStrategy;
pub use bear::BearStrategy;
pub use sideways::SidewaysStrategy;
pub use switcher::StrategySwitcher;
