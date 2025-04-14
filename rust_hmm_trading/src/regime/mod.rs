//! Regime detection module
//!
//! Provides regime detection and interpretation based on HMM.

mod detector;
mod interpreter;

pub use detector::RegimeDetector;
pub use interpreter::{Regime, RegimeInfo, RegimeInterpreter};
