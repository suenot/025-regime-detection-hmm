//! Bybit API client module
//!
//! Provides async methods to fetch market data from Bybit exchange.

mod bybit;
mod error;

pub use bybit::{BybitClient, Interval};
pub use error::{ApiError, ApiResult};
