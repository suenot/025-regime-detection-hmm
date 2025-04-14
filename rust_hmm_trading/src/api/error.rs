//! API error types

use thiserror::Error;

/// API error types
#[derive(Error, Debug)]
pub enum ApiError {
    #[error("HTTP request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    #[error("Invalid interval: {0}")]
    InvalidInterval(String),

    #[error("API response error: code={code}, message={message}")]
    ApiResponseError { code: i32, message: String },

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Network timeout")]
    Timeout,
}

/// Result type for API operations
pub type ApiResult<T> = Result<T, ApiError>;
