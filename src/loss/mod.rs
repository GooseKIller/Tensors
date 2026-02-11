//! # Loss functions
//!
//! for evaluating model performance
//!
//! [SSE] - Sum of squared errors
//!
//! [MSE] - Mean square error
//!
//! [MAPE] - Mean absolute percentage error

mod crossentropy;
mod mape;
mod mse;

mod binarycrossentropy;
mod sse;

pub use binarycrossentropy::*;
pub use crossentropy::*;
pub use mape::*;
pub use mse::*;
pub use sse::*;
