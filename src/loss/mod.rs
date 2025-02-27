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

use crate::linalg::Matrix;
use crate::Float;
pub use binarycrossentropy::*;
pub use crossentropy::*;
pub use mape::*;
pub use mse::*;
pub use sse::*;

pub trait Loss<T: Float> {
    fn call(&self, output: &Matrix<T>, target: &Matrix<T>) -> T;
    fn gradient(&self, output: &Matrix<T>, target: &Matrix<T>) -> Matrix<T>;
}
