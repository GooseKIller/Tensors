//! # Loss functions
//!
//! for evaluating model performance
//!
//! [SSE] - Sum of squared errors
//!
//! [MSE] - Mean square error
//!
//! [MAPE] - Mean absolute percentage error

mod mse;
mod mape;
mod crossentropy;

mod sse;

pub use mse::*;
pub use mape::*;
pub use sse::*;
pub use crossentropy::*;
use crate::Float;
use crate::linalg::Matrix;

pub trait Loss<T: Float> {
    fn call(&self, output: &Matrix<T>, target: &Matrix<T>) -> T;
    fn gradient(&self, output: &Matrix<T>, target: &Matrix<T>) -> Matrix<T>;
}