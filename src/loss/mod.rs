//! # Loss functions
//!
//! for evaluating model performance
//!
//! [MSE] - Mean square error
//!
//! [MAPE] - Mean absolute percentage error

mod mse;
mod mape;

pub use mse::*;
pub use mape::*;
use crate::Float;
use crate::linalg::Matrix;

pub trait Loss<T: Float> {
    fn call(&self, output: &Matrix<T>, target: &Matrix<T>) -> T;
    fn gradient(&self, output: &Matrix<T>, target: &Matrix<T>) -> Matrix<T>;
}