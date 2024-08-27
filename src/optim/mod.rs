//! # Optimization algorithms
//!
//! for training neural networks
//!
//! [Adam] - Adaptive learning rate optimization algorithm for training neural networks
//!
//! [SGD] - Stochastic Gradient Descent optimization algorithm for training neural network

mod sgd;
mod adam;

pub use sgd::*;
pub use adam::*;

use crate::Float;
use crate::linalg::Matrix;

pub trait Optimizer<T: Float> {
    fn step(&mut self, weights: &mut Matrix<T>, gradients: &Matrix<T>);
}
