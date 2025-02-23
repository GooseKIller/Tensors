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
    fn step(&mut self, id:usize, weights: &mut Matrix<T>, gradients: &Matrix<T>);
    fn change_learning_rate(&mut self, new_learning_rate: T);
}
