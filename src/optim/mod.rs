//! # Optimization algorithms
//!
//! for training neural networks
//!
//! [Adam] - Adaptive learning rate optimization algorithm for training neural networks
//!
//! [SGD] - Stochastic Gradient Descent optimization algorithm for training neural network

mod adam;
mod sgd;
mod rmsprop;

pub use adam::*;
pub use sgd::*;
pub use rmsprop::*;

use crate::linalg::Matrix;
use crate::Float;

pub trait Optimizer<T: Float> {
    fn step(&mut self, id: usize, weights: &mut Matrix<T>, gradients: &Matrix<T>);
    fn change_learning_rate(&mut self, new_learning_rate: T);
}
