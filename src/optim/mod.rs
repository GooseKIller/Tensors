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

use crate::Float;

pub trait Optimizer<T: Float> {
    fn step(&mut self);
    fn zero_grad(&self);
    
    /// Clip gradients by global norm.
    /// Returns the total gradient norm before clipping.
    fn clip_grad(&self, max_norm: T) -> T {
        use crate::utils::clip_grad as _clip_grad;
        _clip_grad(&self.params(), max_norm)
    }
    
    /// Get the parameters being optimized
    fn params(&self) -> Vec<crate::autodiff::VarRef<T>>;
}
