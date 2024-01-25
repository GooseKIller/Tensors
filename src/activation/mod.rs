//!# Activation Functions
//!
//! functions for adding non-linearity to a neural network
//!
//! They all have call and derivative methods.
//!
//!1.[Sigmoid]
//!
//!2.[ReLU]
//!
//!3.GelU
//!
//!4.[SoftMax]

mod relu;
mod softmax;
mod sigmoid;

pub use relu::*;
pub use softmax::*;
pub use sigmoid::*;

use crate::linalg::Matrix;
use crate::Float;

/// All activation functions works only with Float types
///
/// All activation function must implement this trait Function
pub trait Function<T: Float>{

    /// Rust does not have similar thing like \_\_call__ in Python
    ///
    /// So just use method call
    fn call(self, matrix: Matrix<T>) -> Matrix<T>;

    /// Derivative for Function
    fn derivative(self, matrix: Matrix<T>) -> Matrix<T>;
}