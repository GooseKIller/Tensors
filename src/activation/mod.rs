//!# Activation Functions
//!
//! functions for adding non-linearity to a neural network
//!
//! They all have call and derivative methods.
//!
//!1.[ELU]
//!
//!2.[LeakyReLU]
//!
//!3.[ReLU]
//!
//!4.[SELU]
//!
//!5.[Sigmoid]
//!
//!6.[SoftMax]

mod relu;
mod softmax;
mod sigmoid;
mod leaky_relu;
mod elu;
mod selu;
mod tanh;

use std::any::Any;
pub use relu::*;
pub use softmax::*;
pub use sigmoid::*;
pub use leaky_relu::*;
pub use elu::*;
pub use selu::*;
pub use tanh::*;

use crate::linalg::Matrix;
use crate::Float;

/// A trait for activation functions and other operations that can be applied to matrices.
///
/// This trait is implemented by all activation functions in the Tensors library.
/// It provides a common interface for applying functions to matrices and computing
/// their gradients during backpropagation.
pub trait Function<T: Float>: Any{
    fn name(&self) -> String;

    /// Applies the function to the input matrix.
    ///
    /// This method is the primary way to use a function (e.g., activation function, layer)
    /// in the Tensors library. It takes an input matrix, applies the function to each element,
    /// and returns the resulting matrix.
    ///
    /// # Arguments
    /// * `matrix` - The input matrix to which the function will be applied.
    ///
    /// # Returns
    /// A new matrix with the function applied to matrix.
    ///
    /// # Notes
    /// - In Python, you might be familiar with the `__call__` method, which allows an object
    ///   to be called like a function (e.g., `sigmoid(input)`). Rust does not have a direct
    ///   equivalent, so we use the `call` method instead.
    /// - If you prefer a more concise syntax, consider implementing the `Function` trait,
    ///   which provides a `forward` method that can be used similarly.
    fn call(&self, matrix: Matrix<T>) -> Matrix<T>;

    /// Derivative for Function
    ///
    /// ## Arguments
    ///
    /// * `matrix` - the input matrix to which the derivative will be applied
    fn derivative(&self, matrix: Matrix<T>) -> Matrix<T>;

    fn is_linear(&self) -> bool{
        false
    }

    fn get_data(&self) -> Option<Matrix<T>> {
        None
    }

    fn set_data(&mut self, _data:Matrix<T>){}

    fn get_weights(&self) -> Option<Matrix<T>> {None}

    fn get_bias(&self) -> Option<Matrix<T>> {None}
    fn is_bias(&self) -> bool {false}
}