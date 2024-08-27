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

use std::any::Any;
pub use relu::*;
pub use softmax::*;
pub use sigmoid::*;
pub use leaky_relu::*;
pub use elu::*;
pub use selu::*;

use crate::linalg::Matrix;
use crate::Float;

/// All activation functions works only with Float types
///
/// All activation function must implement this trait Function
pub trait Function<T: Float>: Any{

    /// Rust does not have similar thing like \_\_call__ in Python
    ///
    /// So just use method call
    fn call(&self, matrix: Matrix<T>) -> Matrix<T>;

    /// Derivative for Function
    fn derivative(&self, matrix: Matrix<T>) -> Matrix<T>;

    fn is_linear(&self) -> bool{
        false
    }

    fn get_data(&self) -> Option<Matrix<T>> {
        None
    }
    fn get_bias(&self) -> Option<Matrix<T>>{
        None
    }

    fn set_data(&mut self, data:Matrix<T>){}
    fn set_bias(&mut self, new_bias:Matrix<T>){}
}