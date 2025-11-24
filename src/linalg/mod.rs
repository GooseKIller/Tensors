//! # Linear Algebra
//!
//! This module includes fundamental linear algebra structures for computations in machine learning.
//!
//! Objects:
//!
//! 1.[Vector] - A one-dimensional array for representing points or directions in space.
//!
//! 2.[Matrix] - A two-dimensional array for representing linear transformations and data structures.
//!
//! 3.[Tensor] (not finished) - A multi-dimensional array for generalizing vectors and matrices to higher dimensions.

mod matrix;
mod matrix_ops;
mod matrix_conv;
mod tensor;
mod tensor_ops;
mod vector;

pub use matrix::*;
pub use tensor::*;
pub use vector::*;
