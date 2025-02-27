//! # Building Blocks for Neural Networks
//!
//! 1.[Linear] - Linear layer for neural network
//!
//! 2.[Sequential] - Sequential model for building neural networks by stacking layers

mod linear;
mod sequential;

pub use linear::*;
pub use sequential::*;
