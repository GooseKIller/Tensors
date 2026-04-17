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

mod elu;
mod leaky_relu;
mod relu;
mod selu;
mod sigmoid;
mod softmax;
mod tanh;
mod prelu;

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::Arc;

pub use elu::*;
pub use leaky_relu::*;
pub use relu::*;
pub use selu::*;
pub use sigmoid::*;
pub use softmax::*;
pub use tanh::*;
pub use prelu::*;

use crate::Float;
use crate::autodiff::{AutoGrad, VarRef};


pub trait Module<T: Float> {
    fn forward(&self, x: &VarRef<T>) -> VarRef<T>;
    fn parameters(&self) -> Vec<VarRef<T>>;

    fn save(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        for param in self.parameters() {
            let data: Vec<f32> = param.value().get_data()
                .iter().map(|x| x.to_f32()).collect();

            for val in data {
                writer.write_all(&val.to_le_bytes())?;
            }
        }

        Ok(())
    }

    fn load(&mut self, path: &str) -> std::io::Result<()> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut buffer = [0u8; 4];

        for param in self.parameters() {
            let var = param.borrow();

            if Arc::strong_count(&var.value.storage) > 1 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!(
                        "Safety violation: tensor has {} owners. Cannot mutate safely.", 
                        Arc::strong_count(&var.value.storage)
                    )
                ));
            }

            let data_ptr = var.value.storage.data.as_ptr() as *mut T;
            let len = var.value.storage.data.len();

            // SAFETY: 
            // 1. We are performing a single-threaded weight loading before the 
            //    training loop starts or after it finishes.
            // 2. The memory pointed to by `data_ptr` is owned by an Arc within 
            //    the Storage struct and is guaranteed to be valid for the duration 
            //    of this function (the Arc is not dropped).
            // 3. We ensure that we do not create multiple mutable references 
            //    to the same data simultaneously across different threads.
            unsafe {
                let data_slice = std::slice::from_raw_parts_mut(data_ptr, len);
                
                for val in data_slice.iter_mut() {
                    reader.read_exact(&mut buffer)?;
                    *val = T::from_f32(f32::from_le_bytes(buffer));
            }
        }
    }

        Ok(())
    }

    fn no_grad(&mut self) {
        for i in self.parameters() {
            i.borrow_mut().requires_grad = false;
        }
    }

    fn grad(&mut self) {
        for i in self.parameters() {
            i.borrow_mut().requires_grad = true;
        }
    }
}
/*
/// A trait for activation functions and other operations that can be applied to matrices.
///
/// This trait is implemented by all activation functions in the Tensors library.
/// It provides a common interface for applying functions to matrices and computing
/// their gradients during backpropagation.
pub trait Function<T: Float>: Any {
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

    fn is_linear(&self) -> bool {
        false
    }

    fn get_data(&self) -> Option<Matrix<T>> {
        None
    }

    fn set_data(&mut self, _data: Matrix<T>) {}

    fn get_weights(&self) -> Option<Matrix<T>> {
        None
    }

    fn get_bias(&self) -> Option<Matrix<T>> {
        None
    }
    fn is_bias(&self) -> bool {false}
}
*/