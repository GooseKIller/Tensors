//!# Activation Functions
//!
//! functions for adding non-linearity to a neural network
//!
//! They all implement [Module], so they can be stacked into a
//! [Sequential](crate::nn::Sequential) model exactly like any other layer.
//!
//!1.[ELU]
//!
//!2.[LeakyReLU]
//!
//!3.[PReLU]
//!
//!4.[ReLU]
//!
//!5.[SELU]
//!
//!6.[Sigmoid]
//!
//!7.[SoftMax]
//!
//!8.[Tanh]

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


/// Anything that can take part in a forward pass: a layer, an activation
/// function, or a whole model.
///
/// # Example
/// ```
/// use tensorrs::{tensor, Float, activation::Module,
///                autodiff::{AutoGrad, Var, VarRef}};
///
/// // A layer without parameters only has to define `forward` and `parameters`
/// struct Double;
///
/// impl<T: Float> Module<T> for Double {
///     fn forward(&self, x: &VarRef<T>) -> VarRef<T> {
///         x + x
///     }
///     fn parameters(&self) -> Vec<VarRef<T>> {
///         vec![]
///     }
/// }
///
/// let x = Var::leaf(tensor![[1.0f32, 2.0]], false);
/// assert_eq!(Double.forward(&x).value().get_data(), vec![2.0, 4.0]);
/// ```
///
/// # Notes
/// Only [Module::forward] and [Module::parameters] have to be implemented — saving,
/// loading and freezing are all derived from `parameters()`.
pub trait Module<T: Float> {
    /// Runs the input through this module and returns the output node.
    ///
    /// # Arguments
    /// * `x` — the input node of the autodiff graph.
    ///
    /// # Returns
    /// The output node. The operations performed here are recorded in the graph,
    /// so `backward()` can walk them in reverse.
    fn forward(&self, x: &VarRef<T>) -> VarRef<T>;

    /// Returns the trainable parameters of this module.
    ///
    /// # Returns
    /// An empty vector for modules without parameters, such as most activation
    /// functions. The order has to stay stable — [Module::save] and [Module::load]
    /// rely on it.
    fn parameters(&self) -> Vec<VarRef<T>>;

    /// Writes every parameter to `path` as raw little-endian `f32`.
    ///
    /// # Example
    /// ```no_run
    /// use tensorrs::{activation::Module, nn::{Linear, Sequential}};
    ///
    /// let model: Sequential<f32> = Sequential::new(vec![
    ///     Box::new(Linear::new(2, 1, true)),
    /// ]);
    /// model.save("model.bin").unwrap();
    /// ```
    ///
    /// # Arguments
    /// * `path` — the file to write to; it is created or truncated.
    ///
    /// # Notes
    /// Only the numbers are stored, not the architecture, so loading them back
    /// requires a model built exactly the same way. Values are always narrowed to
    /// `f32`, so an `f64` model loses precision on a round trip.
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

    /// Reads the parameters back from a file written by [Module::save].
    ///
    /// # Arguments
    /// * `path` — the file to read from.
    ///
    /// # Returns
    /// An error if the file cannot be read, or if a parameter tensor shares its
    /// storage with another tensor and therefore cannot be mutated safely.
    ///
    /// # Notes
    /// Call it before training starts or after it finishes: the parameters are
    /// overwritten in place, and that shared-storage check is the only guard
    /// against another part of the graph observing a half-written tensor.
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

    /// Freezes the module: its parameters stop collecting gradients.
    ///
    /// # Notes
    /// Useful for fine-tuning, where only the last layers are trained. Undo it
    /// with [Module::grad].
    /// Switches the module into training mode.
    ///
    /// # Example
    /// ```
    /// use tensorrs::{activation::Module, nn::{Dropout, Linear, Sequential}};
    ///
    /// let mut model: Sequential<f32> = Sequential::new(vec![
    ///     Box::new(Linear::new(4, 4, true)),
    ///     Box::new(Dropout::new(0.5)),
    /// ]);
    ///
    /// model.eval();  // dropout passes everything through
    /// model.train(); // and back to dropping
    /// ```
    ///
    /// # Notes
    /// Only layers that behave differently between training and inference care —
    /// [Dropout](crate::nn::Dropout) is the one that does. Everything else keeps
    /// the default, which does nothing. A container such as
    /// [Sequential](crate::nn::Sequential) passes the call on to its layers.
    ///
    /// Modules start in training mode.
    fn train(&mut self) {}

    /// Switches the module into inference mode.
    ///
    /// # Notes
    /// The counterpart of [Module::train]. Forgetting to call it before measuring
    /// accuracy is a classic way to get results that look worse than the model is:
    /// dropout would still be throwing activations away.
    fn eval(&mut self) {}

    fn no_grad(&mut self) {
        for i in self.parameters() {
            i.borrow_mut().requires_grad = false;
        }
    }

    /// Unfreezes the module: its parameters collect gradients again.
    ///
    /// # Notes
    /// The counterpart of [Module::no_grad]. Parameters are created unfrozen, so
    /// this is only needed after a [Module::no_grad] call.
    fn grad(&mut self) {
        for i in self.parameters() {
            i.borrow_mut().requires_grad = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        activation::{Module, Sigmoid, Tanh},
        autodiff::{AutoGrad, Var},
        tensor,
    };

    #[test]
    fn tanh_saturates_instead_of_overflowing() {
        // written out as (e^x - e^-x) / (e^x + e^-x) this is inf/inf past |x| ~ 88
        let x = Var::leaf(tensor![[0.0f32, 20.0, 50.0, 90.0, 300.0, -300.0]], true);
        let y = Tanh::new().forward(&x);

        assert_eq!(y.value().get_data(), vec![0.0, 1.0, 1.0, 1.0, 1.0, -1.0]);
        assert!(y.value().get_data().iter().all(|v| v.is_finite()));

        y.sum().backward();
        let grad = x.grad().get_data();

        // a saturated input must get no gradient at all; the old chain through
        // five nodes reported 1 here
        assert_eq!(grad[0], 1.0);
        assert!(grad[1..].iter().all(|&g| g == 0.0), "saturated grads: {grad:?}");
    }

    #[test]
    fn sigmoid_stays_finite_on_both_sides() {
        let x = Var::leaf(tensor![[0.0f32, 100.0, -100.0, 400.0, -400.0]], true);
        let y = Sigmoid::new().forward(&x);

        // sigmoid(-100) is about 3.7e-44, a subnormal f32 rather than a clean zero
        let values = y.value().get_data();
        assert_eq!(values[0], 0.5);
        assert_eq!(values[1], 1.0);
        assert!(values[2] < 1e-40 && values[2] >= 0.0, "got {}", values[2]);
        assert_eq!(values[3], 1.0);
        assert_eq!(values[4], 0.0);

        y.sum().backward();
        let grad = x.grad().get_data();

        assert!(grad.iter().all(|g| g.is_finite()), "grads: {grad:?}");
        assert_eq!(grad[0], 0.25);
        // saturated inputs get a vanishing gradient rather than a NaN; at -100 it
        // is a subnormal instead of a clean zero, which is the honest answer
        assert!(grad[1..].iter().all(|&g| g < 1e-40), "saturated grads: {grad:?}");
    }

    #[test]
    fn both_match_their_analytic_derivative() {
        for value in [-3.0f64, -0.5, 0.0, 0.5, 3.0] {
            let x = Var::leaf(tensor![[value]], true);
            let y = Tanh::new().forward(&x);
            y.backward();

            let expected = 1.0 - value.tanh() * value.tanh();
            assert!((x.grad().get_data()[0] - expected).abs() < 1e-12,
                "tanh' at {value}: got {} want {expected}", x.grad().get_data()[0]);

            let x = Var::leaf(tensor![[value]], true);
            let y = Sigmoid::new().forward(&x);
            y.backward();

            let s = 1.0 / (1.0 + (-value).exp());
            assert!((x.grad().get_data()[0] - s * (1.0 - s)).abs() < 1e-12,
                "sigmoid' at {value}");
        }
    }
}
