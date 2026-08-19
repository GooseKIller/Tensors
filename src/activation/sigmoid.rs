use crate::{Float, activation::Module};

/// Sigmoid activation function.
///
/// Maps input values into the range (0, 1) using the logistic function.
///
/// # Formula
/// ```math
/// \sigma(x) = \frac{1}{1 + e^{-x}}
/// ```
///
/// # Example
/// ```
/// use tensorrs::{tensor, activation::{Module, Sigmoid}, autodiff::{AutoGrad, Var}};
///
/// let x = Var::leaf(tensor![[0.0f32, 400.0, -400.0]], false);
/// let y = Sigmoid::new().forward(&x);
///
/// // saturates cleanly at both ends, at any magnitude
/// assert_eq!(y.value().get_data(), vec![0.5, 1.0, 0.0]);
/// ```
///
/// # Notes
/// A single graph node, see [sigmoid_op](crate::autodiff::sigmoid_op). Its
/// derivative is read off the output as $`\sigma(x)(1 - \sigma(x))`$, so the
/// backward pass costs one multiply per element and no exponentials.
///
/// # See Also
/// - [Wikipedia: Logistic function](https://en.wikipedia.org/wiki/Logistic_function)
pub struct Sigmoid;

impl Sigmoid {
    /// Creates a new `Sigmoid` activation function.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> Module<T> for Sigmoid {
    fn forward(&self, x: &crate::autodiff::VarRef<T>) -> crate::autodiff::VarRef<T> {
        crate::autodiff::sigmoid_op(x)
    }

    fn parameters(&self) -> Vec<crate::autodiff::VarRef<T>> {
        vec![]
    }
}
