use crate::{Float, activation::Module};

/// Hyperbolic Tangent (Tanh) activation function.
///
/// Maps input values to the range `[-1, 1]`.
///
/// # Formula
/// ```math
/// \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
/// ```
///
/// # Example
/// ```
/// use tensorrs::activation::{Module, Tanh};
/// use tensorrs::linalg::Tensor;
/// use tensorrs::autodiff::{Var, AutoGrad};
///
/// let input = Tensor::from(vec![vec![0.0f32], vec![90.0], vec![-90.0]]);
/// let output = Tanh::new().forward(&Var::leaf(input, false));
///
/// // saturates instead of overflowing, at any magnitude
/// assert_eq!(output.value().get_data(), vec![0.0, 1.0, -1.0]);
/// ```
///
/// # Notes
/// A single graph node, see [tanh_op](crate::autodiff::tanh_op). Its derivative is
/// read off the output as `1 - tanh^2(x)`, which costs one multiply per element.
///
/// # See Also
/// - [Wikipedia: Hyperbolic functions](https://en.wikipedia.org/wiki/Hyperbolic_functions)
pub struct Tanh;

impl Tanh {
    /// Creates a new `Tanh` activation function.
    ///
    /// # Returns
    /// A new instance of the `Tanh` activation function.
    pub fn new() -> Self {
        Self
    }
}

impl<T: Float> Module<T> for Tanh {
    fn forward(&self, x: &crate::autodiff::VarRef<T>) -> crate::autodiff::VarRef<T> {
        crate::autodiff::tanh_op(x)
    }

    fn parameters(&self) -> Vec<crate::autodiff::VarRef<T>> {
        vec![]
    }
}
