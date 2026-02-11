use crate::{Float, activation::Module, linalg::Tensor, utils::Var};

/// Hyperbolic Tangent (Tanh) activation function.
///
/// Maps input values to the range `[-1, 1]`.
///
/// # Mathematical Definition
/// For an input `x`, the Tanh function is defined as:
/// ```math
/// tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}
/// ```
///
/// # Examples
/// ```
/// use tensorrs::activation::{Module, Tanh};
/// use tensorrs::linalg::Matrix;
///
/// let tanh = Tanh::new();
/// let input = Matrix::from(vec![vec![0.0], vec![1.0], vec![-1.0]]);
/// let output = tanh.forward(input);
/// println!("Tanh output: {}", output);
/// ```
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
    fn forward(&self, x: &crate::utils::VarRef<T>) -> crate::utils::VarRef<T> {
        let e_val = T::f32_f64(std::f32::consts::E, std::f64::consts::E);
        let e = Var::leaf(Tensor::scalar(e_val), false);

        // e^x
        let exp_x = &e ^ x;
        // e^(-x)
        let exp_neg_x = &e ^ &-x;

        // (e^x - e^-x) / (e^x + e^-x)
        let numerator = &exp_x - &exp_neg_x;
        let denominator = &exp_x + &exp_neg_x;

        &numerator / &denominator
    }

    fn parameters(&self) -> Vec<crate::utils::VarRef<T>> {
        vec![]
    }
}