use crate::{Float, activation::Module, linalg::Tensor, autodiff::Var};

/// Sigmoid activation function.
///
/// Maps input values into the range (0, 1) using the logistic function.
///
/// # Mathematical Definition
/// For an input `x`, the Sigmoid function is defined as:
/// ```math
/// Sigmoid(x) = \frac{1}{1 + e^{-x}}
/// ```
///
/// # See Also
/// - [Wikipedia: Logistic function](https://en.wikipedia.org/wiki/Logistic_function)
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl<T: Float> Module<T> for Sigmoid {
    fn forward(&self, x: &crate::autodiff::VarRef<T>) -> crate::autodiff::VarRef<T> {
        
        let e_val = T::f32_f64(std::f32::consts::E, std::f64::consts::E);
        let e = Var::leaf(Tensor::scalar(e_val), false);
        let exp_neg_x = &e ^ &-x;
        let denom = &exp_neg_x + T::one();
        let one = Var::leaf(Tensor::scalar(T::one()), false);
        &one / &denom
    }

    fn parameters(&self) -> Vec<crate::autodiff::VarRef<T>> {
        vec![]
    }
}