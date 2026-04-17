use crate::{Float, activation::Module, linalg::Tensor, autodiff::{AutoGrad, Var}};

/// Scaled Exponential Linear Unit (SELU).
///
/// Applies the SELU activation function element-wise.
/// SELU is defined with a default scale parameter `λ` and a default alpha parameter `α`.
///
/// # Mathematical Definition
/// For an input `x`, the SELU function is defined as:
///```math
/// SELU(x) = \left\{
/// \begin{array}{ll}
/// \lambda x & \text{if } x > 0 \\
/// \lambda \alpha \left( e^x - 1 \right) & \text{if } x \leq 0
/// \end{array}
/// \right.
/// ```
///
/// By default, the parameters are set to:
/// - α = 1.67326
/// - λ = 1.0507
///
/// # See Also
/// - [velog.io: Scaled Exponential Linear Unit](https://velog.io/@greensox284/Activation-Scaled-Exponential-Linear-Unit-SELU)
pub struct SELU<T: Float> {
    alpha: T,
    lambda: T,
}

impl<T: Float> SELU<T> {
    pub fn new() -> Self {
        let alpha: T = T::selu_alpha(T::default());
        let lambda: T = T::selu_lambda(T::default());
        Self { alpha, lambda }
    }
}

impl<T: Float> From<(T, T)> for SELU<T> {
    fn from(params: (T, T)) -> Self {
        let (alpha, scale) = params;
        Self {
            alpha,
            lambda: scale,
        }
    }
}

impl<T: Float> Module<T> for SELU<T> {
    fn forward(&self, x: &crate::autodiff::VarRef<T>) -> crate::autodiff::VarRef<T> {
        let x_val = x.value();

        let m_pos = Var::leaf(
            x_val.map(|v| if v > T::default() { T::one() } else { T::default() }),
            false
        );
        let m_neg = Var::leaf(
            x_val.map(|v| if v <= T::default() { T::one() } else { T::default() }),
            false
        );

        let pos_part = &(x * self.lambda) & &m_pos;

        let e_val = T::f32_f64(std::f32::consts::E, std::f64::consts::E);
        let e = Var::leaf(Tensor::scalar(e_val), false);
        let exp_x = &e ^ x;

        let scale_factor = self.lambda * self.alpha;
        let neg_part = &(&(&(&exp_x - T::one()) * scale_factor) & &m_neg);

        &pos_part + neg_part
    }

    fn parameters(&self) -> Vec<crate::autodiff::VarRef<T>> {
        vec![]
    }
}