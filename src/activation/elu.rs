use crate::activation::Module;
use crate::linalg::Tensor;
use crate::Float;
use crate::autodiff::{AutoGrad, Var, VarRef};

/// Exponential Linear Unit (ELU) activation function.
///
/// Maps input values such that:
/// - For `x >= 0`: returns `x`
/// - For `x < 0`: returns `α * (e^x - 1)`
///
/// # Mathematical Definition
/// For an input `x`, the ELU function is defined as:
/// ```math
/// \text{ELU}(x) = \left\{
/// \begin{array}{ll}
/// x & \text{if } x \geq 0 \\
/// \alpha \left( e^x - 1 \right) & \text{if } x < 0
/// \end{array}
/// \right.
/// ```
///
/// # See Also
/// - [velog.io: Exponential Linear Unit](https://velog.io/@greensox284/Neural-Exponential-Linear-Unit)

pub struct ELU<T: Float> {
    alpha: T,
}

impl<T: Float> ELU<T> {
    pub fn new(_datatype_num: T) -> Self {
        Self { alpha: 1.into() }
    }
}

/// sets alpha value
impl<T: Float> From<T> for ELU<T> {
    fn from(value: T) -> Self {
        Self { alpha: value }
    }
}

impl<T:Float> Module<T> for ELU<T> {
    fn forward(&self, x: &VarRef<T>) -> VarRef<T> {
        let x_val = x.value();

        let m_pos = Var::leaf(
            x_val.map(|v| if v > T::default() {T::one()} else {T::default()}),
            false
        );
        let m_neg = Var::leaf(
            x_val.map(|v| if v <= T::default() {T::one()} else {T::default()}),
            false
        );

        let pos_part = x & &m_pos;

        let e_val = T::f32_f64(std::f32::consts::E, std::f64::consts::E);
        let e = Var::leaf(Tensor::scalar(e_val), false);
        let exp_x = &e ^ x;

        let neg_part = &(&(&(&exp_x - T::one()) * self.alpha) & &m_neg);

        &pos_part + neg_part
    }

    fn parameters(&self) -> Vec<crate::autodiff::VarRef<T>> {
        vec![]
    }
}


