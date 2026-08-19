use crate::{Float, activation::Module, autodiff::{AutoGrad, Var}};

/// Leaky ReLU activation function.
///
/// Allows a small, non-zero gradient when the input is negative.
///
/// # Mathematical Definition
/// For an input `x`, the Leaky ReLU function is defined as:
/// ```math
///  \text{LeakyReLU}(x) = \max(x, \alpha x)
/// ```
/// or
/// ```math
///  \text{LeakyReLU}(x) = \left\{
/// \begin{array}{ll}
/// x & \text{if } x \geq 0 \\
/// \alpha x & \text{otherwise}
/// \end{array}
/// \right.
/// ```
///
/// # See Also
/// - [velog.io: Leaky ReLU](https://velog.io/@greensox284/Neural-Leaky-Rectified-Linear-Unit-Leaky-ReLU)

pub struct LeakyReLU<T: Float> {
    alpha: T,
}

impl<T: Float> LeakyReLU<T> {
    pub fn new(alpha: T) -> Self {
        Self { alpha }
    }
}

impl<T: Float> Module<T> for LeakyReLU<T> {
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

        let pos_part = x & &m_pos;

        let neg_part = &(x * self.alpha) & &m_neg;

        // 4. The two branches summed: x (where x > 0) + alpha * x (where x <= 0)
        &pos_part + &neg_part
    }

    fn parameters(&self) -> Vec<crate::autodiff::VarRef<T>> {
        vec![]
    }
}