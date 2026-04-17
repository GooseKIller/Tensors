use crate::{Float, activation::Module, autodiff::{AutoGrad, Var}};

/// Rectified Linear Unit (ReLU) activation function.
///
/// Outputs the input directly if it is positive; otherwise, it outputs zero.
///
/// # Mathematical Definition
/// For an input `x`, the ReLU function is defined as:
/// ```math
///  \text{ReLU}(x) = \max(x, 0)
/// ```
/// or
/// ```math
///  \text{ReLU}(x) = \left\{
/// \begin{array}{ll}
/// x & \text{if } x \geq 0 \\
/// 0 & \text{if } x < 0
/// \end{array}
/// \right.
/// ```
///
/// # See Also
/// - [Wikipedia: Rectifier (neural networks)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }
}

impl<T:Float> Module<T> for ReLU {
    fn forward(&self, x: &crate::autodiff::VarRef<T>) -> crate::autodiff::VarRef<T> {
        let x_val = x.value();

        let m_pos = Var::leaf(
            x_val.map(|v| if v > T::default() {T::one()} else {T::default()}),
            false
        );

        x & &m_pos
    }
    fn parameters(&self) -> Vec<crate::autodiff::VarRef<T>> {
        vec![]
    }
}