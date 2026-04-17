use crate::Float;
use crate::autodiff::VarRef;

/// Mean absolute percentage error
///
///  # Formula:
///```math
///  MAPE(ŷ, y) = \frac{1}{n} * \sum_{i=1}^{n} \left| \frac{ŷ_i - y_i}{y_i} \right|
///```
///
/// Where $`ŷ_i`$ predicted and $`y_i`$ expected value
pub fn mape<T:Float>(y: &VarRef<T>, y_pred: &VarRef<T>) -> VarRef<T> {
    let diff = y - y_pred;
    &(&diff ^ T::from_usize(2)).sum() / T::from_usize(y_pred.0.borrow().value.shape[0])
}