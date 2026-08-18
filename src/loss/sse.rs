use crate::Float;
use crate::autodiff::VarRef;

/// Sum of squared errors
///
/// # Formula
///```math
///  SSE(\hat{y}, y) = \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
///```
/// Where $`\hat{y}_i`$ predicted and $`y_i`$ expected value
///
/// # Example
/// ```
/// use tensorrs::{tensor, loss::sse, autodiff::{AutoGrad, Var}};
///
/// let y_pred = Var::leaf(tensor![[2.0f32], [4.0]], false);
/// let y      = Var::leaf(tensor![[1.0f32], [2.0]], false);
///
/// let loss = sse(&y_pred, &y);
/// assert_eq!(loss.value().item(), 5.0); // 1 + 4
/// ```
///
/// # Arguments
/// * `y_pred` — the predicted values.
/// * `y` — the expected values, of the same shape.
///
/// # Returns
/// A scalar node of the autodiff graph.
///
/// # Notes
/// Unlike [mse](crate::loss::mse) the result is not divided by the batch size, so
/// it grows with the number of samples. Squaring makes it sensitive to outliers.
///
/// # See Also
/// [Wikipedia: Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
pub fn sse<T:Float>(y_pred: &VarRef<T>, y: &VarRef<T>) -> VarRef<T> {
    let diff = y_pred - y;
    (&diff ^ T::from_usize(2)).sum()
}
