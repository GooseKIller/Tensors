use crate::Float;
use crate::utils::VarRef;

/// Sum of Squared Errors (SSE) loss function.
///
/// The `SSE` loss function computes the sum of the squared differences between
/// the predicted values and the target values. It is commonly used in regression tasks.
///
/// # Mathematical Definition
/// For predicted values `y_pred` and target values `y_true`, the SSE is defined as:
///
/// SSE = \sum_{i=1}^n (y_{true, i} - y_{pred, i})^2
///
/// # Type Constraints
/// - `T: Float`: The loss function works only with floating-point types (e.g., `f32`, `f64`).
/// # Notes
/// - The `datatype_number` parameter in `new` is a placeholder and is not used in the computation.
///   It is included to ensure type consistency with other loss functions.
/// - SSE is sensitive to outliers due to the squaring of errors.
///
/// # See Also
/// - [Wikipedia: Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
pub fn sse<T:Float>(y: &VarRef<T>, y_pred: &VarRef<T>) -> VarRef<T> {
    let diff = y - y_pred;
    (&(&diff ^ T::from_usize(2)) ^ T::from_f64(0.5)).sum()
}