use crate::Float;
use crate::autodiff::{VarRef, abs_op};

/// Mean absolute percentage error
///
/// # Formula
///```math
///  MAPE(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{\hat{y}_i - y_i}{y_i} \right|
///```
/// Where $`\hat{y}_i`$ predicted and $`y_i`$ expected value, $`n`$ is the batch size
///
/// # Example
/// ```
/// use tensorrs::{tensor, loss::mape, autodiff::{AutoGrad, Var}};
///
/// let y_pred = Var::leaf(tensor![[2.0f32], [4.0]], false);
/// let y      = Var::leaf(tensor![[1.0f32], [2.0]], false);
///
/// let loss = mape(&y_pred, &y);
/// assert_eq!(loss.value().item(), 1.0); // (|1/1| + |2/2|) / 2
/// ```
///
/// # Arguments
/// * `y_pred` — the predicted values, of shape `[batch, ...]`.
/// * `y` — the expected values, of the same shape.
///
/// # Returns
/// A scalar node of the autodiff graph. The result is a fraction, not a
/// percentage — multiply by `100` to read it as one.
///
/// # Notes
/// The error is relative, so it says nothing useful when an expected value is
/// zero. To keep the division finite the denominator is guarded at
/// $`\varepsilon`$ (`1e-7` for `f32`, `1e-12` for `f64`), which makes such a
/// sample dominate the sum instead of turning it into infinity.
pub fn mape<T:Float>(y_pred: &VarRef<T>, y: &VarRef<T>) -> VarRef<T> {
    let eps = T::f32_f64(1e-7, 1e-12);

    // the expected values are a constant here, so the guarded denominator
    // |y| can stay a plain tensor and never enters the graph
    let denom = y.0.borrow().value.map(|v| {
        let a = v.abs();
        if a < eps { eps } else { a }
    });

    let diff = y_pred - y;
    let ratio = &abs_op(&diff) / &denom;

    &ratio.sum() / T::from_usize(y_pred.0.borrow().value.shape[0])
}
