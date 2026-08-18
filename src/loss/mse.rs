use crate::{Float, autodiff::VarRef};

/// Mean squared error
///
/// # Formula
///```math
///  MSE(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
///```
/// Where $`\hat{y}_i`$ predicted and $`y_i`$ expected value, $`n`$ is the batch size
///
/// # Example
/// ```
/// use tensorrs::{tensor, loss::mse, autodiff::{AutoGrad, Var}};
///
/// let y_pred = Var::leaf(tensor![[2.0f32], [4.0]], false);
/// let y      = Var::leaf(tensor![[1.0f32], [2.0]], false);
///
/// let loss = mse(&y_pred, &y);
/// assert_eq!(loss.value().item(), 2.5); // (1 + 4) / 2
/// ```
///
/// # Arguments
/// * `y_pred` — the predicted values, of shape `[batch, ...]`.
/// * `y` — the expected values, of the same shape.
///
/// # Returns
/// A scalar node of the autodiff graph. Call `backward()` on it to propagate the
/// gradients back to the model parameters.
///
/// # Notes
/// $`n`$ is taken from the first dimension of `y_pred`, so the error is averaged
/// over the batch and summed over everything else.
pub fn mse<T:Float>(y_pred: &VarRef<T>, y: &VarRef<T>) -> VarRef<T> {
    let diff = y_pred - y;
    &(&diff ^ T::from_usize(2)).sum() / T::from_usize(y_pred.0.borrow().value.shape[0])
}
