use crate::{Float, autodiff::{AutoGrad, VarRef, clamp_op, log_op}};

/// Binary cross entropy, for two-class classification
///
/// # Formula
///```math
///  BCE(\hat{y}, y) = -\frac{1}{n} \sum_{i=1}^{n}
///      \Big( y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \Big)
///```
/// Where $`\hat{y}_i \in (0, 1)`$ is the predicted probability,
/// $`y_i \in \{0, 1\}`$ the expected label and $`n`$ the batch size
///
/// # Example
/// ```
/// use tensorrs::{tensor, loss::binary_cross_entropy, autodiff::{AutoGrad, Var}};
///
/// let pred   = Var::leaf(tensor![[0.9f32], [0.1]], false);
/// let target = Var::leaf(tensor![[1.0f32], [0.0]], false);
///
/// let loss = binary_cross_entropy(&pred, &target);
/// assert!((loss.value().item() - 0.10536052).abs() < 1e-6); // -ln(0.9)
/// ```
///
/// # Arguments
/// * `pred` — the predicted **probabilities**, usually the output of a
///   [Sigmoid](crate::activation::Sigmoid).
/// * `target` — the expected labels, `0.0` or `1.0`.
///
/// # Returns
/// A scalar node of the autodiff graph.
///
/// # Notes
/// This function expects probabilities, **not** logits. Feed it the output of a
/// [Sigmoid](crate::activation::Sigmoid) layer.
///
/// The prediction is clamped into $`[\varepsilon, 1 - \varepsilon]`$ (`1e-7` for
/// `f32`, `1e-12` for `f64`), so that neither logarithm can reach infinity. A
/// clamped value receives no gradient, see [clamp_op](crate::autodiff::clamp_op).
pub fn binary_cross_entropy<T: Float>(pred: &VarRef<T>, target: &VarRef<T>) -> VarRef<T> {
    let eps = T::f32_f64(1e-7, 1e-12);
    let one = T::one();

    // keep the prediction inside (0, 1), so that both logarithms stay finite
    let safe_pred = clamp_op(pred, eps, one - eps);
    let one_minus_pred = &-&safe_pred + one;

    let log_pred = log_op(&safe_pred);
    let log_one_minus_pred = log_op(&one_minus_pred);

    let term1 = target & &log_pred;                      // y * log(p)
    let term2 = &(&-target + one) & &log_one_minus_pred; // (1-y) * log(1-p)

    let total_loss = -&(&term1 + &term2).sum();
    let batch_size = T::from_usize(pred.value().get_shape()[0]);

    &total_loss / batch_size
}
