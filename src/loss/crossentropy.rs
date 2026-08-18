use crate::{Float, autodiff::{AutoGrad, VarRef, clamp_op, log_op}};

/// Cross entropy, for multi-class classification
///
/// # Formula
///```math
///  H(\hat{y}, y) = -\frac{1}{n} \sum_{i=1}^{n} y_i \ln(\hat{y}_i)
///```
/// Where $`\hat{y}_i`$ is the predicted probability, $`y_i`$ the one-hot encoded
/// expected value and $`n`$ the batch size
///
/// # Example
/// ```
/// use tensorrs::{tensor, loss::cross_entropy, autodiff::{AutoGrad, Var}};
///
/// let pred   = Var::leaf(tensor![[0.7f32, 0.3]], false);
/// let target = Var::leaf(tensor![[1.0f32, 0.0]], false); // class 0
///
/// let loss = cross_entropy(&pred, &target);
/// assert!((loss.value().item() - 0.35667494).abs() < 1e-6); // -ln(0.7)
/// ```
///
/// # Arguments
/// * `pred` — the predicted **probabilities**, usually the output of a
///   [SoftMax](crate::activation::SoftMax).
/// * `target` — the one-hot encoded expected classes, see
///   [one_hot_encoding](crate::utils::one_hot_encoding).
///
/// # Returns
/// A scalar node of the autodiff graph.
///
/// # Notes
/// This function expects probabilities, **not** logits — unlike the equally named
/// loss in some other frameworks. Feed it the output of a
/// [SoftMax](crate::activation::SoftMax) layer.
///
/// The prediction is clamped into $`[\varepsilon, 1 - \varepsilon]`$ (`1e-7` for
/// `f32`, `1e-12` for `f64`) before the logarithm, so a saturated prediction
/// cannot turn the loss into infinity. A clamped value receives no gradient, see
/// [clamp_op](crate::autodiff::clamp_op).
pub fn cross_entropy<T: Float>(pred: &VarRef<T>, target: &VarRef<T>) -> VarRef<T> {
    let eps = T::f32_f64(1e-7, 1e-12);
    let one = T::one();

    // keep the prediction inside (0, 1): ln(0) is infinite and ln(x < 0) is NaN
    let safe_pred = clamp_op(pred, eps, one - eps);
    let log_pred = log_op(&safe_pred);

    // -sum(target * log(pred)) / batch_size
    let product = target & &log_pred;
    let batch_size = T::from_usize(pred.value().get_shape()[0]);
    let sum_loss = product.sum();

    &(-&sum_loss) / batch_size
}
