use crate::{Float, utils::{AutoGrad, VarRef, log_op}};

pub fn binary_cross_entropy<T: Float>(pred: &VarRef<T>, target: &VarRef<T>) -> VarRef<T> {
    let eps = T::f32_f64(1e-7, 1e-12); // Чуть меньше eps
    let one = T::one();

    // Clamp: ограничиваем pred в диапазоне [eps, 1-eps], чтобы log не взорвался
    // Если у вас нет clamp, используйте вашу логику с eps, но аккуратнее:
    let log_pred = log_op(&(pred + eps));
    let log_one_minus_pred = log_op(&(&(&-pred + one) + eps));

    let term1 = target & &log_pred;                     // y * log(p)
    let term2 = &(&-target + one) & &log_one_minus_pred; // (1-y) * log(1-p)

    let total_loss = -&(&term1 + &term2).sum();
    let batch_size = T::from_usize(pred.value().get_shape()[0]);

    &total_loss / batch_size
}