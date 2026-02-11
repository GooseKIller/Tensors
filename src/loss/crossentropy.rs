use crate::{Float, utils::{AutoGrad, VarRef, log_op}};

pub fn cross_entropy<T: Float>(pred: &VarRef<T>, target: &VarRef<T>) -> VarRef<T> {
    // Формула: -1/n * sum(target * log(pred + eps))
    // eps нужен, чтобы не получить log(0)
    let eps = T::from_f64(1e-10);
    
    // 1. log(pred + eps)
    // Здесь мы используем твой будущий log_op
    let log_pred = log_op(&(pred + eps));

    // 2. target * log(pred)
    let product = target & &log_pred;

    // 3. -sum(product) / batch_size
    let batch_size = T::from_usize(pred.value().get_shape()[0]);
    let sum_loss = product.sum();
    
    &(-&sum_loss) / batch_size
}