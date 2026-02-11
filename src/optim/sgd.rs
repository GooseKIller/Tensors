use super::Optimizer;
use crate::Float;
use crate::utils::{AutoGrad, VarRef};

/// Stochastic Gradient Descent(SGD)
///
/// # Formula:
/// ```math
/// W^{t+1}_{i} = W^t_{i+1} - \eta \cdot \nabla L(W_{t})
/// ```
/// where:
/// ```math
/// - W^{t+1}_{i} — updated weights,
/// - W^t_{i} — current weights,
/// - \eta — learning rate,
/// - \nabla L(W_{t}) — gradient of the loss function with respect to the weights.
/// ```
pub struct SGD<T: Float> {
    params: Vec<VarRef<T>>,
    lr: T,
}

impl<T: Float> SGD<T> {
    pub fn new(params: Vec<VarRef<T>>, lr: T) -> Self {
        Self { params, lr }
    }
}

impl<T:Float> Optimizer<T> for SGD<T> {
    fn step(&mut self) {
        for param in &self.params {
            let mut p = param.borrow_mut();
            let grad = p.grad.borrow().shallow_copy();

            p.value = &p.value - &(&grad * self.lr);
        }    
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}
