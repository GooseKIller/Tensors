use crate::Float;
use crate::linalg::Matrix;
use super::Optimizer;


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
    learning_rate: T,
}

impl<T: Float> SGD<T> {
    pub fn new(learning_rate: T) -> Self {
        Self {
            learning_rate
        }
    }
}

impl<T: Float> Optimizer<T> for SGD<T> {
    fn step(&mut self, _id: usize, weights: &mut Matrix<T>, gradients: &Matrix<T>) {
        let g = gradients.clone() * self.learning_rate;
        *weights = weights.clone() + &g;
    }
    fn change_learning_rate(&mut self, new_learning_rate: T) {
        self.learning_rate = new_learning_rate;
    }
}

#[cfg(test)]
mod tests {

}
