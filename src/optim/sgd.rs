use crate::Float;
use crate::linalg::Matrix;
use super::Optimizer;

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
    fn step(&mut self, weights: &mut Matrix<T>, gradients: &Matrix<T>, input_data: &Matrix<T>) {
        let a = input_data.transpose() * &gradients.clone() * self.learning_rate;
        *weights = weights.clone() + &a;
    }
    fn change_learning_rate(&mut self, new_learning_rate: T) {
        self.learning_rate = new_learning_rate;
    }
}

#[cfg(test)]
mod tests {

}
