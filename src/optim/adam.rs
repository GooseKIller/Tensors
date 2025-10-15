use crate::activation::Function;
use crate::linalg::Matrix;
use crate::optim::Optimizer;
use crate::Float;
use rayon::prelude::IntoParallelRefMutIterator;
use rayon::prelude::*;

/// Adaptive Moment Estimation (ADAM) optimizer.
///
/// # Formulas
/// 1. **First Moment Estimate (m)**:
/// ```math
/// m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
/// ```
///
/// 2. **Second Moment Estimate (v)**:
/// ```math
/// v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
/// ```
///
/// 3. **Bias-Corrected Estimates**:
/// ```math
/// \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
/// ```
///
/// ```math
/// \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
/// ```
///
/// 4. **Parameter Update**:
/// ```math
/// \theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
/// ```
///
/// where:
/// - $ m_t $ is the first moment estimate (mean of gradients).
/// - $ v_t $ is the second moment estimate (uncentered variance of gradients).
/// - $ g_t $ is the gradient at time step $ t $.
/// - $ \beta_1 $ and $ \beta_2 $ are the decay rates for the moment estimates.
/// - $ \alpha $ is the learning rate.
/// - $ \epsilon $ is a small constant to prevent division by zero.
/// - $ \theta_t $ are the parameters being optimized.
pub struct Adam<T: Float> {
    learning_rate: T,
    beta1: T,
    beta2: T,
    epsilon: T,
    m: Vec<Matrix<T>>,
    v: Vec<Matrix<T>>,
    t: Vec<usize>,
}

impl<T: Float> Adam<T> {
    pub fn new(learning_rate: T, architecture: &Vec<Box<dyn Function<T>>>) -> Self {
        let beta1 = T::from_f64(0.9);
        let beta2 = T::from_f64(0.999);
        let epsilon = T::from_f64(1e-8);
        Self::full_new(learning_rate, beta1, beta2, epsilon, architecture)
    }

    pub fn full_new(
        learning_rate: T,
        beta1: T,
        beta2: T,
        epsilon: T,
        architecture: &Vec<Box<dyn Function<T>>>,
    ) -> Self {
        let mut m = vec![];
        let mut v = vec![];
        for lay in architecture {
            if lay.is_linear() {
                let shape = lay.get_data().unwrap().shape();
                m.push(Matrix::zeros(shape));
                v.push(Matrix::zeros(shape));
            }
        }
        let t = vec![0; m.len()];
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m,
            v,
            t: t,
        }
    }
}

impl<T: Float> Optimizer<T> for Adam<T> {
    fn step(&mut self, id: usize, weights: &mut Matrix<T>, gradients: &Matrix<T>) {
        self.t[id] += 1;
        let t = self.t[id];

        self.m[id]
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, x)| {
                let prev_x = *x;
                *x = self.beta1 * prev_x + (T::one() - self.beta1) * gradients.data[i];
            });

        self.v[id]
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, x)| {
                let prev_x = *x;
                *x = self.beta2 * prev_x
                    + (T::one() - self.beta2) * gradients.data[i].powf(T::from_usize(2));
            });
        weights.data.par_iter_mut().enumerate().for_each(|(i, x)| {
            let m_hat = self.m[id].data[i] / (T::one() - self.beta1.powf(T::from_usize(t)));
            let v_hat = self.v[id].data[i] / (T::one() - self.beta2.powf(T::from_usize(t)));

            *x += self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon)
        });
    }
    fn change_learning_rate(&mut self, new_learning_rate: T) {
        self.learning_rate = new_learning_rate;
    }
}

#[cfg(test)]
mod tests {
    use crate::activation::{Function, Sigmoid};
    use crate::linalg::Matrix;
    use crate::loss::SSE;
    use crate::nn::{Linear, Sequential};
    use crate::optim::Adam;
    use crate::{matrix, DataType};

    #[test]
    fn learn_with_adam() {
        let input = matrix![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let output = matrix![[0.0], [1.0], [1.0], [0.0]];

        let layers: Vec<Box<dyn Function<f32>>> = vec![
            Box::new(Linear::new(2, 2, true)),
            Box::new(Sigmoid::new()),
            Box::new(Linear::new(2, 1, true)),
            Box::new(Sigmoid::new()),
        ];
        let mut optim = Adam::new(0.02f32, &layers);
        let mut model = Sequential::new(layers);
        let loss = SSE::new(DataType::f32());
        let mut loss_num = 100f32;
        println!("{}", model.forward(input.clone()));
        for i in 0..10000 {
            if loss_num < 0.001 {
                println!("i:{i} LOSS:{loss_num}");
                break;
            }
            loss_num = model.train(input.clone(), output.clone(), &mut optim, &loss);
            if i % 1000 == 0 {
                println!("{loss_num}");
            }
        }
        println!("{}", model.forward(input));
    }
}
