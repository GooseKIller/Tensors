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
    fn step(&mut self, weights: &mut Matrix<T>, gradients: &Matrix<T>) {
        let a = gradients.clone() * self.learning_rate;
        *weights = weights.clone() - &a;
    }
    fn change_learning_rate(&mut self, new_learning_rate: T) {
        self.learning_rate = new_learning_rate;
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::{Matrix, Vector};
    use crate::loss::{Loss, MSE};
    use crate::{DataType, vector};
    use crate::optim::Optimizer;
    use crate::optim::sgd::SGD;

    #[test]
    fn sgd_test() {
        let n = 1000;
        let m = 3;

        //training data
        let mut x = vec![vec![0f64; m]; n];
        let mut y = vec![0f64; n];
        for i in 0..n {
            let i_num = rand::random::<f64>();
            let j_num = rand::random::<f64>();
            x[i] = vec![i_num, 1f64 - i_num, j_num];
            y[i] = 2_f64 * i_num + 5_f64 * (1f64 - i_num) + 8_f64 * j_num;
        }

        // batch size, learning rate, epochs
        let b = 100;
        let alpha = 0.1;
        let e = 500;

        let mut w = Vector::from(vec![0.0f64; m]);
        let answer = vector![3f64, 4f64, -1f64];

        let mse = MSE::new(DataType::f64());
        let mut sgd = SGD::new(alpha);

        for _ in 0..e {
            for i in (b..n).step_by(b) {
                let x_batch = Matrix::from(x[i - b..i].to_vec()); //Matrix::from(x.to_vec());//
                let y_batch = Vector::from(y[i - b..i].to_vec()); //y.to_vec());
                let f = &x_batch * &w;

                let err = mse.gradient(&Matrix::from(f), &Matrix::from(y_batch));
                //let err = f - y_batch;
                let grad = x_batch.transpose() * &Vector::from(err); // * 2f64 * (1f64 / (b as f64));

                let mut w_1 = Matrix::from(w.clone());
                sgd.step(&mut w_1, &Matrix::from(grad));
                w = Vector::from(w_1);

                if (answer.clone() - &w).abs_sum() < 0.01 {
                    break;
                }
            }
        }
        println!("Веса {}", w);
    }
}
