use rayon::prelude::IntoParallelRefMutIterator;
use rayon::prelude::*;
use crate::Float;
use crate::linalg::Matrix;
use crate::optim::Optimizer;

// WARNING untested
pub struct Adam<T: Float> {
    learning_rate: T,
    beta1: T,
    beta2: T,
    epsilon: T,
    m: Option<Matrix<T>>, // first moment
    v: Option<Matrix<T>>, // second moment
    t: usize, // step
}

impl<T: Float> Adam<T> {
    pub fn new(learning_rate: T, beta1: T, beta2: T, epsilon: T) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m: None,
            v: None,
            t: 0,
        }
    }
}
/// WARNING DO NOT WORK
impl<T:Float> Optimizer<T> for Adam<T>  {
    fn step(&mut self, weights: &mut Matrix<T>, gradients: &Matrix<T>) {
        if self.m.is_none(){
            self.m = Some(Matrix::zeros([1, weights.data.len()]))
        }
        if self.v.is_none(){
            self.v = Some(Matrix::zeros([1, weights.data.len()]))
        }

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        self.t += 1;

        *m = (m.clone() * self.beta1) + &(gradients.clone() * (T::one() - self.beta1));
        let gradients_square= Matrix::new(
            gradients.clone().data.iter().map(|x| x.powf(T::from(2))).collect(),
            gradients.rows,
            gradients.cols
        );


        *v = (v.clone() * self.beta2) + &(gradients_square * (T::one() - self.beta2));

        let m_hat = m.clone() * (T::one()/(T::one() - self.beta1.powf(
            T::from_usize(self.t)
        )));
        let v_hat = v.clone() * (T::one()/(T::one() - self.beta2.powf(
            T::from_usize(self.t)
        )));

        // Обновление весов
        weights
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, weight)| {
                *weight = *weight - (self.learning_rate * m_hat.data[i]) / (v_hat.data[i].sqrt() + self.epsilon);
            });
        /*
        // Инициализация моментов, если они еще не инициализированы
        if self.m.is_none() {
            self.m = Some(Matrix::from(vec![T::default(); weights.data.len()]));
        }
        if self.v.is_none() {
            self.v = Some(Matrix::from(vec![T::default(); weights.data.len()]));
        }

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        // Увеличиваем шаг
        self.t += 1;

        let len = gradients.data.len(); // Длина градиентов
        for i in 0..len {
            m.data[i] = self.beta1 * m.data[i] + (T::one() - self.beta1) * gradients.data[i];
            v.data[i] = self.beta2 * v.data[i] + (T::one() - self.beta2) * gradients.data[i] * gradients.data[i];
        }


        // Коррекция смещения
        let num_beta1 = T::one() / (T::one() - self.beta1.powf(T::from_usize(self.t)));
        let m_hat: Matrix<T> = m.clone() * num_beta1;
        let num_beta1 = T::one() / (T::one() - self.beta2.powf(T::from_usize(self.t)));
        let v_hat: Matrix<T> = v.clone() * num_beta1;

        // Обновление весов
        weights
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, weight)| {
                *weight = *weight - (self.learning_rate * m_hat.data[i]) / (v_hat.data[i].sqrt() + self.epsilon);
            });*/
    }
    fn change_learning_rate(&mut self, new_learning_rate: T) {
        self.learning_rate = new_learning_rate;
    }
}


#[cfg(test)]
mod tests {
    use crate::linalg::{Matrix, Vector};
    use crate::loss::{Loss, MSE};
    use crate::{DataType};
    use crate::optim::adam::Adam;
    use crate::optim::Optimizer;

    #[test]
    fn adam_test() {
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

        let mse = MSE::new(DataType::f64());
        let mut adam = Adam::new(alpha, 0.9, 0.999, 1e-8);

        for _ in 0..e {
            for i in (b..n).step_by(b) {
                let x_batch = Matrix::from(x[i - b..i].to_vec()); //Matrix::from(x.to_vec());//
                let y_batch = Vector::from(y[i - b..i].to_vec()); //y.to_vec());
                let f = &x_batch * &w;

                let loss = mse.call(&Matrix::from(f.clone()), &Matrix::from(y_batch.clone()));

                let err = mse.gradient(&Matrix::from(f), &Matrix::from(y_batch));
                //let err = f - y_batch;
                let grad = x_batch.transpose() * &Vector::from(err); // * 2f64 * (1f64 / (b as f64));

                let mut w_1 = Matrix::from(w.clone());
                adam.step(&mut w_1, &Matrix::from(grad));
                w = Vector::from(w_1);

                if loss.abs() < 0.001 {
                    break;
                }
            }
        }
        println!("Веса {}", w);
    }
}