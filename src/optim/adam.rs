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
    fn step(&mut self, weights: &mut Matrix<T>, gradients: &Matrix<T>, _input_data: &Matrix<T>) {
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
}