use rand::distributions::Standard;
use rand::random;
use crate::activation::{Function, Tanh};
use crate::Float;
use crate::linalg::Matrix;

pub struct RNN<T: Float> {
    w_h: Matrix<T>,
    w_x: Matrix<T>,
    bias: Matrix<T>,
    h_prev: Matrix<T>,
}

impl<T:Float> RNN<T>
    where
        Standard: rand::distributions::Distribution<T>{
    pub fn new(input_size: usize, hidden_size: usize, ) -> Self {
        let w_h_limit = (T::from_usize(6) / T::from_usize(2*hidden_size)).sqrt();
        let w_x_limit = (T::from_usize(6) / T::from_usize(input_size + hidden_size)).sqrt();

        let bias = Matrix::from_num(T::default(), 1, hidden_size);
        // Xavier Method
        Self {
            w_h: Matrix::from_num(T::one(), hidden_size, hidden_size)
                .map(|_| random::<T>() * T::from(2) * w_h_limit - w_h_limit),

            w_x: Matrix::from_num(T::one(), input_size, hidden_size)
                .map(|_| random::<T>() * T::from(2) * w_x_limit - w_x_limit),
            h_prev: Matrix::from_num(T::one(), 1, 1),
            bias,
        }
    }

    pub fn forward(&mut self, x_t: &Matrix<T>, h_prev: &Matrix<T>) -> Matrix<T> {
        let h_t =  x_t * &self.w_x + &(h_prev * &self.w_h) + &self.bias;
        let tanh = Tanh::new();
        self.h_prev = tanh.call(h_t);
        self.h_prev.clone()
    }
}

impl<T:Float> Function<T> for RNN<T> {
    fn name(&self) -> String {
        let w = self.w_x.shape();
        let h = self.w_h.shape();
        let b = self.bias.shape();
        format!("RNN: (w_x {} {}) (w_h {} {}) (b {} {})",
                w[0], w[1],
                h[0], h[1],
                b[0], b[1]
        )
    }

    fn call(&self, matrix: Matrix<T>) -> Matrix<T> {
        let h_t =  matrix * &self.w_x + &(&self.h_prev * &self.w_h) + &self.bias;
        let tanh = Tanh::new();
        tanh.call(h_t)
    }

    fn derivative(&self, _matrix: Matrix<T>) -> Matrix<T> {
        todo!()
    }

    fn is_linear(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::activation::Function;
    use crate::matrix;
    use crate::nn::rnn::RNN;
    use crate::linalg::Matrix;

    #[test]
    fn new_rnn() {
        let mut a = RNN::new(3, 5);
        let x_t = matrix![[0.5, 0.3, 0.8]];
        let h_prev = matrix![[0., 0., 0., 0., 0.]];
        println!("{}", a.forward(&x_t, &h_prev));
        println!("{}", a.call(x_t));
    }
}