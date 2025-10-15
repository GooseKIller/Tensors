use crate::activation::Function;
use crate::Float;
use crate::linalg::Matrix;
use crate::optim::Optimizer;
use rayon::prelude::IntoParallelRefMutIterator;
use rayon::prelude::*;

pub struct  RMSprop<T:Float> {
    learning_rate: T,
    decay: T,
    epsilon: T,
    cache: Vec<Matrix<T>>,
}

impl<T:Float> RMSprop<T> {
    pub fn new(learning_rate: T, architecture: &Vec<Box<dyn Function<T>>>) -> Self {
        let decay = T::from_f64(0.9);
        let epsilon = T::from_f64(1e-8);
        let mut cache = vec![];

        for layer in architecture {
            if layer.is_linear() {
                let shape = layer.get_data().unwrap().shape();
                cache.push(Matrix::zeros(shape))
            }
        }

        Self {
            learning_rate,
            decay,
            epsilon,
            cache,
        }
    }
}

impl<T: Float> Optimizer<T> for RMSprop<T> {
    fn step(&mut self, id: usize, weights: &mut Matrix<T>, gradients: &Matrix<T>) {
        let decay = self.decay;
        let epsilon = self.epsilon;
        let lr = self.learning_rate;

        self.cache[id]
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, v)| {
                let grad = gradients.data[i];
                *v = decay * *v + (T::one() - decay) * grad * grad;
            });

        weights
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, w)| {
                *w -= lr * gradients.data[i] / (self.cache[id].data[i].sqrt() + epsilon);
            });

    }
    fn change_learning_rate(&mut self, new_learning_rate: T) {
        self.learning_rate = new_learning_rate;
    }
}


#[cfg(test)]
mod tests {
    use crate::activation::{Function, Sigmoid};
    use crate::loss::SSE;
    use crate::{DataType, matrix};
    use crate::linalg::Matrix;
    use crate::nn::{Linear, Sequential};
    use crate::optim::RMSprop;

    #[test]
    fn learn_with_rmsprop() {
        let input = matrix![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let output = matrix![[0.0], [1.0], [1.0], [0.0]];

        let layers: Vec<Box<dyn Function<f32>>> = vec![
            Box::new(Linear::new(2, 2, true)),
            Box::new(Sigmoid::new()),
            Box::new(Linear::new(2, 1, true)),
            Box::new(Sigmoid::new()),
        ];
        let mut optim = RMSprop ::new(0.002f32, &layers);
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