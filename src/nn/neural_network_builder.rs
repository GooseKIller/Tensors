use crate::activation::Function;
use crate::linalg::Matrix;
use crate::nn::Sequential;
use crate::Float;
use crate::loss::Loss;
use crate::optim::Optimizer;


pub struct NeuralNetworkBuilder<T: Float> {
    layers: Sequential<T>,
    optimizer: Box<dyn Optimizer<T>>,
    loss: Box<dyn Loss<T>>,
}

impl<T: Float> NeuralNetworkBuilder<T> {
    pub fn new(layers: Vec<Box<dyn Function<T>>>,
        optimizer: Box<dyn Optimizer<T>>,
        loss: Box<dyn Loss<T>>
        ) -> Self {
            Self {
                layers: Sequential::new(layers),
                optimizer,
                loss
            }
    }

    pub fn change_optim(&mut self, new_optim: Box<dyn Optimizer<T>>) {
        self.optimizer = new_optim;
    }

    pub fn change_loss(&mut self, new_loss: Box<dyn Loss<T>>) {
        self.loss = new_loss;
    }
}