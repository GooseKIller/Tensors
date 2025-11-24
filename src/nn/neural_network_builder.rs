/*use crate::activation::Function;
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


#[cfg(test)]
mod tests {
    use crate::{activation::{Function, Sigmoid}, loss::MSE, nn::{Linear, neural_network_builder::NeuralNetworkBuilder}, optim::SGD};

    #[test]
    fn some() {
        let layers: Vec<Box< dyn Function<f32>>> = vec![
            Box::new(Linear::new(2, 2, true)),// First layer: Linear transformation
            Box::new(Sigmoid::new()),// Activation function
            Box::new(Linear::new(2, 1, true)),// Second layer: Linear transformation
            Box::new(Sigmoid::new())// Activation function
        ];
        // Create the sequential model
        //let mut model = Sequential::new(layers);
        let sgd = SGD::new(0.01);
        let mse = MSE::new(0.0);
        let _a = NeuralNetworkBuilder::new(layers,
         Box::new(sgd),
          Box::new(mse));
    }
}*/