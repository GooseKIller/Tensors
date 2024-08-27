use crate::Float;
use crate::activation::Function;
use crate::linalg::Matrix;
use crate::optim::Optimizer;
use crate::loss::Loss;
use crate::nn::Linear;

pub struct Sequential<T:Float>{
    layers: Vec<Box<dyn Function<T>>>
}

impl<T:Float> Sequential<T> {
    pub fn new(layers: Vec<Box<dyn Function<T>>>) -> Self{
        Self{
            layers,
        }
    }

    pub fn add<F: Function<T> + 'static>(&mut self, layer: F) {
        self.layers.push(Box::new(layer));
    }

    pub fn forward(&self, input:Matrix<T>) -> Matrix<T>{
        let mut output = input;
        for layer in &self.layers{
            output = layer.call(output);
        }
        output
    }

    pub fn train<L: Loss<T>, O: Optimizer<T>>(
        &mut self,
        input: Matrix<T>,
        target: Matrix<T>,
        optimizer: &mut O,
        loss_fn: &L,
    ) {

        // forward prop
        let mut activations = vec![input.clone()];
        for layer in &self.layers{
            let output = layer.call(activations.last().unwrap().clone());
            activations.push(output);
        }

        // gradient loss calc
        let loss_gradient = loss_fn.gradient(&activations.last().unwrap(), &target);

        // back propagation
        let mut deltas = vec![loss_gradient];
        for i in (0..self.layers.len()).rev(){
            if self.layers[i].is_linear(){
                let delta = self.layers[i].derivative(
                    deltas.last().unwrap().clone()
                );
                deltas.push(delta)
            }


            //deltas.push(delta);
        }
        deltas.reverse();

        //Updating weights
        for (i, layer) in self.layers.iter_mut().enumerate(){
            let gradient = activations[i].clone().transpose() * &deltas[i+1].clone();

            if let Some(mut data) = layer.get_data() {
                optimizer.step(&mut data, &gradient);
                layer.set_data(data);
            }
        }
    }

}

#[cfg(test)]
mod test{
    use crate::activation::{Function, ReLU};
    use crate::linalg::Matrix;
    use crate::loss::{Loss, MSE};
    use crate::{DataType, matrix};
    use crate::nn::Linear;
    use crate::nn::sequential::Sequential;
    use crate::optim::SGD;

    #[test]
    fn learning_multilayer_nn(){
        let layers: Vec<Box<dyn Function<f64>>> = vec![
            Box::new(Linear::new(3, 2, true)),
            Box::new(ReLU::new()),
            Box::new(Linear::new(2, 1, true)),
        ];
        /*
        let layers: Vec<Box<dyn Function<f64>>> = vec![
            Box::new(Linear::from(Matrix::from_num(1.0, 3, 2))),
            Box::new(ReLU::new()),
            Box::new(Linear::from(Matrix::from_num(1.0, 2, 1))),
        ];*/
        let mut seq = Sequential::new(layers);

        let inp = matrix![[0.1, 0.2, 0.3]];
        let out = matrix![[1.0]];
        //println!("Before: {}", seq.forward(inp.clone()));

        let loss = MSE::new(DataType::f64());
        let mut optim = SGD::new(0.1);//, 0.9, 0.999, 1e-8);

        for i in 0..1000{
            if loss.call(&seq.forward(inp.clone()), &out.clone()) < 0.000001{
                println!("LOSS:{}, i: {}", loss.call(&seq.forward(inp.clone()), &out), i);
                break;
            }
            seq.train(inp.clone(), out.clone(), &mut optim, &loss);
        }
        println!("After: {}", seq.forward(inp.clone()));
    }
}