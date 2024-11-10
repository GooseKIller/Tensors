use std::fs::File;
use std::{fs, io};
use serde::{Deserialize, Serialize};
use crate::Float;
use crate::activation::Function;
use crate::linalg::Matrix;
use crate::optim::Optimizer;
use crate::loss::Loss;
use crate::nn::Linear;

#[derive(Serialize, Deserialize)]
struct ModelArchitecture {
    layers: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct LinearLayer {
    data: String,
    rows: usize,
    cols: usize,
    bias: bool,
}

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
    ) {/*

        // forward pass
        let pred = self.forward(input.clone());

        let mut grad = loss_fn.gradient(&pred, &target);

        for layer in self.layers.iter().rev(){
            if layer.is_linear(){
                if layer.is_bias(){

                } else {

                }
            } else {
                grad = grad * &layer.derivative(input.clone());
            }
        }*/

        // forward pass
        let mut activations = vec![input.clone()];
        for layer in &self.layers{
            let output = layer.call(activations.last().unwrap().clone());
            activations.push(output);
        }

        //loss function
        let loss_gradient = loss_fn.gradient(
            activations.last().unwrap(),
            &target
        );


        let mut curr_gradient = loss_gradient;

        for (i, layer) in self.layers.iter_mut().enumerate().rev(){
            let activation = if layer.is_linear() {
                &activations[i]
            } else {
                &layer.derivative(activations[i].clone())
            };

            // 1. Calculate the gradients of the weights
            // Gradients of weights = activation * gradient
            let weight_gradient = activation.clone().transpose() * &curr_gradient; // (num_inputs, num_outputs)

            // Gradient Clipping
            let max_norm = T::one();
            let norm = weight_gradient.clone().norm(T::from(2));
            if norm > max_norm{
                let scaling_factor = max_norm / norm;
                curr_gradient *= scaling_factor;
            }

            let gradient = if layer.is_bias() {
                // 2. Calculate the gradients of the bias
                // bias gradients = the sum of the gradient across the rows
                let bias_gradient = curr_gradient.sum_rows(); // (1, num_outputs)

                Matrix::new(
                    [weight_gradient.data, bias_gradient.data].concat(),
                    weight_gradient.rows + 1,
                    weight_gradient.cols
                )
            } else {
                weight_gradient
            };

            // 3. Updating weights and offsets
            if let Some(mut data) = layer.get_data() {
                optimizer.step(&mut data, &gradient);
                layer.set_data(data);
            }

            // 4. Calculate the gradient for the previous layer
            // Gradient for the previous layer = current gradient * transposed weights
            if let Some(weights) = layer.get_weights() { // (num_inputs, num_outputs)
                curr_gradient = &curr_gradient * &weights.transpose(); // (num_outputs, num_inputs)
            }

        }
    }

    pub fn save(&self, name: &str) -> io::Result<()>{
        fs::create_dir_all(name)?;

        let arch:Vec<String> = self.layers
            .iter()
            .map(|layer| layer.name())
            .collect();
        let architecture = ModelArchitecture{
            layers: arch
        };
        let arch_file_path = format!("{}/architecture.json", name);
        let arch_file = File::create(arch_file_path)?;
        serde_json::to_writer(arch_file, &architecture)?;

        for (i, layer) in self.layers.iter().enumerate(){
            if layer.is_linear() {
                let layer_file_path = format!("{}/layer_{}.json", name, i);
                let layer_file = File::create(layer_file_path)?;

                let data_mx = layer.get_data().unwrap();
                let is_bias = layer.get_bias().is_some();
                let data = LinearLayer{
                    data:data_mx.data_as_string(),
                    rows:data_mx.rows,
                    cols:data_mx.cols,
                    bias:is_bias
                };
                serde_json::to_writer(layer_file, &data)?;
            }
        }
        Ok(())
    }

    pub fn load(&mut self, name:&str) -> io::Result<()>{
        let arch_file_path = format!("{}/architecture.json", name);
        let arch_file = File::open(arch_file_path)?;
        let architecture: ModelArchitecture = serde_json::from_reader(arch_file)?;

        let this_model:Vec<String> = self.layers
            .iter()
            .map(|layer| layer.name())
            .collect();

        for (i, layer_name) in architecture.layers.iter().enumerate() {
            if layer_name != &this_model[i] {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!(
                    "!!!Wrong architecture expected {} got {}!!!",
                    this_model[i], layer_name
                )));
            }
        }

        for i in 0..architecture.layers.len(){
            if architecture.layers[i].starts_with("Linear") {
                let layer_file_path = format!("{}/layer_{}.json", name, i);
                let layer_file = File::open(layer_file_path)?;
                let layer_data: LinearLayer = serde_json::from_reader(layer_file)?;

                let linear_data = Linear {
                    matrix: Matrix::new(
                        layer_data.data.split(" ").map(|x| T::from_str(x)).collect(),
                        layer_data.rows,
                        layer_data.cols
                    ),
                    bias: layer_data.bias
                };
                self.layers[i] = Box::new(linear_data);
            }
        }

        Ok(())
    }

}

#[cfg(test)]
mod test{
    use std::io;
    use crate::activation::{Function, ReLU};
    use crate::linalg::Matrix;
    use crate::loss::{Loss, MSE};
    use crate::{DataType, matrix};
    use crate::nn::Linear;
    use crate::nn::sequential::{Sequential};
    use crate::optim::SGD;


    #[test]
    fn saving_model() -> io::Result<()>{
        let layers: Vec<Box<dyn Function<f64>>> = vec![
            Box::new(Linear::new(1, 64, true)),
            Box::new(ReLU::new()),
            Box::new(Linear::new(64, 1, true)),
        ];
        let model = Sequential::new(layers);

        model.save("model")?;
        println!("{}", model.forward(matrix![[1f64]]));

        Ok(())
    }

    #[test]
    fn loading_model() -> io::Result<()>{
        let layers: Vec<Box<dyn Function<f64>>> = vec![
            Box::new(Linear::new(1, 64, true)),
            Box::new(ReLU::new()),
            Box::new(Linear::new(64, 1, true)),
        ];
        let mut model = Sequential::new(layers);
        model.load("model")?;
        println!("{}", model.forward(matrix![[1f64]]));
        Ok(())
    }

    #[test]
    fn learning_multilayer_nn(){
        let n = 100;
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


        let layers: Vec<Box<dyn Function<f64>>> = vec![
            Box::new(Linear::new(3, 2, true)),
            Box::new(ReLU::new()),
            Box::new(Linear::new(2, 1, true)),
        ];
        let mut model = Sequential::new(layers);

        let loss = MSE::new(DataType::f64());
        let mut optim = SGD::new(0.1);//, 0.9, 0.999, 1e-8);


        let mut min_loss = loss.call(
            &model.forward(Matrix::from(x[0].clone())),
            &Matrix::from_num(y[0],1,1)
        );
        for _ in 0..100{
            for i in 0..n{
                let x_batch = Matrix::from(x[i].clone());
                let y_batch = Matrix::from_num(y[i].clone(),1,1);

                let output = model.forward(x_batch.clone());
                if loss.call(&output, &y_batch) < 0.001{
                    break;
                }

                model.train(
                    x_batch,
                    y_batch,
                    &mut optim,
                    &loss
                )

            }
            min_loss = min_loss.min(loss.call(
                &model.forward(Matrix::from(x[0].clone())),
                &Matrix::from_num(y[0],1,1)
            ));
        }

        println!("MIN Loss{}", min_loss);

        println!("After: {}", model.forward(
            Matrix::from(x[0].clone())
        ));
        println!("EXCEPTED: {}", y[0])
    }


    #[test]
    fn learn(){
        let layers: Vec<Box<dyn Function<f64>>> = vec![
            Box::new(Linear::new(1, 3, true)),
            Box::new(ReLU::new()),
            Box::new(Linear::new(3, 1, true)),
        ];
        let mut model = Sequential::new(layers);
        let mut oprim = SGD::new(0.1);
        let loss_fn = MSE::new(DataType::f64());
        let mut i = 0;
        while loss_fn.call(&model.forward(matrix![[10000.0]]), &matrix![[-125.0]]) > 1e-3 || i != 200{
            i += 1;
            model.train(
                matrix![[1000.0]],
                matrix![[-125.0]],
                &mut oprim,
                &loss_fn
            );
            println!("{}", model.forward(matrix![[1000.0]]));
        }
        println!("{i}");
    }
}