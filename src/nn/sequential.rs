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

    /// Will return Loss and train Network
    pub fn train<L: Loss<T>, O: Optimizer<T>>(
        &mut self,
        input: Matrix<T>,
        target: Matrix<T>,
        optimizer: &mut O,
        loss_fn: &L,
    ) -> T{
        let mut outputs = vec![input.clone()];
        for layer in &self.layers {
            let output = layer.call(outputs.last().unwrap().clone());
            outputs.push(output);
        }

        // TODO(Add DGC(Dynamic Gradient Controller)) if is NaN then decrease DGC
        let loss = loss_fn.call(&outputs.last().unwrap(), &target);
        let mut deltas = vec![loss_fn.gradient(&outputs.last().unwrap(), &target)];
        let mut delta = deltas.last().unwrap().clone();
        for (i, layer) in self.layers.iter().enumerate().skip(1).rev(){
            if layer.is_linear(){
                delta = layer.derivative(delta);
            } else {
                let matrix = layer.derivative(outputs[i].clone());
                delta = delta.hadamard(&matrix);
            }
            deltas.push(delta.clone());
        }

        for ((layer, delta), input_data) in self.layers
            .iter_mut().zip(deltas
                .iter().rev()).zip(outputs
                    .iter().take(outputs.len() - 1)) {
            if let Some(mut weight) = layer.get_data() {
                if layer.is_bias() {
                    let mut changed_data = input_data.clone();
                    changed_data.add_column(vec![T::one(); input_data.rows]);
                    optimizer.step(&mut weight, delta, &changed_data);
                }else {
                    optimizer.step(&mut weight, delta, input_data);
                }
                layer.set_data(weight);
            }
        }

        return loss;
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
    use crate::activation::{Function, Sigmoid};
    use crate::linalg::Matrix;
    use crate::loss::{SSE};
    use crate::{DataType, matrix};
    use crate::nn::Linear;
    use crate::nn::sequential::{Sequential};
    use crate::optim::SGD;

    #[test]
    fn learn_xor_test() {
        let input = matrix![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let output = matrix![[0.0], [1.0], [1.0], [0.0]];

        let layers: Vec<Box< dyn Function<f32>>> = vec![
            Box::new(Linear::new(2, 2, true)),
            Box::new(Sigmoid::new()),
            Box::new(Linear::new(2, 1, true)),
            Box::new(Sigmoid::new())
        ];
        let mut model = Sequential::new(layers);
        let mut optim = SGD::new(1f32);
        let loss = SSE::new(DataType::f32());
        println!("{}", model.forward(input.clone()));
        for i in 0..10000{
            let loss = model.train(
                input.clone(),
                output.clone(),
                &mut optim,
                &loss
            );
            if i % 1000 == 0 {
                println!("{loss}");
            }
        }
        println!("{}", model.forward(input));
    }
}