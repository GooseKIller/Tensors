use crate::activation::Function;
use crate::linalg::Matrix;
use crate::loss::Loss;
use crate::nn::Linear;
use crate::optim::Optimizer;
use crate::Float;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::ops::Index;
use std::{fs, io};

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

/// A simple implementation of a Multilayer Perceptron (MLP).
///
/// This struct represents a sequential model composed of multiple layers,
/// where each layer can be a linear transformation followed by an activation function.
/// # Example
/// ```
/// use tensorrs::activation::{Function, Sigmoid};
/// use tensorrs::nn::{Linear, Sequential};
/// // Define the layers of the MLP
/// let layers: Vec<Box< dyn Function<f32>>> = vec![
///             Box::new(Linear::new(2, 2, true)),// First layer: Linear transformation
///             Box::new(Sigmoid::new()),// Activation function
///             Box::new(Linear::new(2, 1, true)),// Second layer: Linear transformation
///             Box::new(Sigmoid::new())// Activation function
/// ];
/// // Create the sequential model
/// let mut model = Sequential::new(layers);
/// ```
pub struct Sequential<T: Float> {
    layers: Vec<Box<dyn Function<T>>>,
    clip_threshold: T,
}

impl<T: Float> Sequential<T> {
    /// Creates a new Sequential model with the given layers.
    ///
    /// # Arguments
    /// * `layers` - A vector(`Vec<Box<dyn Function<T>>>`) of layers to be included in the model.
    pub fn new(layers: Vec<Box<dyn Function<T>>>) -> Self {
        Self { layers, clip_threshold: T::from(5) }
    }

    /// Sets a new value for the gradient clipping threshold.
    ///
    /// By default, the clipping
    /// threshold is set to 5.0.
    pub fn set_clip_threshold(&mut self, new_value: T) {
        self.clip_threshold = new_value;
    }

    pub fn layers(&self) -> Vec<&Box<dyn Function<T>>> {
        let mut layers_copy = vec![];
        for i in &self.layers {
            layers_copy.push(i);
        }
        layers_copy
    }

    /// Adds a new layer to the Sequential model.
    ///
    /// This method takes ownership of the layer and stores it in the model.
    ///
    /// # Arguments
    /// * `layer` - A layer that implements the `Function<T>` trait.
    ///
    /// # Type Parameters
    /// * `F` - The type of the layer being added, which must implement `Function<T>`.
    ///
    pub fn add<F: Function<T> + 'static>(&mut self, layer: F) {
        self.layers.push(Box::new(layer));
    }

    /// Forward pass through the model.
    ///
    /// # Arguments
    /// * `input` - The input matrix to be processed through the model.
    ///
    /// # Returns
    /// The output tensor after passing through all layers.
    ///
    pub fn forward(&self, input: Matrix<T>) -> Matrix<T> {
        let mut output = input;
        for layer in &self.layers {
            output = layer.call(output);
        }
        output
    }

    /// Trains the neural network for one step using the specified optimizer and computes the loss.
    ///
    /// # Parameters:
    /// - `input`: A matrix representing the input batch for the training step.
    /// - `target`: A matrix representing the expected output batch for the input data.
    /// - `optimizer`: A mutable reference to an optimizer.
    /// - `loss_fn`: A reference to a loss function.
    ///
    /// # Returns:
    /// - Returns the computed loss value for the current training step.
    /// # Example
    ///```
    /// use tensorrs::activation::{Function, Sigmoid};
    /// use tensorrs::{DataType, matrix};
    /// use tensorrs::linalg::Matrix;
    /// use tensorrs::loss::SSE;
    /// use tensorrs::nn::{Linear, Sequential};
    /// use tensorrs::optim::SGD;
    ///
    /// let input = matrix![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    /// let output = matrix![[0.0], [1.0], [1.0], [0.0]];
    ///
    /// let layers: Vec<Box< dyn Function<f32>>> = vec![
    ///             Box::new(Linear::new(2, 2, true)),
    ///             Box::new(Sigmoid::new()),
    ///             Box::new(Linear::new(2, 1, true)),
    ///             Box::new(Sigmoid::new())
    /// ];
    /// let mut model = Sequential::new(layers);
    /// let mut optim = SGD::new(1f32);
    /// let loss = SSE::new(DataType::f32());
    /// for i in 0..10000{
    ///    let loss = model.train(
    ///        input.clone(),
    ///        output.clone(),
    ///        &mut optim,
    ///        &loss
    ///    );
    ///    if i % 1000 == 0 {
    ///        println!("{loss}");
    ///    }
    /// }
    /// println!("{}", model.forward(input));
    ///```
    pub fn train<L: Loss<T>, O: Optimizer<T>>(
        &mut self,
        input: Matrix<T>,
        target: Matrix<T>,
        optimizer: &mut O,
        loss_fn: &L,
    ) -> T {
        let mut outputs = vec![input.clone()];
        for layer in &self.layers {
            let output = layer.call(outputs.last().unwrap().clone());
            outputs.push(output);
        }

        let loss = loss_fn.call(&outputs.last().unwrap(), &target);
        let mut deltas = vec![loss_fn.gradient(&outputs.last().unwrap(), &target)];
        let mut delta = deltas.last().unwrap().clone();
        for (i, layer) in self.layers.iter().enumerate().skip(1).rev() {
            if layer.is_linear() {
                delta = layer.derivative(delta);
            } else {
                let matrix = layer.derivative(outputs[i].clone());
                delta = delta.hadamard(&matrix);
            }
            deltas.push(delta.clone());
        }

        let mut i = 0;
        for ((layer, delta), input_data) in self
            .layers
            .iter_mut()
            .zip(deltas.iter().rev())
            .zip(outputs.iter().take(outputs.len() - 1))
        {
            if let Some(mut weight) = layer.get_data() {
                let mut grads = if layer.is_bias() {
                    let mut changed_data = input_data.clone();
                    changed_data.add_column(vec![T::one(); input_data.rows]);
                    changed_data.transpose() * delta
                } else {
                    input_data.transpose() * delta
                };

                // gradient clipping
                let grad_norm = grads.norm(T::from(2));
                if grad_norm > self.clip_threshold {
                    let scale = self.clip_threshold / grad_norm;
                    grads *= scale;
                }

                optimizer.step(i, &mut weight, &grads);
                i += 1;
                layer.set_data(weight);
            }
        }

        return loss;
    }

    pub fn save(&self, name: &str) -> io::Result<()> {
        fs::create_dir_all(name)?;

        let arch: Vec<String> = self.layers.iter().map(|layer| layer.name()).collect();
        let architecture = ModelArchitecture { layers: arch };
        let arch_file_path = format!("{}/architecture.json", name);
        let arch_file = File::create(arch_file_path)?;
        serde_json::to_writer(arch_file, &architecture)?;

        for (i, layer) in self.layers.iter().enumerate() {
            if layer.is_linear() {
                let layer_file_path = format!("{}/layer_{}.json", name, i);
                let layer_file = File::create(layer_file_path)?;

                let data_mx = layer.get_data().unwrap();
                let is_bias = layer.get_bias().is_some();
                let data = LinearLayer {
                    data: data_mx.data_as_string(),
                    rows: data_mx.rows,
                    cols: data_mx.cols,
                    bias: is_bias,
                };
                serde_json::to_writer(layer_file, &data)?;
            }
        }
        Ok(())
    }

    pub fn load(&mut self, name: &str) -> io::Result<()> {
        let arch_file_path = format!("{}/architecture.json", name);
        let arch_file = File::open(arch_file_path)?;
        let architecture: ModelArchitecture = serde_json::from_reader(arch_file)?;

        let this_model: Vec<String> = self.layers.iter().map(|layer| layer.name()).collect();

        for (i, layer_name) in architecture.layers.iter().enumerate() {
            if layer_name != &this_model[i] {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "!!!Wrong architecture expected {} got {}!!!",
                        this_model[i], layer_name
                    ),
                ));
            }
        }

        for i in 0..architecture.layers.len() {
            if architecture.layers[i].starts_with("Linear") {
                let layer_file_path = format!("{}/layer_{}.json", name, i);
                let layer_file = File::open(layer_file_path)?;
                let layer_data: LinearLayer = serde_json::from_reader(layer_file)?;

                let linear_data = Linear {
                    matrix: Matrix::new(
                        layer_data.data.split(" ").map(|x| T::from_str(x)).collect(),
                        layer_data.rows,
                        layer_data.cols,
                    ),
                    bias: layer_data.bias,
                };
                self.layers[i] = Box::new(linear_data);
            }
        }

        Ok(())
    }
}

impl<T: Float> Index<usize> for Sequential<T> {
    type Output = Box<dyn Function<T>>;
    fn index(&self, index: usize) -> &Box<dyn Function<T>> {
        &self.layers[index]
    }
}

#[cfg(test)]
mod test {
    use crate::activation::{Function, Sigmoid, Tanh};
    use crate::linalg::{Matrix, Vector};
    use crate::loss::{CrossEntropy, MSE, SSE};
    use crate::nn::sequential::Sequential;
    use crate::nn::Linear;
    use crate::optim::{Adam, SGD};
    use crate::{matrix, DataType};
    use crate::utils::one_hot_encoding;

    #[test]
    fn learn_xor_test() {
        let input = matrix![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let output = matrix![[0.0], [1.0], [1.0], [0.0]];

        let layers: Vec<Box<dyn Function<f32>>> = vec![
            Box::new(Linear::new(2, 2, true)),
            Box::new(Sigmoid::new()),
            Box::new(Linear::new(2, 1, true)),
            Box::new(Sigmoid::new()),
        ];
        let mut model = Sequential::new(layers);
        let mut optim = SGD::new(1f32);
        let loss = SSE::new(DataType::f32());
        for i in 0..10000 {
            let loss = model.train(input.clone(), output.clone(), &mut optim, &loss);
            if i % 1000 == 0 {
                println!("{loss}");
            }
        }
        println!("{}", model.forward(input));
    }

    #[test]
    fn regresion() {
        let x = Matrix::from(Vector::range(100.0, 105.0, 0.125).unwrap());
        let f = |x| 30.0 * x - 6.0;
        let y = x.map(f);

        let layers: Vec<Box<dyn Function<f32>>> = vec![Box::new(Linear::new(1, 1, true))];
        let mut optim = Adam::new(0.1, &layers);
        let mut model = Sequential::new(layers);
        let loss = MSE::new(DataType::f32());
        for i in 0..1000 {
            let loss_num = model.train(x.transpose(), y.transpose(), &mut optim, &loss);
            if i % 100 == 0 {
                println!("{loss_num}");
            }
        }
        let num = 435.0;
        println!(
            "REAL VAL: {}\nMODEL VAL: {}",
            f(num),
            model.forward(matrix![[num]])
        );
        println!("{}", model[0].get_data().unwrap());
    }

    #[test]
    fn classification() {
        let x = Matrix::from(Vector::range(0., 10., 0.1).unwrap());
        let f = |x: f32| if x < 5. {0.} else { 1. };
        let y = x.map(f);

        let layers: Vec<Box<dyn Function<f32>>> = vec![
            Box::new(Linear::new(1, 1, true)), // Линейный слой с 1 входом и 2 выходами (для двух классов)
            Box::new(Tanh::new()),         // Функция активации Softmax для классификации
        ];

        let mut optim = Adam::new(0.001, &layers);
        let mut model = Sequential::new(layers);
        let loss = CrossEntropy::new(DataType::f32());

        let _y_data = Matrix::from(
            one_hot_encoding(
                y.get_data()
                .iter().
                map(|x| *x as usize)
                .collect::<Vec<usize>>(),
            2,
                0.0f32)
        );

        for i in 0..10000 {
            let loss_num = model.train(
                x.transpose(),
                y.transpose(),
                &mut optim,
                &loss
            );
            if i % 100 == 0 {
                println!("Loss: {}", loss_num);
            }
        }

        let test_values = vec![2.0, 4.0, 6.0, 8.0];
        for &num in &test_values {
            let prediction = model.forward(matrix![[num]]);
            println!("{}", prediction);
            /*
            let preds = Vector::from(prediction);
            println!("Input: {}, Predicted class: {}", num, preds.argmax());
             */
            println!("Input: {}, Predicted class: {}",
                     num,
                     prediction.compare_num(0.5))
        }
        println!("{}", model[0].get_data().unwrap())

    }
}
