use crate::{Float, activation::Module};


/// A simple implementation of a Multilayer Perceptron (MLP).
///
/// This struct represents a sequential model composed of multiple layers,
/// where each layer can be a linear transformation followed by an activation function.
///
/// # Example
/// ```
/// use tensorrs::activation::Sigmoid;
/// use tensorrs::nn::{Linear, Sequential};
/// // Define the layers of the MLP
/// let model = Sequential::new(vec![
///             Box::new(Linear::<f32>::new(2, 2, true)),// First layer: Linear transformation
///             Box::new(Sigmoid::new()),// Activation function
///             Box::new(Linear::new(2, 1, true)),// Second layer: Linear transformation
///             Box::new(Sigmoid::new())// Activation function
/// ]);
/// ```
///
/// # Notes
/// `Sequential` implements [`Module`] itself, so a model can be nested inside
/// another model exactly like a single layer.
pub struct Sequential<T:Float> {
    layers: Vec<Box<dyn  Module<T>>>,
}

impl<T: Float> Sequential<T> {
    /// Creates a new Sequential model from a list of layers.
    ///
    /// # Example
    /// ```
    /// use tensorrs::activation::ReLU;
    /// use tensorrs::nn::{Linear, Sequential};
    ///
    /// let model: Sequential<f32> = Sequential::new(vec![
    ///     Box::new(Linear::new(4, 8, true)),
    ///     Box::new(ReLU::new()),
    ///     Box::new(Linear::new(8, 1, true)),
    /// ]);
    /// ```
    ///
    /// # Arguments
    /// * `layers` — the layers, in the order they will be applied.
    pub fn new(layers: Vec<Box<dyn Module<T>>>) -> Self {
        Self { layers }
    }

    /// Appends a layer to the end of the model.
    ///
    /// # Example
    /// ```
    /// use tensorrs::activation::Sigmoid;
    /// use tensorrs::nn::{Linear, Sequential};
    ///
    /// let mut model: Sequential<f32> = Sequential::new(vec![
    ///     Box::new(Linear::new(2, 1, true)),
    /// ]);
    /// model.add(Sigmoid::new());
    /// ```
    ///
    /// # Arguments
    /// * `layer` — the layer to append; it is moved into the model and boxed.
    ///
    /// # Type Parameters
    /// * `F` — the type of the layer, which must implement [`Module<T>`](Module).
    pub fn add<F: Module<T> + 'static>(&mut self, layer: F) {
        self.layers.push(Box::new(layer));
    }
}

impl<T: Float> Module<T> for Sequential<T> {
    /// Runs the input through every layer, in order.
    ///
    /// # Example
    /// ```
    /// use tensorrs::activation::{Module, ReLU};
    /// use tensorrs::autodiff::{AutoGrad, Var};
    /// use tensorrs::nn::{Linear, Sequential};
    /// use tensorrs::tensor;
    ///
    /// let model: Sequential<f32> = Sequential::new(vec![
    ///     Box::new(Linear::new(2, 4, true)),
    ///     Box::new(ReLU::new()),
    ///     Box::new(Linear::new(4, 1, true)),
    /// ]);
    ///
    /// let x = Var::leaf(tensor![[0.0, 1.0], [1.0, 0.0]], false);
    /// let y = model.forward(&x);
    /// assert_eq!(y.value().get_shape(), vec![2, 1]);
    /// ```
    ///
    /// # Arguments
    /// * `x` — the input node of the autodiff graph, of shape `[batch, in_features]`.
    ///
    /// # Returns
    /// The output node. Calling `backward()` on a loss built from it propagates
    /// the gradients back through every layer.
    fn forward(&self, x: &crate::autodiff::VarRef<T>) -> crate::autodiff::VarRef<T> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out
    }
    /// Collects the trainable parameters of every layer.
    ///
    /// # Example
    /// ```
    /// use tensorrs::activation::Module;
    /// use tensorrs::nn::{Linear, Sequential};
    /// use tensorrs::optim::Adam;
    ///
    /// let model: Sequential<f32> = Sequential::new(vec![
    ///     Box::new(Linear::new(2, 1, true)),
    /// ]);
    /// let optim = Adam::new(model.parameters(), 0.1);
    /// ```
    ///
    /// # Returns
    /// The parameters of all layers, flattened and in layer order — ready to be
    /// handed to an [`Optimizer`](crate::optim::Optimizer).
    fn parameters(&self) -> Vec<crate::autodiff::VarRef<T>> {
        let mut params = vec![];
        for layer in &self.layers {
            params = [params, layer.parameters()].concat()
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use crate::{activation::{Module, ReLU, Sigmoid}, loss::*, nn::{Linear, Sequential}, optim::{Adam, Optimizer}, tensor, autodiff::{AutoGrad, Var}};

    #[test]
    fn seq_test() {
        let x_val = tensor![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let y_val = tensor![[0.0], [1.0], [1.0], [0.0]];

        let layers: Sequential<f32> = Sequential::new(vec![
            Box::new(Linear::new(2, 4, true)),
            Box::new(ReLU::new()),
            Box::new(Linear::new(4, 1, true)),
            Box::new(Sigmoid::new()),
        ]);
        let x = Var::leaf(x_val, false);
        let y = Var::leaf(y_val, false);
        
        let mut optim = Adam::new(layers.parameters(), 0.1);
        for i in 0..1000 {
            let y_pred = layers.forward(&x);
            let loss = binary_cross_entropy(&y_pred, &y);

            loss.backward();
            optim.step();
            if i % 100 == 0{
                println!("{i}: {}", y_pred.value());
                println!("{i}: {}", loss.value());
            }
            loss.zero_grad();

            if loss.value().item() < 0.01 {
                println!("Early exit");
                println!("{i}: {}", y_pred.value());
                println!("{i}: {}", loss.value());
                break;
            }
            //break;
        }
    }
}
