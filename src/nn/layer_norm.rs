use crate::{Float, activation::Module, linalg::Tensor, autodiff::{Var, VarRef, sum_axis_op}};

pub struct LayerNorm<T: Float> {
    pub gamma: VarRef<T>,
    pub beta: VarRef<T>,
    pub epsilon: T,
}

impl<T: Float> LayerNorm<T> {
    pub fn new(features: usize, epsilon: T) -> Self {
        let gamma = Var::leaf(Tensor::from_num(T::one(),
        vec![1, features]), true);
        let beta = Var::leaf(Tensor::from_num(T::default(),
         vec![1, features]), true);
        Self { gamma, beta, epsilon }
    }
}

impl<T: Float> Module<T> for LayerNorm<T> {
    fn forward(&self, x: &VarRef<T>) -> VarRef<T> {
        let shape = x.0.borrow().value.shape.clone();
        let ndim = shape.len();
        let axis = ndim - 1; // the last axis
        let features = shape[axis];

        let mean = &sum_axis_op(x, axis, true) / T::from_usize(features);
        let sq = &(x - &mean) ^ T::from_usize(2);
        let var = &sum_axis_op(&sq, axis, true) / T::from_usize(features);
        let denom = &(&var + self.epsilon) ^ T::from_f32(0.5);
        let x_hat = &(x - &mean) / &denom;

        &(&x_hat & &self.gamma) + &self.beta
        /*
        let features = x.0.borrow().value.shape.get(1).cloned().unwrap_or(1);

        let mean = &sum_axis_op(x, 1, true) / T::from_usize(features);

        let sq = &(x - &mean) ^ T::from_usize(2);
        let var = &sum_axis_op(&sq, 1, true) / T::from_usize(features);

        let denom = &(&var + self.epsilon) ^ T::from_f32(0.5);
        let x_hat = &(x - &mean) / &denom;

        &(&x_hat & &self.gamma) + &self.beta
        */
        
    }
    fn parameters(&self) -> Vec<VarRef<T>> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor};

    // A helper for comparing tensors within a tolerance
    fn assert_tensor_approx_eq(t1: &Tensor<f32>, t2: &Tensor<f32>, eps: f32) {
        assert_eq!(t1.get_shape(), t2.get_shape());
        for (a, b) in t1.get_data().iter().zip(t2.get_data().iter()) {
            assert!((a - b).abs() < eps, "{} != {}", a, b);
        }
    }

    #[test]
    fn test_layer_norm_forward_basic() {
        let features = 3;
        let epsilon = 1e-5;
        let layer = LayerNorm::new(features, epsilon);

        let x = Var::leaf(tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], false);

        let output = layer.forward(&x);
        let output_tensor = output.0.borrow().value.clone();

        // The expected result (normalised along the last axis)
        let expected = tensor![[-1.22474487, 0.0, 1.22474487],
                               [-1.22474487, 0.0, 1.22474487]];

        assert_tensor_approx_eq(&output_tensor, &expected, 1e-5);
    }

    #[test]
    fn test_layer_norm_forward_with_gamma_beta() {
        let features = 2;
        let epsilon = 1e-5;
        let layer = LayerNorm::new(features, epsilon);

        // Setting gamma and beta (of shape [1, features])
        layer.gamma.0.borrow_mut().value = tensor![[2.0, 3.0]];
        layer.beta.0.borrow_mut().value = tensor![[0.5, -0.5]];

        let x = Var::leaf(tensor![[1.0, 2.0], [3.0, 4.0]], false);

        let output = layer.forward(&x);
        let output_tensor = output.0.borrow().value.clone();

        // The expected result after the scale and the shift
        let expected = tensor![[-1.5, 2.5],
                               [-1.5, 2.5]];

        assert_tensor_approx_eq(&output_tensor, &expected, 1e-4);
    }

    #[test]
    fn test_layer_norm_forward_single_element() {
        let features = 1;
        let epsilon = 1e-5;
        let layer = LayerNorm::new(features, epsilon);

        let x = Var::leaf(tensor![[2.0], [5.0]], false); // of shape [2, 1]

        let output = layer.forward(&x);
        let output_tensor = output.0.borrow().value.clone();

        // With a single feature x_hat is 0
        let expected = tensor![[0.0], [0.0]];

        assert_tensor_approx_eq(&output_tensor, &expected, 1e-5);
    }

    #[test]
    fn test_layer_norm_parameters() {
        let features = 5;
        let epsilon = 1e-5;
        let layer = LayerNorm::new(features, epsilon);

        let params = layer.parameters();
        assert_eq!(params.len(), 2);

        // Check the shape and the initial values
        let gamma_data = params[0].0.borrow().value.clone();
        let beta_data = params[1].0.borrow().value.clone();

        let expected_gamma = tensor![[1.0, 1.0, 1.0, 1.0, 1.0]];
        let expected_beta = tensor![[0.0, 0.0, 0.0, 0.0, 0.0]];

        assert_tensor_approx_eq(&gamma_data, &expected_gamma, 1e-5);
        assert_tensor_approx_eq(&beta_data, &expected_beta, 1e-5);
    }

    #[test]
    fn test_layer_norm_epsilon_affects_output() {
        let features = 2;
        let epsilon_large = 1.0;
        let layer_large = LayerNorm::new(features, epsilon_large);

        let epsilon_small = 1e-8;
        let layer_small = LayerNorm::new(features, epsilon_small);

        let x = Var::leaf(tensor![[1.0, 2.0], [3.0, 4.0]], false);

        let out_large = layer_large.forward(&x);
        let out_small = layer_small.forward(&x);

        let tensor_large = out_large.0.borrow().value.clone();
        let tensor_small = out_small.0.borrow().value.clone();

        // Check that the results differ
        let mut diff = false;
        for (a, b) in tensor_large.get_data().iter().zip(tensor_small.get_data().iter()) {
            if (a - b).abs() > 1e-6 {
                diff = true;
                break;
            }
        }
        assert!(diff, "Outputs with different epsilon should differ");
    }

    // A further test, covering a range of sizes
    #[test]
    fn test_layer_norm_various_shapes() {
        let shapes = vec![
            (1, 1),   // 2x1
            (2, 4),   // 2x4
            (3, 2),   // 3x2
        ];

        for (batch, features) in shapes {
            let layer = LayerNorm::new(features, 1e-5);
            // Build a tensor of consecutive numbers
            let mut data = Vec::new();
            for i in 0..batch {
                for j in 0..features {
                    data.push((i * features + j) as f32);
                }
            }
            // tensor! cannot be built dynamically, so from_vec it is
            let x = Var::leaf(Tensor::new(data, vec![batch, features]), false);
            let output = layer.forward(&x);
            assert_eq!(output.0.borrow().value.get_shape(), &[batch, features]);
        }
    }
}