use crate::Float;
use crate::linalg::Tensor;
use crate::activation::Module;
use crate::autodiff::{AutoGrad, Var, sum_axis_op};

/// Softmax function (normalized exponential function).
///
/// Converts a vector of K real numbers into a probability distribution of K possible outcomes.
/// The output values are in the range `[0, 1]` and sum to 1.
///
/// # Mathematical Definition
/// For an input vector `x`, the Softmax function is defined as:
/// ```math
/// \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}
/// ```
///
/// # Examples
/// ```
/// use tensorrs::activation::{Module, SoftMax};
/// use tensorrs::linalg::Matrix;
/// use tensorrs::tensor;
/// use tensorrs::autodiff::{AutoGrad, Var, VarRef};
///
/// let softmax = SoftMax::new();
/// let input = tensor![[1.0, 2.0, 3.0]];
/// let output = softmax.forward(&Var::leaf(input, false));
/// println!("Softmax output: {}", output.value());
/// //[{0.09003057 0.24472848 0.66524094},
/// // {0.090030566 0.24472846 0.66524094}]
/// ```
///
/// # See Also
/// - [Wikipedia: Softmax function](https://en.wikipedia.org/wiki/Softmax_function)
pub struct SoftMax;

impl SoftMax {
    pub fn new() -> Self {
        Self
    }
}

impl<T: Float> Module<T> for SoftMax {
    fn forward(&self, x: &crate::autodiff::VarRef<T>) -> crate::autodiff::VarRef<T> {
        let value = x.value();
        let axis = value.shape.len() - 1;

        // 1. Shift by the maximum along the axis. Softmax is invariant to a
        // shift, and without one e^x already overflows around 90 in f32 - while
        // attention scores QK^T/sqrt(d) reach such values easily.
        //
        // The maximum enters as a constant rather than a node of the graph: the
        // gradient through it cancels out mathematically anyway, but adds noise.
        let mut keepdim_shape = value.get_shape();
        keepdim_shape[axis] = 1;
        let max_per_row = Tensor::new(value.max_axis(axis).get_data(), keepdim_shape);
        let shifted = x - &Var::leaf(max_per_row, false);

        // 2. The exponent is now always <= 0, so the exponential lies in (0, 1]
        let e_val = T::f32_f64(std::f32::consts::E, std::f64::consts::E);
        let e = Var::leaf(Tensor::scalar(e_val), false);
        let exp_x = &e ^ &shifted;

        // 3. The exponentials summed along the same axis, necessarily with
        // keepdim - otherwise the division will not broadcast back
        let sum_exp = sum_axis_op(&exp_x, axis, true);

        &exp_x / &sum_exp
    }

    fn parameters(&self) -> Vec<crate::autodiff::VarRef<T>> {
        vec![]
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::*;
    use crate::autodiff::Var;
    use crate::tensor; // the macro for building tensors

    // A helper for checking that two values are close
    fn assert_approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "Values not equal: {} and {}", a, b);
    }

    #[test]
    fn test_activations_forward() {
        let input_data = tensor![[-1.0, 0.0, 1.0]];
        let x = Var::leaf(input_data, true);

        // 1. ReLU: [-1, 0, 1] -> [0, 0, 1]
        let relu_out = ReLU::new().forward(&x);
        assert_eq!(relu_out.value().get_data(), vec![0.0, 0.0, 1.0]);

        // 2. LeakyReLU (alpha=0.1): [-1, 0, 1] -> [-0.1, 0, 1]
        let lrelu_out = LeakyReLU::new(0.1).forward(&x);
        assert_eq!(lrelu_out.value().get_data(), vec![-0.1, 0.0, 1.0]);

        // 3. Sigmoid: 0.0 -> 0.5
        let sig_out = Sigmoid::new().forward(&x);
        assert_approx(sig_out.value().get_data()[1], 0.5);

        // 4. Tanh: 0.0 -> 0.0
        let tanh_out = Tanh::new().forward(&x);
        assert_approx(tanh_out.value().get_data()[1], 0.0);
    }

    #[test]
    fn test_activations_gradients() {
        // Check that the gradient makes it through the nonlinearities
        let x = Var::leaf(tensor![[-2.0, 2.0]], true);

        // Testing ELU
        let elu = ELU::new(1.0);
        let out = elu.forward(&x);
        out.backward(); 
        
        let grads = x.grad();
        // For x > 0 the derivative of ELU is 1
        assert_approx(grads.get_data()[1], 1.0);
        // For x < 0 the derivative of ELU is alpha * e^x = 1.0 * e^(-2.0)
        assert_approx(grads.get_data()[0], (-2.0f32).exp());
        
        x.zero_grad();
    }

    #[test]
    fn test_softmax_logic() {
        // Softmax has to sum to 1.0
        let x = Var::leaf(tensor![[1.0, 2.0, 3.0]], true);
        let sm = SoftMax::new();
        let out = sm.forward(&x);

        let sum: f32 = out.value().get_data().iter().sum();
        assert_approx(sum, 1.0);

        // Check that the largest number carries the largest probability
        let data = out.value().get_data();
        assert!(data[2] > data[1] && data[1] > data[0]);
    }

    #[test]
    fn test_selu_scaling() {
        let x = Var::leaf(tensor![[1.0]], true);
        let selu = SELU::new(); // the parameters come from the Float trait
        let out = selu.forward(&x);
        
        // For x > 0: lambda * x
        // lambda defaults to about 1.0507
        assert_approx(out.value().get_data()[0], 1.0507);
    }
}