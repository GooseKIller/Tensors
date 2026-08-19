use crate::{
    Float,
    activation::Module,
    autodiff::{AutoGrad, VarRef, reshape_op},
};

/// Collapses everything but the batch axis into one.
///
/// # Example
/// ```
/// use tensorrs::{activation::Module, linalg::Tensor, nn::Flatten,
///                autodiff::{AutoGrad, Var}};
///
/// // the output of a convolution: 2 images, 4 channels, 4x4
/// let x = Var::leaf(Tensor::<f32>::randn(vec![2, 4, 4, 4]), false);
///
/// assert_eq!(Flatten.forward(&x).value().get_shape(), vec![2, 64]);
/// ```
///
/// # Notes
/// This is the join between the convolutional part of a model and the fully
/// connected head: [Conv2d](crate::nn::Conv2d) speaks
/// `[batch, channels, height, width]` while [Linear](crate::nn::Linear) wants
/// `[batch, features]`.
///
/// Nothing is computed here — only the shape changes, and the backward pass
/// changes it back.
pub struct Flatten;

impl Flatten {
    /// Creates a `Flatten` layer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> Module<T> for Flatten {
    /// Reshapes `[batch, ...]` into `[batch, features]`.
    ///
    /// # Panics
    /// If the input has no dimensions at all.
    fn forward(&self, x: &VarRef<T>) -> VarRef<T> {
        let shape = x.value().get_shape();
        assert!(!shape.is_empty(), "!!!Flatten got a tensor with no dimensions!!!");

        let batch = shape[0];
        let features: usize = shape[1..].iter().product();

        reshape_op(x, vec![batch, features])
    }

    fn parameters(&self) -> Vec<VarRef<T>> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{autodiff::Var, linalg::Tensor};

    #[test]
    fn keeps_the_batch_axis_and_the_values() {
        let x = Var::leaf(
            Tensor::new((1..=24).map(|v| v as f64).collect(), vec![2, 3, 4]),
            true,
        );

        let flat = Flatten.forward(&x);
        assert_eq!(flat.value().get_shape(), vec![2, 12]);
        assert_eq!(flat.value().get_data(), x.value().get_data());

        // the gradient has to arrive back in the original shape
        flat.sum().backward();
        assert_eq!(x.grad().get_shape(), vec![2, 3, 4]);
        assert!(x.grad().get_data().iter().all(|&v| v == 1.0));
    }
}
