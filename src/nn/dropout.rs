use crate::{
    Float,
    activation::Module,
    autodiff::{AutoGrad, Var, VarRef},
    linalg::Tensor,
};

/// Randomly zeroes activations while training, and does nothing at inference.
///
/// # Formula
///```math
///  y_i = \begin{cases}
///      \dfrac{x_i}{1 - p}, & \text{with probability } 1 - p \\[6pt]
///      0, & \text{with probability } p
///  \end{cases}
///```
/// Where $`p`$ is the chance of dropping a value
///
/// # Example
/// ```
/// use tensorrs::{linalg::Tensor, activation::Module, nn::Dropout,
///                autodiff::{AutoGrad, Var}};
///
/// let x = Var::leaf(Tensor::from_num(1.0f32, vec![4, 4]), false);
///
/// let mut drop = Dropout::new(0.5);
///
/// // in inference mode the input passes through untouched
/// drop.eval();
/// assert_eq!(drop.forward(&x).value().get_data(), x.value().get_data());
///
/// // in training mode some values are zeroed and the rest are scaled up
/// drop.train();
/// assert!(drop.forward(&x).value().get_data().iter().all(|&v| v == 0.0 || v == 2.0));
/// ```
///
/// # Notes
/// The surviving values are divided by `1 - p`, so the expected sum stays the same
/// and inference needs no correction at all — this is *inverted* dropout, which is
/// what every framework does today.
///
/// A module starts in training mode. Call [Module::eval] before measuring
/// anything: dropout left on during evaluation quietly makes a model look worse
/// than it is.
///
/// # See Also
/// [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)
pub struct Dropout {
    p: f32,
    training: bool,
}

impl Dropout {
    /// Creates a dropout layer that drops a value with probability `p`.
    ///
    /// # Example
    /// ```
    /// use tensorrs::nn::Dropout;
    ///
    /// let drop = Dropout::new(0.2); // keeps four values out of five
    /// assert!(drop.is_training());
    /// ```
    ///
    /// # Arguments
    /// * `p` — the chance of dropping a value, from `0.0` up to but not including `1.0`.
    ///
    /// # Panics
    /// If `p` is outside `[0, 1)`. At `p = 1` every value would be dropped and the
    /// scaling would divide by zero.
    pub fn new(p: f32) -> Self {
        assert!((0.0..1.0).contains(&p),
            "!!!Dropout: p must be in [0, 1), got {p}!!!");

        Self { p, training: true }
    }

    /// Reports whether the layer is currently dropping values.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Returns the chance of dropping a value.
    pub fn p(&self) -> f32 {
        self.p
    }

    /// Starts dropping values again.
    ///
    /// # Notes
    /// The same thing [Module::train] does, but callable without naming the
    /// element type — `drop.train()` rather than `Module::<f32>::train(&mut drop)`.
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Stops dropping values, so the input passes through untouched.
    ///
    /// # Notes
    /// The same thing [Module::eval] does, see [Dropout::train].
    pub fn eval(&mut self) {
        self.training = false;
    }
}

impl<T: Float> Module<T> for Dropout {
    /// Zeroes a random share of the input while training, passes it through otherwise.
    ///
    /// # Arguments
    /// * `x` — the input node; any shape.
    ///
    /// # Returns
    /// In inference mode, `x` itself — no node is added to the graph at all.
    fn forward(&self, x: &VarRef<T>) -> VarRef<T> {
        if !self.training || self.p == 0.0 {
            return x.clone();
        }

        let value = x.value();
        let keep = 1.0 - self.p as f64;
        let scale = T::from_f64(1.0 / keep);

        let mut rng = rand::thread_rng();
        let mask_data: Vec<T> = value
            .packed_data()
            .iter()
            .map(|_| {
                if rand::Rng::gen_bool(&mut rng, keep) {
                    scale
                } else {
                    T::default()
                }
            })
            .collect();

        let mask = Var::leaf(Tensor::new(mask_data, value.get_shape()), false);

        x & &mask
    }

    fn parameters(&self) -> Vec<VarRef<T>> {
        vec![]
    }

    fn train(&mut self) {
        Dropout::train(self);
    }

    fn eval(&mut self) {
        Dropout::eval(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_leaves_the_input_alone() {
        let x = Var::leaf(Tensor::from_num(1.0f64, vec![8, 8]), false);
        let mut drop = Dropout::new(0.5);
        drop.eval();

        assert_eq!(drop.forward(&x).value().get_data(), x.value().get_data());
    }

    #[test]
    fn training_drops_and_rescales() {
        let x = Var::leaf(Tensor::from_num(1.0f64, vec![64, 64]), false);
        let drop = Dropout::new(0.5);

        let out = drop.forward(&x).value().get_data();

        // every value is either dropped or scaled by 1 / (1 - p)
        assert!(out.iter().all(|&v| v == 0.0 || v == 2.0));

        // and roughly half of them survive
        let kept = out.iter().filter(|&&v| v != 0.0).count();
        let share = kept as f64 / out.len() as f64;
        assert!((0.4..0.6).contains(&share), "kept {share} of the values");
    }

    #[test]
    fn the_expected_value_is_preserved() {
        let x = Var::leaf(Tensor::from_num(1.0f64, vec![128, 128]), false);
        let drop = Dropout::new(0.3);

        let out = drop.forward(&x).value().get_data();
        let mean = out.iter().sum::<f64>() / out.len() as f64;

        // that is the whole point of scaling the survivors up
        assert!((mean - 1.0).abs() < 0.05, "mean drifted to {mean}");
    }

    #[test]
    fn the_gradient_follows_the_mask() {
        let x = Var::leaf(Tensor::from_num(1.0f64, vec![4, 4]), true);
        let drop = Dropout::new(0.5);

        let out = drop.forward(&x);
        let values = out.value().get_data();
        out.sum().backward();

        // a dropped value gets no gradient, a kept one gets the scale it was given
        for (value, grad) in values.iter().zip(x.grad().get_data().iter()) {
            let expected = if *value == 0.0 { 0.0 } else { 2.0 };
            assert_eq!(*grad, expected);
        }
    }

    #[test]
    fn a_zero_rate_is_a_no_op() {
        let x = Var::leaf(Tensor::from_num(1.0f64, vec![4, 4]), false);
        let drop = Dropout::new(0.0);

        assert_eq!(drop.forward(&x).value().get_data(), x.value().get_data());
    }
}
