use crate::{Float, activation::Module, linalg::Tensor, autodiff::{AutoGrad, Var, VarRef}};

/// Parametric Rectified Linear Unit (PReLU) activation function.
///
/// Maps input values such that:
/// - For `x > 0`: returns `x`
/// - For `x <= 0`: returns `α * x`, where `α` is a **learnable parameter**.
///
/// Unlike Leaky ReLU, where `α` is a fixed hyperparameter, PReLU allows the 
/// neural network to learn the optimal slope for negative values during training.
///
/// # Mathematical Definition
/// For an input `x` and a learnable parameter `α`, the PReLU function is defined as:
/// ```math
/// \text{PReLU}(x) = \left\{
/// \begin{array}{ll}
/// x & \text{if } x > 0 \\
/// \alpha x & \text{if } x \leq 0
/// \end{array}
/// \right.
/// ```
///
/// # Key Features
/// - **Learnable Slope**: The parameter `α` is updated via backpropagation.
/// - **Prevents "Dying ReLU"**: By providing a non-zero gradient for negative values, 
///   it helps keep neurons active.
///
/// # See Also
/// - [Delving Deep into Rectifiers (Original Paper)](https://arxiv.org)
/// - [PyTorch: PReLU Documentation](https://pytorch.org)

pub struct PReLU<T: Float>{
    value: VarRef<T>,
}

impl<T:Float> PReLU<T> {
    pub fn new() -> Self {
        Self {
            value: Var::leaf(Tensor::scalar(T::from_f32(0.25)), true)
        }
    }
}

impl<T:Float> From<T> for PReLU<T> {
    fn from(value: T) -> Self {
        Self {
            value: Var::leaf(Tensor::scalar(value), true)
        }
    }
}

impl<T:Float> Module<T> for PReLU<T>  {
    fn forward(&self, x: &VarRef<T>) -> VarRef<T> {
        let x_val = x.value();

        let m_pos = Var::leaf(
            x_val.map(|v| if v > T::default() { T::one() } else { T::default() }),
            false
        );
        let m_neg = Var::leaf(
            x_val.map(|v| if v <= T::default() { T::one() } else { T::default() }),
            false
        );

        let pos_part = x & &m_pos;

        let neg_part = &(x & &self.value) & &m_neg;

        // 4. Сумма веток: x (если x > 0) + alpha * x (если x <= 0)
        &pos_part + &neg_part
    }
    fn parameters(&self) -> Vec<VarRef<T>> {
        vec![self.value.clone()]
    }
}


#[cfg(test)]
mod tests {
    use crate::{activation::{Module, PReLU}, loss::mse, nn::{Linear, Sequential}, optim::{Adam, Optimizer}, tensor, autodiff::{AutoGrad, Var}};

    #[test]
    fn one_layer_xor() {
        let model = Sequential::new(vec![
            Box::new(Linear::<f32>::new(2, 1, false)),
            Box::new(PReLU::new()),
        ]);

        let x_val = tensor![[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let y_val = tensor![[0.0], [1.0], [1.0], [0.0]];

        let x = Var::leaf(x_val, false);
        let y = Var::leaf(y_val, false);

        let mut optim = Adam::new(model.parameters(), 0.1);
        for i in 0..300 {
            optim.zero_grad();
            let output = model.forward(&(&(&x * 2.0) - 1.0));
            let loss = mse(&output, &y);

            let value = loss.value().item();
            if value.is_nan() || value < 0.05 {
                println!("Early stop {i}: {value}");
                break;
            }
            println!("i: {i}, {}", value);
            loss.backward();
            optim.clip_grad(5.0);
            optim.step();
        }

        println!("Printing model weights:");
        for i in model.parameters() {
            println!("{}", i);
        }
    }
}