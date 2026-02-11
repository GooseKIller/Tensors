use crate::{Float, activation::Module, linalg::Tensor, utils::Var};

/// Sigmoid activation function.
///
/// Maps input values into the range (0, 1) using the logistic function.
///
/// # Mathematical Definition
/// For an input `x`, the Sigmoid function is defined as:
/// ```math
/// Sigmoid(x) = \frac{1}{1 + e^{-x}}
/// ```
///
/// # See Also
/// - [Wikipedia: Logistic function](https://en.wikipedia.org/wiki/Logistic_function)
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl<T: Float> Module<T> for Sigmoid {
    fn forward(&self, x: &crate::utils::VarRef<T>) -> crate::utils::VarRef<T> {
        // Формула: 1 / (1 + exp(-x))
        
        let e_val = T::f32_f64(std::f32::consts::E, std::f64::consts::E);
        let e = Var::leaf(Tensor::scalar(e_val), false);

        // 2. Считаем знаменатель: 1 + e^(-x)
        let exp_neg_x = &e ^ &-x;
        let denom = &exp_neg_x + T::one();

        // 3. Результат: 1 / denom
        // Используем твой div_op (оператор /)
        let one = Var::leaf(Tensor::scalar(T::one()), false);
        &one / &denom
    }

    fn parameters(&self) -> Vec<crate::utils::VarRef<T>> {
        vec![]
    }
}