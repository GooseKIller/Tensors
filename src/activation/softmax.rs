use crate::Float;
use crate::linalg::Tensor;
use crate::activation::Module;
use crate::utils::{AutoGrad, Var, sum_axis_op};

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
/// use tensorrs::activation::{Function, SoftMax};
/// use tensorrs::linalg::Matrix;
/// use tensorrs::matrix;
///
/// let softmax = SoftMax::new();
/// let input = matrix![[1.0, 2.0, 3.0]];
/// let output = softmax.forward(input);
/// println!("Softmax output: {}", output);
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
    fn forward(&self, x: &crate::utils::VarRef<T>) -> crate::utils::VarRef<T> {
        let e_val = T::f32_f64(std::f32::consts::E, std::f64::consts::E);
        let e = Var::leaf(Tensor::scalar(e_val), false);
        let exp_x = &e ^ x;

        // 2. Считаем сумму экспонент вдоль последней оси (axis = rank - 1)
        // Нам обязательно нужен keepdim=true для последующего деления (broadcasting)
        let axis = x.value().shape.len() - 1;
        let sum_exp = sum_axis_op(&exp_x, axis, true);

        // 3. Вычисляем вероятности: exp(x) / sum(exp(x))
        // Здесь используется твой Div оператор
        &exp_x / &sum_exp    
    }

    fn parameters(&self) -> Vec<crate::utils::VarRef<T>> {
        vec![]
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::*;
    use crate::utils::Var;
    use crate::tensor; // Предполагаю, у тебя есть макрос для создания тензоров

    // Вспомогательная функция для проверки близости значений
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
        // Проверяем, что градиент проходит через нелинейности
        let x = Var::leaf(tensor![[-2.0, 2.0]], true);

        // Тестируем ELU
        let elu = ELU::new(1.0);
        let out = elu.forward(&x);
        out.backward(); 
        
        let grads = x.grad();
        // Для x > 0, производная ELU = 1
        assert_approx(grads.get_data()[1], 1.0);
        // Для x < 0, производная ELU = alpha * e^x = 1.0 * e^(-2.0)
        assert_approx(grads.get_data()[0], (-2.0f32).exp());
        
        x.zero_grad();
    }

    #[test]
    fn test_softmax_logic() {
        // Softmax должен давать в сумме 1.0
        let x = Var::leaf(tensor![[1.0, 2.0, 3.0]], true);
        let sm = SoftMax::new();
        let out = sm.forward(&x);

        let sum: f32 = out.value().get_data().iter().sum();
        assert_approx(sum, 1.0);

        // Проверяем, что самое большое число имеет наибольшую вероятность
        let data = out.value().get_data();
        assert!(data[2] > data[1] && data[1] > data[0]);
    }

    #[test]
    fn test_selu_scaling() {
        let x = Var::leaf(tensor![[1.0]], true);
        let selu = SELU::new(0.0); // Параметры подтянутся из трейта Float
        let out = selu.forward(&x);
        
        // Для x > 0: lambda * x
        // По умолчанию lambda ≈ 1.0507
        assert_approx(out.value().get_data()[0], 1.0507);
    }
}