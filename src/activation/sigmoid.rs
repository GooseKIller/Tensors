use crate::activation::Function;
use crate::linalg::Matrix;
use crate::Float;

/// Exponential Linear Unit (ELU) activation function.
///
/// Maps input values such that:
/// - For `x >= 0`: returns `x`
/// - For `x < 0`: returns `α * (e^x - 1)`
///
/// # Mathematical Definition
/// For an input `x`, the ELU function is defined as:
/// ```math
/// \text{ELU}(x) = \left\{
/// \begin{array}{ll}
/// x & \text{if } x \geq 0 \\
/// \alpha \left( e^x - 1 \right) & \text{if } x < 0
/// \end{array}
/// \right.
/// ```
///
/// # See Also
/// - [Wikipedia: Exponential Linear Unit](https://en.wikipedia.org/wiki/Exponential_linear_unit)

pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }

    fn num_fun<T: Float>(&self, num: T) -> T {
        T::one() / (T::one() + (-num).exp())
    }

    fn num_der<T: Float>(&self, num: T) -> T {
        num.exp() / (num.exp() + T::one()).powf(T::from(2))
    }
}

impl<T: Float> Function<T> for Sigmoid {
    fn name(&self) -> String {
        String::from("Sigmoid")
    }

    fn call(&self, matrix: Matrix<T>) -> Matrix<T> {
        let [row, cols] = [matrix.rows, matrix.cols];
        let mut data = Vec::with_capacity(row * cols);
        for i in matrix.data {
            data.push(self.num_fun(i));
        }
        Matrix {
            data,
            rows: row,
            cols,
        }
    }
    ///# Derivative of Sigmoid
    ///```math
    /// Sigmoid'(x) = \frac{e^x}{(e^x+1)^2}
    /// ```
    fn derivative(&self, matrix: Matrix<T>) -> Matrix<T> {
        let [rows, cols] = [matrix.rows, matrix.cols];
        let mut data = Vec::with_capacity(rows * cols);
        for i in matrix.data {
            data.push(self.num_der(i));
        }
        Matrix { data, rows, cols }
    }
}

#[cfg(test)]
mod tests {
    use crate::activation::{Function, Sigmoid};
    use crate::linalg::Matrix;
    use crate::matrix;

    #[test]
    fn relu() {
        let matrix = matrix![[0.0, -10.0]];
        let a = Sigmoid::new();
        let matrix = a.call(matrix);
        println!("{}", matrix);
    }

    #[test]
    fn derivative_relu() {
        let matrix = matrix![[0.0, -10.0]];
        let a = Sigmoid::new();
        let matrix = a.derivative(matrix);
        println!("{}", matrix);
    }
}
