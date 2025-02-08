use crate::activation::Function;
use crate::linalg::Matrix;
use crate::Float;

/// Rectified Linear Unit (ReLU) activation function.
///
/// Outputs the input directly if it is positive; otherwise, it outputs zero.
///
/// # Mathematical Definition
/// For an input `x`, the ReLU function is defined as:
/// ```math
///  \text{ReLU}(x) = \max(x, 0)
/// ```
/// or
/// ```math
///  \text{ReLU}(x) = \left\{
/// \begin{array}{ll}
/// x & \text{if } x \geq 0 \\
/// 0 & \text{if } x < 0
/// \end{array}
/// \right.
/// ```
///
/// # See Also
/// - [Wikipedia: Rectifier (neural networks)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self { Self }
}

impl<T: Float> Function<T> for ReLU {
    fn name(&self) -> String {
        String::from("ReLU")
    }

    fn call(&self, matrix: Matrix<T>) -> Matrix<T> {
        let [row, cols] = [matrix.rows, matrix.cols];
        let mut data = Vec::with_capacity(row * cols);
        for i in matrix.data {
            let num = if i > T::default() { i } else { T::default() };
            data.push(num);
        }
        Matrix {
            data,
            rows: row,
            cols,
        }
    }

    /// # Derivative of Relu
    ///```math
    ///  ReLU'(x) = \left\{
    /// \begin{array}{ll}
    /// 1 & \text{if } x \geq 0 \\
    /// 0 & \text{if } x < 0
    /// \end{array}
    /// \right.
    ///```
    fn derivative(&self, matrix: Matrix<T>) -> Matrix<T> {
        let [row, cols] = [matrix.rows, matrix.cols];
        let mut data = Vec::with_capacity(row * cols);
        for i in matrix.data {
            let num = if i > T::default() { 1.into() } else { 0.into() };
            data.push(num);
        }
        Matrix {
            data,
            rows: row,
            cols,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::activation::{Function, ReLU};
    use crate::linalg::Matrix;
    use crate::matrix;

    #[test]
    fn relu() {
        let matrix = matrix![[10.0, -10.0]];
        let a = ReLU::new();
        let matrix = a.call(matrix);
        println!("{}", matrix);
    }

    #[test]
    fn derivative_relu() {
        let matrix = matrix![[10.0, -10.0]];
        let a = ReLU::new();
        let matrix = a.derivative(matrix);
        println!("{}", matrix);
    }
}
