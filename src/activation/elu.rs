use crate::activation::Function;
use crate::linalg::Matrix;
use crate::Float;

/// Exponential Linear Unit
///
/// # Defined as:
///
/// ```math
///   \text{ELU}(x) = \left\{
/// \begin{array}{ll}
/// x & \text{if } x \geq 0 \\
/// \alpha \left( e^x - 1 \right) & \text{if } x < 0
/// \end{array}
/// \right.
/// ```
pub struct ELU<T: Float> {
    alpha: T,
}

impl<T: Float> ELU<T> {
    pub fn new(_datatype_num: T) -> Self {
        Self { alpha: 1.into() }
    }

    pub fn from(num: T) -> Self {
        Self { alpha: num }
    }

    fn elu_num(&self, num: T) -> T {
        let zero = T::default();
        let one: T = 1.into();
        if num > zero {
            num
        } else {
            self.alpha * (num.exp() - one)
        }
    }

    fn elu_der(&self, num: T) -> T {
        let zero = T::default();
        if num > zero {
            1.into()
        } else {
            self.alpha * num.exp()
        }
    }
}

/// sets alpha value
impl<T: Float> From<T> for ELU<T> {
    fn from(value: T) -> Self {
        Self { alpha: value }
    }
}

impl<T: Float> Function<T> for ELU<T> {
    fn name(&self) -> String {
        String::from("ELU")
    }
    fn call(&self, matrix: Matrix<T>) -> Matrix<T> {
        let [row, cols] = [matrix.rows, matrix.cols];
        let mut data = Vec::with_capacity(row * cols);
        for i in matrix.data {
            let num = self.elu_num(i);
            data.push(num);
        }
        Matrix::new(data, row, cols)
    }
    /// # Derivative of ELU
    ///```math
    /// \text{ELU}'(x) = \left\{
    /// \begin{array}{ll}
    /// 1 & \text{if } x \geq 0 \\
    /// \alpha e^x & \text{if } x < 0
    /// \end{array}
    /// \right.
    ///```
    fn derivative(&self, matrix: Matrix<T>) -> Matrix<T> {
        let [row, cols] = [matrix.rows, matrix.cols];
        let mut data = Vec::with_capacity(row * cols);
        for i in matrix.data {
            let num = self.elu_der(i);
            data.push(num);
        }
        Matrix::new(data, row, cols)
    }
}

#[cfg(test)]
mod tests {
    use crate::activation::elu::ELU;
    use crate::activation::Function;
    use crate::linalg::Matrix;
    use crate::{matrix, DataType};

    #[test]
    fn test_elu() {
        let matrix = matrix![[1.0, 0.0]];
        let a = ELU::new(DataType::f64());
        let matrix = a.call(matrix);
        println!("{}", matrix);
    }

    #[test]
    fn derivative_test() {
        let matrix = matrix![[1.0, 0.0]];
        let a = ELU::from(1.0);
        let matrix = a.derivative(matrix);
        println!("{}", matrix);
    }
}
