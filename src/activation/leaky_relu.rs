use crate::activation::Function;
use crate::linalg::Matrix;
use crate::Float;

/// Leaky ReLU activation function.
///
/// Allows a small, non-zero gradient when the input is negative.
///
/// # Mathematical Definition
/// For an input `x`, the Leaky ReLU function is defined as:
/// ```math
///  \text{LeakyReLU}(x) = \max(x, \alpha x)
/// ```
/// or
/// ```math
///  \text{LeakyReLU}(x) = \left\{
/// \begin{array}{ll}
/// x & \text{if } x \geq 0 \\
/// \alpha x & \text{otherwise}
/// \end{array}
/// \right.
/// ```
///
/// # See Also
/// - [velog.io: Leaky ReLU](https://velog.io/@greensox284/Neural-Leaky-Rectified-Linear-Unit-Leaky-ReLU)

pub struct LeakyReLU<T: Float> {
    alpha: T,
}

impl<T: Float> LeakyReLU<T> {
    pub fn new(_: T) -> Self {
        let two = T::from_usize(2);
        let ten = T::from_usize(10);
        Self { alpha: two / ten }
    }

    fn leaky_num(&self, num: T) -> T {
        let zero = T::default();
        if num > zero {
            num
        } else {
            num * self.alpha
        }
    }

    fn leaky_der(&self, num: T) -> T {
        let zero = T::default();
        if num > zero {
            1.into()
        } else {
            self.alpha
        }
    }
}

impl<T: Float> From<T> for LeakyReLU<T> {
    fn from(value: T) -> Self {
        Self { alpha: value }
    }
}

impl<T: Float> Function<T> for LeakyReLU<T> {
    fn name(&self) -> String {
        String::from("LeakyReLU")
    }
    fn call(&self, matrix: Matrix<T>) -> Matrix<T> {
        let [row, cols] = [matrix.rows, matrix.cols];
        let mut data = Vec::with_capacity(row * cols);
        for i in matrix.data {
            let num = self.leaky_num(i);
            data.push(num);
        }
        Matrix::new(data, row, cols)
    }
    /// # Derivative of LeakyReLU
    /// ```math
    /// LeakyReLU'(x) = \left\{
    /// \begin{array}{ll}
    /// 1 & \text{if } x \geq 0 \\
    /// \alpha & \text{otherwise}
    /// \end{array}
    /// \right.
    /// ```
    fn derivative(&self, matrix: Matrix<T>) -> Matrix<T> {
        let [row, cols] = [matrix.rows, matrix.cols];
        let mut data = Vec::with_capacity(row * cols);
        for i in matrix.data {
            let num = self.leaky_der(i);
            data.push(num);
        }
        Matrix::new(data, row, cols)
    }
}

#[cfg(test)]
mod tests {
    use crate::activation::leaky_relu::LeakyReLU;
    use crate::activation::Function;
    use crate::linalg::Matrix;
    use crate::{matrix, DataType};

    #[test]
    fn leaky_relu() {
        let matrix = matrix![[10.0, -10.0]];
        let a = LeakyReLU::new(DataType::f64());
        let matrix = a.call(matrix);
        println!("{}", matrix);
    }

    #[test]
    fn derivative_leaky_relu() {
        let matrix = matrix![[10.0, -10.0]];
        let a = LeakyReLU::from(0.2);
        let matrix = a.derivative(matrix);
        println!("{}", matrix);
    }
}
