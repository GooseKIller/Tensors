use crate::activation::Function;
use crate::Float;
use crate::linalg::Matrix;

/// Hyperbolic Tangent (Tanh) activation function.
///
/// Maps input values to the range `[-1, 1]`.
///
/// # Mathematical Definition
/// For an input `x`, the Tanh function is defined as:
/// ```math
/// tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}
/// ```
///
/// # Examples
/// ```
/// use tensors::activation::{Function, Tanh};
/// use tensors::linalg::Matrix;
///
/// let tanh = Tanh::new();
/// let input = Matrix::from(vec![vec![0.0], vec![1.0], vec![-1.0]]);
/// let output = tanh.call(input);
/// println!("Tanh output: {}", output);
/// ```
///
/// # See Also
/// - [Wikipedia: Hyperbolic functions](https://en.wikipedia.org/wiki/Hyperbolic_functions)
pub struct Tanh;

impl Tanh {
    /// Creates a new `Tanh` activation function.
    ///
    /// # Returns
    /// A new instance of the `Tanh` activation function.
    pub fn new() -> Self {
        Self
    }

    fn num_fun<T:Float>(&self, num: T) -> T {
        let e_z = num.exp();
        let e_mz = (-num).exp();
        (e_z - e_mz) / (e_z + e_mz)
    }

    fn num_der<T:Float>(&self, num:T) -> T {
        let val = self.num_fun(num);
        T::one() - val * val
    }
}

impl<T:Float> Function<T> for Tanh {
    fn name(&self) -> String {
        "Tanh".to_string()
    }

    fn call(&self, matrix: Matrix<T>) -> Matrix<T> {
        matrix.map(|x| self.num_fun(x))
    }

    fn derivative(&self, matrix: Matrix<T>) -> Matrix<T> {
        matrix.map(|x| self.num_der(x))
    }

}


#[cfg(test)]
mod tests {
    use crate::activation::Function;
    use crate::activation::tanh::Tanh;
    use crate::matrix;
    use crate::linalg::Matrix;

    #[test]
    fn tanh_test() {
        let a = matrix![[1f32,0f32,2f32,3f32]];
        let tanh = Tanh::new();
        println!("{:?} {}",a.shape(), tanh.call(a));
    }

    #[test]
    fn tanh_der() {
        let a = matrix![[1f32,0f32,2f32,3f32]];
        let tanh = Tanh::new();
        println!("{:?} {}",a.shape(), tanh.derivative(a));
    }
}