use crate::activation::Function;
use crate::Float;
use crate::linalg::{Matrix, Vector};

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
/// use tensors::activation::{Function, SoftMax};
/// use tensors::linalg::Matrix;
/// use tensors::matrix;
///
/// let softmax = SoftMax::new();
/// let input = matrix![[1.0, 2.0, 3.0]];
/// let output = softmax.call(input);
/// println!("Softmax output: {}", output);
/// //[{0.09003057 0.24472848 0.66524094},
/// // {0.090030566 0.24472846 0.66524094}]
/// ```
///
/// # See Also
/// - [Wikipedia: Softmax function](https://en.wikipedia.org/wiki/Softmax_function)
pub struct SoftMax;

impl SoftMax{
    pub fn new() -> Self {Self}

    fn vec_fun<T:Float>(&self, vector: Vector<T>) -> Vector<T>{
        let mut sum:T = T::default();
        let vector:Vec<T> = vector.into();
        for i in &vector{
            sum += i.exp();
        }
        let mut answer = Vector::from_num(T::default(),vector.len());
        for i in 0..vector.len(){
            answer[i] = vector[i].exp()/sum;
        }
        answer
    }
}

impl<T:Float> Function<T> for SoftMax {
    fn name(&self) -> String {
        String::from("SoftMax")
    }
    fn call(&self, matrix: Matrix<T>) -> Matrix<T> {
        let mut data:Vec<Vector<T>> = Vec::with_capacity(matrix.rows);
        for i in 0..matrix.rows{
            let vector = matrix.get_row(i);
            let vector = self.vec_fun(vector);
            data.push(vector)
        }
        Matrix::from(data)

    }

    /// $`Softmax'(x_i)= Softmax(x_i) * (δ_{ij} - Softmax(x_j))`$
    ///
    /// $`δ_{ij}`$ - the Kronecker symbol, which is 1 when i = j, and 0 otherwise
    fn derivative(&self, matrix: Matrix<T>) -> Matrix<T> {
        let [row, cols] = matrix.shape();
        let softmax = self.call(matrix.clone());
        let identity_minus_softmax = Matrix::identity(T::default(), row, cols) - &softmax;
        softmax.hadamard(&identity_minus_softmax).transpose() * &matrix
    }
}

#[cfg(test)]
mod tests{
    use crate::activation::{Function, SoftMax};
    use crate::matrix;
    use crate::linalg::Matrix;

    #[test]
    fn softmax_cqll(){
        let matrix = matrix![[2.0, 4.0],
                                        [1.0, 3.0]];
        let a = SoftMax::new();
        let matrix = a.call(matrix);
        println!("{}", matrix);
    }

    #[test]
    fn der_softmax(){
        let matrix = matrix![[2.0, 4.0],
                                        [1.0, 3.0]];
        let a = SoftMax::new();
        let matrix = a.derivative(matrix);
        println!("{}", matrix);
    }

    #[test]
    fn softmax() {
        let matrix:Matrix<f32> = matrix![[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]];
        let softmax = SoftMax::new();
        println!("{}", softmax.call(matrix));
    }
}