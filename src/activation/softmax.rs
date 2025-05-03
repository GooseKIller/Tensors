use crate::activation::Function;
use crate::linalg::{Matrix, Vector};
use crate::Float;
use rayon::prelude::*;

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
/// let output = softmax.call(input);
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

    fn vec_fun<T: Float>(&self, vector: Vector<T>) -> Vector<T> {
        let max = vector.max_val().unwrap();
        let shifted = vector.map_vec(|x| x - max);
        let sum = shifted.map_vec(|x| x.exp()).sum();
        shifted.map_vec(|x| x.exp() / sum)
    }
}

impl<T: Float> Function<T> for SoftMax {
    fn name(&self) -> String {
        String::from("SoftMax")
    }
    fn call(&self, matrix: Matrix<T>) -> Matrix<T> {
        let mut data: Vec<Vector<T>> = Vec::with_capacity(matrix.rows);
        for i in 0..matrix.rows {
            let vector = self.vec_fun(matrix.get_row(i));
            data.push(vector)
        }
        Matrix::from(data)
    }

    /// $`Softmax'(x_i)= Softmax(x_i) * (δ_{ij} - Softmax(x_j))`$
    ///
    /// $`δ_{ij}`$ - the Kronecker symbol, which is 1 when i = j, and 0 otherwise
    ///
    /// WARNING UNTESTED WELL AND DO NOT WORK NORMAL
    fn derivative(&self, matrix: Matrix<T>) -> Matrix<T> {
        let s = self.call(matrix.clone());
        let size = (matrix.rows, matrix.cols);

        // Параллельное вычисление градиентов для каждой строки
        let grad_rows: Vec<Vector<T>> = (0..size.0)
            .into_par_iter()
            .map(|i| {
                let row = s.get_row(i);
                let diag = row.clone();
                let outer = row.outer(&row);

                //(0..size.1)
                  //  .map(|j| diag[j] - outer[[j, j]])
                    //.collect::<Vector<_>>();
                let mut data = vec![T::default(); size.1];
                data
                    .iter_mut()
                    .enumerate()
                    .for_each(|(j, x)|{
                        *x = diag[j] - outer[[i, j]];
                    });
                Vector::from(
                    data
                )
            })
            .collect();

        Matrix::from(grad_rows)
    }
}

#[cfg(test)]
mod tests {
    use crate::activation::{Function, SoftMax};
    use crate::linalg::Matrix;
    use crate::matrix;

    #[test]
    fn softmax_call() {
        let matrix = matrix![[2.0, 4.0], [1.0, 3.0]];
        let a = SoftMax::new();
        let matrix = a.call(matrix);
        println!("{}", matrix);
    }

    #[test]
    fn der_softmax() {
        let matrix = matrix![[2.0, 4.0], [1.0, 3.0]];
        let a = SoftMax::new();
        let matrix = a.derivative(matrix);
        println!("{}", matrix);
    }

    #[test]
    fn softmax() {
        let matrix: Matrix<f32> = matrix![[0.9, 0.1, 0.8, 0.2]];
        let softmax = SoftMax::new();
        println!("{}", softmax.call(matrix.clone()));
        println!("{}", softmax.derivative(matrix));
    }
}
