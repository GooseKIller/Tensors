use crate::Float;
use crate::linalg::Matrix;
use crate::loss::Loss;

/// Mean absolute percentage error
///
///  # Formula:
///```math
///  MAPE(ŷ, y) = \frac{1}{n} * \sum_{i=1}^{n} \left| \frac{ŷ_i - y_i}{y_i} \right|
///```
///
/// Where $`ŷ_i`$ predicted and $`y_i`$ expected value
pub struct MAPE<T:Float>(T);

impl<T:Float> MAPE<T> {
    pub fn new(datatype_number: T) -> Self{
        Self(datatype_number)
    }
}

impl<T:Float> Loss<T> for MAPE<T> {
    fn call(&self, output: &Matrix<T>, target: &Matrix<T>) -> T {
        if output.size() != target.size() {
            panic!("!!!Size of output matrix and target must be equal!!!\nOutput size:{:?} Target size: {:?}", output.size(), target.size())
        }
        let length = output.data.len();
        let diff = output - target;
        let mut total_loss = T::default();
        for i in 0..length{
            if target.data[i] != T::default(){
                total_loss += (diff.data[i] / target.data[i]).abs();
            }
        }
        total_loss / T::from_usize(length)
    }

    /// # Formula
    /// ```math
    ///   \frac{∂MAPE}{∂ŷ_i} = \frac{1}{N} * \frac{1}{ŷ_i} * sign(ŷ_i - y_i)
    /// ```
    fn gradient(&self, output: &Matrix<T>, target: &Matrix<T>) -> Matrix<T> {
        if output.size() != target.size() {
            panic!("!!!Size of output matrix and target must be equal!!!")
        }

        let length = output.data.len();
        let mut grad = vec![T::default(); length];

        for i in 0..length{
            if target.data[i] != T::default() {
                let sign = if output.data[i] > target.data[i] {
                    T::one() // 1
                } else if output.data[i] < target.data[i] {
                    Float::neg(T::one()) // -1
                } else {
                    T::default() // 0
                };
                grad[i] = sign / target.data[i];
            }
        }
        Matrix::new(grad, target.rows, output.cols) * (T::from_usize(1)/ T::from_usize(length))
    }

}