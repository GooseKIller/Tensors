use crate::linalg::Matrix;
use crate::loss::Loss;
use crate::Float;

/// Sum of Squared Errors (SSE) loss function.
///
/// The `SSE` loss function computes the sum of the squared differences between
/// the predicted values and the target values. It is commonly used in regression tasks.
///
/// # Mathematical Definition
/// For predicted values `y_pred` and target values `y_true`, the SSE is defined as:
///
/// SSE = \sum_{i=1}^n (y_{true, i} - y_{pred, i})^2
///
/// # Type Constraints
/// - `T: Float`: The loss function works only with floating-point types (e.g., `f32`, `f64`).
/// # Notes
/// - The `datatype_number` parameter in `new` is a placeholder and is not used in the computation.
///   It is included to ensure type consistency with other loss functions.
/// - SSE is sensitive to outliers due to the squaring of errors.
///
/// # See Also
/// - [Wikipedia: Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
pub struct SSE<T: Float>(T);

impl<T: Float> SSE<T> {
    /// Creates a new `SSE` loss function.
    ///
    /// # Arguments
    /// * `datatype_number` - A placeholder value of type `T`. This is not used in the computation
    ///   but ensures type consistency with other loss functions.
    ///
    /// # Returns
    /// A new instance of the `SSE` loss function.
    pub fn new(datatype_number: T) -> Self {
        Self(datatype_number)
    }
}

impl<T: Float> Loss<T> for SSE<T> {
    fn call(&self, output: &Matrix<T>, target: &Matrix<T>) -> T {
        if output.shape() != target.shape() {
            panic!(
                "!!!Size of output matrix and target must be equal!!!\
            \nOutput size:{:?} Target size: {:?}",
                output.shape(),
                target.shape()
            )
        }
        let diff = target - output;
        diff.map(|x| x.powf(T::from(2))).sum()
    }

    fn gradient(&self, output: &Matrix<T>, target: &Matrix<T>) -> Matrix<T> {
        if output.shape() != target.shape() {
            panic!("!!!Size of output matrix and target must be equal!!!")
        }
        target - output
    }
}
