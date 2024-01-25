use crate::activation::Function;
use crate::linalg::Matrix;
use crate::Float;

/// ReLU(rectified linear unit) - activation function
///
/// f(x) = max(x, 0)
///
/// or
///
/// f(x) = {x if x > 0;
///
///0 otherwise}
pub struct ReLU;


impl ReLU{
    pub fn new() -> Self{
        Self{}
    }
}

impl<T:Float> Function<T> for ReLU{
    fn call(self, matrix: Matrix<T>) -> Matrix<T> {
        let [row, cols] = [matrix.rows, matrix.cols];
        let mut data = Vec::with_capacity(row*cols);
        for i in matrix.data{
            let num = if i > T::default(){
                i
            } else {
                T::default()
            };
            data.push(num);
        }
        Matrix{ data, rows:row, cols }
    }

    ///Derivative of Relu
    ///
    ///if x > 0 then 1
    ///
    ///else 0
    fn derivative(self, matrix: Matrix<T>) -> Matrix<T> {
        let [row, cols] = [matrix.rows, matrix.cols];
        let mut data = Vec::with_capacity(row*cols);
        for i in matrix.data{
            let num = if i > T::default() {1.into()} else {0.into()};
            data.push(num);
        }
        Matrix{ data, rows:row, cols }
    }
}

#[cfg(test)]
mod tests{
    use crate::activation::{Function, ReLU};
    use crate::matrix;
    use crate::linalg::Matrix;

    #[test]
    fn relu(){
        let matrix = matrix![[10.0, -10.0]];
        let a = ReLU::new();
        let matrix = a.call(matrix);
        println!("{}", matrix);
    }

    #[test]
    fn derivative_relu(){
        let matrix = matrix![[10.0, -10.0]];
        let a = ReLU::new();
        let matrix = a.derivative(matrix);
        println!("{}", matrix);
    }
}