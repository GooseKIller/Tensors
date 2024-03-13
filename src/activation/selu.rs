use crate::activation::Function;
use crate::Float;
use crate::linalg::Matrix;


/// Scaled Exponential Linear Units, are activation functions that induce self-normalization. SELU network neuronal activations automatically converge to a zero mean and unit variance.
/// Mathematically, it is expressed as:
///
/// f(x)=λx        if  x>0
///
/// f(x)=λα(ex−1)  if  x≤0
///
/// α = 1.67326
///
/// λ = 1.0507
pub struct SELU<T:Float>{
    alpha: T,
    lambda: T,
}

impl<T:Float> SELU<T> {

    ///Number here for generics
    ///
    /// This number would not be used anywhere
    fn new(_:T) -> Self{
        let alpha:T = T::selu_alpha(T::default());
        let lambda:T = T::selu_lambda(T::default());
      Self{
          alpha,
          lambda,
      }
    }

    fn selu_num(&self, x:T) -> T{
        let one:T = 1.into();
        if x > T::default(){
            self.lambda * x
        } else {
            self.lambda * self.alpha * (x.exp() - one)
        }
    }

    /// Maybe wrong
    fn selu_der(&self, x:T) -> T{
        if x > T::default() {
            self.lambda
        } else {
            self.lambda * self.alpha * x.exp()
        }
    }
}

impl<T:Float> From<(T, T)> for SELU<T>{
    fn from(params: (T, T)) -> Self {
        let (alpha, scale) = params;
        Self{alpha, lambda: scale }
    }
}

impl<T:Float> Function<T> for SELU<T>{
    fn call(&self, matrix: Matrix<T>) -> Matrix<T> {
        let [row, cols] = [matrix.rows, matrix.cols];
        let mut data = Vec::with_capacity(row * cols);
        for i in matrix.data {
            let num = self.selu_num(i);
            data.push(num);
        }
        Matrix::new(data, row, cols)
    }
    fn derivative(&self, matrix: Matrix<T>) -> Matrix<T> {
        let [row, cols] = [matrix.rows, matrix.cols];
        let mut data = Vec::with_capacity(row * cols);
        for i in matrix.data {
            let num = self.selu_der(i);
            data.push(num);
        }
        Matrix::new(data, row, cols)
    }
}

#[cfg(test)]
mod tests{
    use crate::activation::{Function, SELU};
    use crate::{matrix, DataType};
    use crate::linalg::Matrix;

    #[test]
    fn selu(){
        let matrix = matrix![[0.0, 1.0]];
        let a = SELU::new(DataType::f64());
        let matrix = a.call(matrix);
        println!("{}", matrix);
    }

    #[test]
    fn derivative_selu(){
        let matrix = matrix![[0.0, 1.0]];
        let a = SELU::new(DataType::f64());
        let matrix = a.derivative(matrix);
        println!("{}", matrix);
    }
}