use crate::activation::Function;
use crate::Float;
use crate::linalg::Matrix;

pub struct ELU<T:Float>{
    alpha: T,
}
impl<T:Float> ELU<T> {
    fn new(_:T) -> Self{
        Self{
            alpha: 1.into()
        }
    }

    fn elu_num(&self, num: T) -> T{
        let zero = T::default();
        let one:T = 1.into();
        if num > zero{
            num
        } else {
            self.alpha*(num.exp() - one)
        }
    }

    fn elu_der(&self, num:T) -> T{
        let zero = T::default();
        if num > zero{
            1.into()
        } else {
            self.alpha*num.exp()
        }
    }
}

impl<T:Float> From<T> for ELU<T>  {
    fn from(value: T) -> Self {
        Self{
            alpha:value,
        }
    }
}

impl<T:Float> Function<T> for ELU<T> {
    fn call(&self, matrix: Matrix<T>) -> Matrix<T> {
        let [row, cols] = [matrix.rows, matrix.cols];
        let mut data = Vec::with_capacity(row * cols);
        for i in matrix.data {
            let num = self.elu_num(i);
            data.push(num);
        }
        Matrix::new(data, row, cols)
    }
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
mod tests{
    use crate::activation::elu::ELU;
    use crate::activation::Function;
    use crate::{matrix, DataType};
    use crate::linalg::Matrix;

    #[test]
    fn test_elu(){
        let matrix = matrix![[1.0, 0.0]];
        let a = ELU::new(DataType::f64());
        let matrix = a.call(matrix);
        println!("{}", matrix);
    }

    #[test]
    fn derivative_test(){
        let matrix = matrix![[1.0, 0.0]];
        let a = ELU::from(1.0);
        let matrix = a.derivative(matrix);
        println!("{}", matrix);
    }
}