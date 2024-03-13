use crate::activation::Function;
use crate::Float;
use crate::linalg::Matrix;

pub struct LeakyReLU<T:Float>{
    alpha: T,
}

impl<T:Float> LeakyReLU<T> {
    fn new(_:T) -> Self{
        let two:T = 2.into();
        let ten:T = 10.into();
        Self{
            alpha: two/ten
        }
    }

    fn leaky_num(&self, num: T) -> T{
        let zero = T::default();
        if num > zero{
            num
        } else {
            num * self.alpha
        }
    }

    fn leaky_der(&self, num:T) -> T{
        let zero = T::default();
        if num > zero{
            1.into()
        } else {
            self.alpha
        }
    }
}

impl<T:Float> From<T> for LeakyReLU<T>  {
    fn from(value: T) -> Self {
        Self{
            alpha:value,
        }
    }
}

impl<T:Float> Function<T> for LeakyReLU<T>{
    fn call(&self, matrix: Matrix<T>) -> Matrix<T> {
        let [row, cols] = [matrix.rows, matrix.cols];
        let mut data = Vec::with_capacity(row*cols);
        for i in matrix.data{
            let num = self.leaky_num(i);
            data.push(num);
        }
        Matrix::new(data, row, cols)
    }
    fn derivative(&self, matrix: Matrix<T>) -> Matrix<T> {
        let [row, cols] = [matrix.rows, matrix.cols];
        let mut data = Vec::with_capacity(row*cols);
        for i in matrix.data{
            let num = self.leaky_der(i);
            data.push(num);
        }
        Matrix::new(data, row, cols)
    }
}

#[cfg(test)]
mod tests{
    use crate::activation::{Function};
    use crate::activation::leaky_relu::LeakyReLU;
    use crate::{matrix, DataType};
    use crate::linalg::Matrix;

    #[test]
    fn leaky_relu(){
        let matrix = matrix![[10.0, -10.0]];
        let a = LeakyReLU::new(DataType::f64());
        let matrix = a.call(matrix);
        println!("{}", matrix);
    }

    #[test]
    fn derivative_leaky_relu(){
        let matrix = matrix![[10.0, -10.0]];
        let a = LeakyReLU::from(0.2);
        let matrix = a.derivative(matrix);
        println!("{}", matrix);
    }
}