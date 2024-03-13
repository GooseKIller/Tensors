use crate::activation::Function;
use crate::Float;
use crate::linalg::Matrix;

/// f(x) = \frac{e^x}{e^x+1}
pub struct Sigmoid;

impl Sigmoid{
    fn new() -> Self{
        Self{}
    }

    fn num_fun<T:Float>(&self, num:T) -> T{
        let one:T = 1.into();
        one/(one+num.neg().exp())
    }


    fn num_der<T:Float>(&self, num:T) -> T{
        let one:T = 1.into();
        num.exp()/((num.exp()+one)*(num.exp()+one))
    }
}

impl<T:Float> Function<T> for Sigmoid{
    fn call(&self, matrix: Matrix<T>) -> Matrix<T> {
        let [row, cols] = [matrix.rows, matrix.cols];
        let mut data = Vec::with_capacity(row*cols);
        for i in matrix.data{
            data.push(self.num_fun(i));
        }
        Matrix{ data, rows:row, cols }
    }
    fn derivative(&self, matrix: Matrix<T>) -> Matrix<T> {
        let [row, cols] = [matrix.rows, matrix.cols];
        let mut data = Vec::with_capacity(row*cols);
        for i in matrix.data{
            data.push(self.num_der(i));
        }
        Matrix{ data, rows:row, cols }
    }
}

#[cfg(test)]
mod tests{
    use crate::activation::{Function, Sigmoid};
    use crate::matrix;
    use crate::linalg::Matrix;

    #[test]
    fn relu(){
        let matrix = matrix![[0.0, -10.0]];
        let a = Sigmoid::new();
        let matrix = a.call(matrix);
        println!("{}", matrix);
    }

    #[test]
    fn derivative_relu(){
        let matrix = matrix![[0.0, -10.0]];
        let a = Sigmoid::new();
        let matrix = a.derivative(matrix);
        println!("{}", matrix);
    }
}
