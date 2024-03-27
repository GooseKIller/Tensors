use crate::activation::Function;
use crate::Float;
use crate::linalg::Matrix;

///Mean squared error
///
/// Number is here to help rust understand which type of data it will work
pub struct MSE<T: Float>{
    datatype_number: T
}

impl<T: Float> MSE<T>{
    pub fn new(a: T) -> Self{
        Self{
            datatype_number:a,
        }
    }

    pub fn call(self, input: Matrix<T>, target: Matrix<T>) -> f64{
        if input.size() != target.size(){
            panic!("!!!Size of input matrix and target must be equal!!!")
        }
        let length = input.data.len();
        let difference = input - &target;
        let mut total_loss = T::default();
        for i in 0..difference.data.len(){
            total_loss += difference.data[i] * difference.data[i];
        }
        total_loss.to_f64() / (length as f64)
    }
}

#[cfg(test)]
mod tests{
    use crate::loss::mse::MSE;
    use crate::{DataType, matrix};
    use crate::linalg::Matrix;

    #[test]
    fn mse_loss_test(){
        let input  = matrix![[1.0, 2.0, 3.0, 4.0]];
        let output = matrix![[1.0, 3.0, 3.5, 4.5]];

        let mse = MSE::new(DataType::f64());
        assert_eq!(mse.call(input, output), 0.375);
    }
}
