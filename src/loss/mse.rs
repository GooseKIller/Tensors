use crate::Float;
use crate::linalg::Matrix;

///Mean squared error
///
/// Number is here to help rust understand which type of data it will work
pub struct MSE<T: Float>{ _datatype_number: T }

impl<T: Float> MSE<T>{
    pub fn new(a: T) -> Self{
        Self{
            _datatype_number:a,
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
    use std::ops::{Sub, SubAssign};
    use crate::loss::mse::MSE;
    use crate::{DataType, matrix, Num, vector};
    use crate::activation::Function;
    use crate::linalg::{Matrix, Vector};
    use crate::nn::Linear;

    #[test]
    fn mse_loss_test(){
        let input  = matrix![[1.0, 2.0, 3.0, 4.0]];
        let output = matrix![[1.0, 3.0, 3.5, 4.5]];

        let mse = MSE::new(DataType::f64());
        assert_eq!(mse.call(input, output), 0.375);
    }

    #[test]
    fn grad_test(){
        let answer = vector![3f64, 2f64];
        let data:Vec<f64> = (0..10).into_iter().map(|x1| {x1 as f64}).collect();
        let target:Vec<f64> = data.clone().into_iter().map(|x1| { vector![x1, 1f64].scalar(&answer)}).collect();
        let data:Vec<Vec<f64>> = (0..10).into_iter().map(|x1| {vec![x1 as f64, 1f64]}).collect();

        let x = Matrix::from(data);
        let y = Vector::from(target);

        let mut w = vector![0.5f64, 0.5f64];
        let n = y.length as f64;
        let mut attempts = 0usize;
        println!("{}\n{}", x, y);
        while (answer.clone() - &w).abs_sum() > 0.1f64 && attempts < 10000{
            let y_pred = x.clone() * &w;
            let error = y_pred - &y;

            w = w - x.clone().transpose() * &error * (2f64/n) * 0.01;
            attempts += 1;
        }
        println!("weights:{w}\n for {attempts} attempts");//mean attempts 446
    }

    #[test]
    fn stochastic_descent_test(){
        let n = 1000;
        let m = 3;

        //training data
        let mut x = vec![vec![0f64; m]; n];
        let mut y = vec![0f64; n];
        for i in 0..n{
            let i_num = rand::random::<f64>() as f64;
            let j_num = rand::random::<f64>();
            //println!("{}", i_num);
            x[i] = vec![i_num, 1f64-i_num, j_num];
            y[i] = 2_f64*i_num + 5_f64*(1f64-i_num)  + 8_f64*j_num;

        }

        // batch size, learning rate, epochs
        let b = 100;
        let alpha = 0.1;
        let e = 500;

        let mut w = Vector::from(vec![0.0f64; m]);
        let answer = vector![3f64, 4f64, -1f64];
        for _ in 0..e {
            for i in (b..n).step_by(b){
                let x_batch = Matrix::from(x[i- b..i].to_vec());//Matrix::from(x.to_vec());//
                let y_batch = Vector::from(y[i- b..i].to_vec());//y.to_vec());
                let f = &x_batch * &w;
                let err = f - y_batch;
                let grad = x_batch.transpose() * &err * 2f64 * (1f64 / (b as f64));
                w = w - grad * alpha;
                if (answer.clone() - &w).abs_sum() < 0.01{
                    break;
                }
            }

        }
        println!("Веса {}", w);
    }
}
