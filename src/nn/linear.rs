use rand::distributions::{Distribution, Standard};
use crate::{Float};
use crate::linalg::{Matrix, Vector};
use rand::random;
use crate::activation::Function;

pub struct Linear<T: Float>{
    pub(crate) matrix:Matrix<T>,
    bias:bool,
}

impl<T:Float> Linear<T>
    where Standard: Distribution<T>{

    pub fn new(row:usize, col:usize, bias:bool) -> Self{
        if bias{
            let mut data = Vec::with_capacity(col*(row+1));
            for _ in 0..((row+1)*col){
                data.push(random::<T>());
            }
            let mut matrix = Matrix::new(data, row+1, col);
            return Self{
                matrix,
                bias
            }
        }
        let mut data = Vec::with_capacity(row*col);
        for _ in 0..(col*row){
            data.push(random::<T>());
        }
        let mut matrix = Matrix::new(data, row, col);
        Self{
            matrix,
            bias
        }
    }

    /// From Matrix with added bias
    ///
    ///# Example
    ///
    /// ```
    ///use tensors::linalg::Matrix;
    ///use tensors::matrix;
    ///use tensors::nn::Linear;
    ///
    ///let linear:Linear<f64> = Linear::new(1, 2, true);//with one bias row it will be 2x2
    ///assert_eq!([2, 2], linear.shape());
    ///```
    pub fn shape(&self) -> [usize; 2] {
        [self.matrix.rows, self.matrix.cols]
    }

    /// Return weights matrix
    ///
    /// # Example
    ///
    /// ```
    /// use tensors::matrix;
    /// use tensors::linalg::Matrix;
    /// use tensors::nn::Linear;
    /// let act1:Linear<f64> = Linear::from(matrix![[1.0],
    ///                                 [1.0]]);
    /// let sum_num = 2.0;
    ///assert_eq!(sum_num, act1.get_data().sum());
    /// ```
    pub fn get_data(&self) -> Matrix<T>{
        self.matrix.clone()
    }
}



impl<T:Float> From<Matrix<T>> for Linear<T>{

    /// From Matrix with added bias
    ///
    ///# Example
    ///
    /// ```
    ///use tensors::linalg::Matrix;
    ///use tensors::matrix;
    ///use tensors::nn::Linear;
    ///
    ///let mx:Matrix<f64> = matrix![[1.0],
    ///                 [1.0]]; // this is bias
    ///let linear = Linear::from(mx);
    ///println!("{:?}", linear.shape());
    ///```
    fn from(value: Matrix<T>) -> Self {
        Self{
            matrix:value,
            bias:true,
        }
    }
}

impl<T:Float> Function<T> for Linear<T> {
    fn call(&self, mut matrix: Matrix<T>) -> Matrix<T> {
        if self.bias{
            let rows = matrix.row();
            let num_bias:Vector<T> = Vector::from_num(1.into(), rows);
            matrix.add_column(num_bias.into())
        }
        matrix * &self.matrix
    }

    fn derivative(&self, matrix: Matrix<T>) -> Matrix<T> {
        matrix
    }
}

#[cfg(test)]
mod tests{
    use crate::activation::Function;
    use crate::linalg::Matrix;
    use crate::matrix;
    use crate::nn::linear::Linear;

    #[test]
    fn new_linear(){
        let a: Linear<f64> = Linear::new(1, 1, true);
        println!("{}", a.matrix);
    }

    #[test]
    fn call_linear(){
        let a: Linear<f64> = Linear::new(1,1,true);
        let m = Matrix::from_num(1.0, 1, 1);
        let call = a.call(m);
        assert_eq!(Matrix::from_num(a.matrix.sum(),1,1), call);
    }

    #[test]
    fn from_matrix(){
        let matrix = matrix![[1.0],[2.0]];
        let linear = Linear::from(matrix);
        let m = Matrix::from_num(0.0, 1, 1);
        let call = linear.call(m);
        assert_eq!(Matrix::from_num(2.0,1,1), call)
    }

    #[test]
    fn linear_shape(){
        let linear = Linear::new(2, 1, true);
        let matrix = matrix![[1.0, 2.0]];

        let ans = linear.call(matrix);
        assert_eq!(1, ans.rows);
        assert_eq!(1, ans.cols);
    }
}