use rand::distributions::{Distribution, Standard};
use crate::{Float};
use crate::linalg::Matrix;
use rand::random;
use crate::activation::Function;

pub struct Linear<T: Float>{
    pub matrix:Matrix<T>,
    bias: Option<Matrix<T>>
}

impl<T:Float> Linear<T>
    where Standard: Distribution<T>{

    pub fn new(row:usize, col:usize, bias:bool) -> Self{
        let mut data = Vec::with_capacity(col*row);
        for _ in 0..((row)*col){ data.push(random::<T>()); }

        if bias{
            let matrix = Matrix::new(data, row, col);
            let bias = Matrix::from_num(T::default(), 1, col);
            return Self{
                matrix,
                bias: Some(bias)
            }
        }

        let matrix = Matrix::new(data, row, col);
        Self{
            matrix,
            bias: None
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
    ///assert_eq!(sum_num, act1.get_weights().sum());
    /// ```
    pub fn get_weights(&self) -> Matrix<T>{
        self.matrix.clone()
    }
    pub fn get_bias(&self) -> Option<Matrix<T>> {self.bias.clone()}
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
            bias: None
        }
    }
}

impl<T:Float> Function<T> for Linear<T> {
    fn call(&self, mut matrix: Matrix<T>) -> Matrix<T> {
        if let Some(bias) = &self.bias {
            return matrix * &self.matrix + bias
        }
        matrix * &self.matrix
    }

    /// not real derivative just delta calculating
    fn derivative(&self, mut matrix: Matrix<T>) -> Matrix<T> {// not real derivative just calculating delta
        /*
        if self.bias {
            let no_bias_matrix = self.matrix.clone().get_resize(
                self.matrix.rows-1,
                self.matrix.cols);
            return matrix * &no_bias_matrix
        }*/
        matrix * &self.matrix.clone().transpose()
    }

    fn is_linear(&self) -> bool{
        true
    }

    fn get_data(&self) -> Option<Matrix<T>> {
        Some(self.matrix.clone())
    }
    fn get_bias(&self) -> Option<Matrix<T>> {
        self.bias.clone()
    }
    fn set_data(&mut self, data: Matrix<T>) {
        self.matrix = data;
    }
    fn set_bias(&mut self, new_bias: Matrix<T>) {
        if let Some(mut bias) = &mut self.bias{
            *bias = new_bias;
        }
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
        let matrix = matrix![[1.0],
                                        [2.0]];
        let linear = Linear::from(matrix);
        let m =  matrix![[1.0, 1.0]];
        let call = linear.call(m);
        assert_eq!(Matrix::from_num(3.0,1,1), call)
    }
}