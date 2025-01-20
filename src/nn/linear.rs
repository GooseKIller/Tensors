use rand::distributions::{Distribution, Standard};
use crate::{Float};
use crate::linalg::{Matrix, Vector};
use rand::random;
use crate::activation::Function;

pub struct Linear<T: Float>{
    pub matrix:Matrix<T>,
    pub(crate) bias: bool
}

impl<T:Float> Linear<T>
    where Standard: Distribution<T>{

    /// Creates a new matrix with added bias(optional)
    ///
    /// bias realized as double row
    pub fn new(row:usize, col:usize, bias:bool) -> Self{

        // Xavier method
        let mut data = Vec::with_capacity(row * col);
        let limit = (T::from_usize(6) / T::from_usize(row + col)).sqrt();//sqrt(6) / sqrt(n_i + n_i+1)

        for _ in 0..(col * row) {
            let value = random::<T>() * T::from(2) * limit - limit; // [-limit, limit)
            data.push(value);
        }

        if bias {
            let mut bias_data = Vec::with_capacity(col);
            for _ in 0..col {
                bias_data.push(T::default()); // Инициализация смещений нулями
            }
            data.extend(bias_data);
        }

        let matrix = Matrix::new(data, row + if bias { 1 } else { 0 }, col);
        Self {
            matrix,
            bias,
        }

        /*
        if bias{
            let mut data = Vec::with_capacity(col*(row+1));
            for _ in 0..((row+1)*col){
                data.push(random::<T>());
            }
            let matrix = Matrix::new(data, row+1, col);
            return Self{
                matrix,
                bias
            }
        }

        let mut data = Vec::with_capacity(row*col);
        for _ in 0..(col*row){
            data.push(random::<T>());
        }
        let matrix = Matrix::new(data, row, col);
        Self{
            matrix,
            bias
        }*/
    }

    /// Creates a matrix without random numbers
    ///
    /// the same as ::new method
    pub fn zeros(row:usize, col:usize, bias:bool) -> Self{
        if bias{
            let data = vec![T::default(); col*(row+1)];
            let matrix = Matrix::new(data, row+1, col);
            return Self{
                matrix,
                bias
            }
        }

        let data = vec![T::default(); col*row];
        let matrix = Matrix::new(data, row, col);
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
    ///assert_eq!(sum_num, act1.get_weights().sum());
    /// ```
    pub fn get_weights(&self) -> Matrix<T>{
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
            bias: true
        }
    }
}

impl<T:Float> Function<T> for Linear<T> {
    fn name(&self) -> String {
        let shape = self.matrix.shape();
        format!("Linear_{}:{} {}", self.bias as u8, shape[0], shape[1])
    }
    fn call(&self, mut matrix: Matrix<T>) -> Matrix<T> {
        if self.bias{
            let rows = matrix.row();
            let num_bias:Vector<T> = Vector::from_num(1.into(), rows);
            matrix.add_column(num_bias.into())
        }
        matrix * &self.matrix
    }

    /// not real derivative just delta calculating
    fn derivative(&self, matrix: Matrix<T>) -> Matrix<T> {
        if self.bias {
            let ans = &matrix * &self.matrix.transpose();
            return ans.rem_col(ans.cols-1);
        }
        &matrix * &self.matrix.transpose()
    }

    fn is_linear(&self) -> bool{
        true
    }

    fn get_data(&self) -> Option<Matrix<T>> {
        Some(self.matrix.clone())
    }
    
    fn set_data(&mut self, _data: Matrix<T>) {
        self.matrix = _data;
    }

    fn get_weights(&self) -> Option<Matrix<T>> {
        let weights = &self.matrix.data[0..(self.matrix.rows-1)*self.matrix.cols];
        Some(
            Matrix::new(
                weights.to_owned(),
                self.matrix.rows-1,
                self.matrix.cols
            )
        )
    }

    fn get_bias(&self) -> Option<Matrix<T>> {
        if !self.bias{
            return None;
        }
        Some(
            Matrix::from(
                self.matrix.get_row(
                    self.matrix.row()-1
                )
            )
        )
    }

    fn is_bias(&self) -> bool {
        self.bias
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
        let m =  matrix![[1.0]];
        let call = linear.call(m);
        assert_eq!(Matrix::from_num(3.0,1,1), call)
    }
}