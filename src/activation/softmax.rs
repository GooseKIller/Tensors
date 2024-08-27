use crate::activation::Function;
use crate::Float;
use crate::linalg::{Matrix, Vector};

/// Softmax function or normalized exponential function
///
/// converts a vector of K real numbers into a probability distribution of K possible outcomes.
/// # Defined as:
///
///```math
///  Softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} {e}^(x_j)}
///```
pub struct SoftMax;

impl SoftMax{
    pub fn new() -> Self{
        Self{}
    }

    fn vec_fun<T:Float>(&self, vector: Vector<T>) -> Vector<T>{
        let mut sum:T = T::default();
        let vector:Vec<T> = vector.into();
        for i in &vector{
            sum += i.exp();
        }
        let mut answer = Vector::from_num(T::default(),vector.len());
        for i in 0..vector.len(){
            answer[i] = vector[i].exp()/sum;
        }
        answer
    }
}

impl<T:Float> Function<T> for SoftMax {
    fn call(&self, matrix: Matrix<T>) -> Matrix<T> {
        let mut data:Vec<Vector<T>> = Vec::with_capacity(matrix.rows);
        for i in 0..matrix.rows{
            let vector = matrix.get_row(i);
            let vector = self.vec_fun(vector);
            data.push(vector)
        }
        Matrix::from(data)

    }

    /// $`Softmax'(x_i)= Softmax(x_i) * (δ_{ij} - Softmax(x_j))`$
    ///
    /// $`δ_{ij}`$ - the Kronecker symbol, which is 1 when i = j, and 0 otherwise
    fn derivative(&self, matrix: Matrix<T>) -> Matrix<T> {
        let [row, cols] = [matrix.rows, matrix.cols];
        let softmax = self.call(matrix);
        let mut det = Matrix::from_num(T::default(), row, cols);
        for i in 0..row{
            for j in 0..cols{
                let kronecker:T = if i == j {1.into()} else { 0.into() };
                det[[i, j]] = softmax[[i ,j]] * (kronecker - softmax[[i ,j]]);
            }
        }
        det

    }
}

#[cfg(test)]
mod tests{
    use crate::activation::{Function, SoftMax};
    use crate::matrix;
    use crate::linalg::Matrix;

    #[test]
    fn relu(){
        let matrix = matrix![[0.0, 1.0],
                                        [1.0, 0.0]];
        let a = SoftMax::new();
        let matrix = a.call(matrix);
        println!("{}", matrix);
    }

    #[test]
    fn derivative_relu(){
        let matrix = matrix![[0.0, -10.0]];
        let a = SoftMax::new();
        let matrix = a.derivative(matrix);
        println!("{}", matrix);
    }
}