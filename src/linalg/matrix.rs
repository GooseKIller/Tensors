use std::fmt::{Display, Formatter};
use std::ops::{Add, Index, IndexMut, Mul, Sub};
use crate::linalg::{Num, Vector};

// this is unreadable
/// Matrix definition
///
/// # Example
/// ```
/// use tensors::linalg::Matrix;
/// use tensors::matrix;
///
/// let matrix_a = matrix![[1,2,3],
///                     [4,5,6],
///                     [7,8,9]];
/// //this will create matrix
/// //[[1 2 3]
/// //[4 5 6]
/// //[7 8 9]]
/// let matrix_b = Matrix::new(vec![1,2,3,4,5,6,7,8,9], 3, 3);// same as matrix_a
/// ```
#[macro_export]
macro_rules! matrix {
    ($([$($x:expr),* $(,)*]),* $(,)*) => {
        Matrix::from(vec![
            $(vec![
                $($x,)*
            ],)*
        ])
    };
}

///reference by (skyl4b)<https://github.com/TheAlgorithms/Rust/blob/master/src/math/matrix_ops.rs>
#[derive(PartialEq, Eq, Debug)]
pub struct Matrix<T:Num>{
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T:Num> Matrix<T> {

    /// Create matrix from vector and usize, usize
    /// # Example
    /// ```
    /// use tensors::linalg::Matrix;
    /// let example = Matrix::new(vec![0, -1, -1, 0], 2, 2);
    /// // Will create matrix
    /// // [0 -1]
    /// // [-1 0]
    /// ```
    pub fn new(data:Vec<T>, rows:usize, cols:usize) -> Self{
        if data.len() != rows * cols{
            panic!("!!!Inconsistent data and dimensions combination for matrix!!!")
        }
        Self{ data, rows, cols }
    }

    /// Create matrix from number where all value will be equal num
    ///
    /// # Example
    ///
    /// ```
    /// use tensors::linalg::Matrix;
    /// let matrix_a = Matrix::from_num(1, 2, 2);
    /// //[1 1]
    /// //[1 1]
    /// ```
    pub fn from_num(num:T, rows:usize, cols:usize) -> Self{
        Self{
            data: vec![num; rows * cols],
            rows,
            cols,
        }
    }

    /// Creates a single matrix
    ///
    /// need to implement type
    ///
    /// # Example
    /// ```
    /// use tensors::linalg::Matrix;
    /// let a:Matrix<f64> = Matrix::single(2, 2);
    /// // will create matrix
    /// // [1 0]
    /// // [0 1]
    /// ```
    pub fn single(rows:usize, cols:usize) -> Self{
        let mut matrix = Vec::with_capacity(rows*cols);
        for i in 0..rows{
            for j in 0..cols{
                if i == j {
                    matrix.push(1.into());
                } else {
                    matrix.push(T::default());
                }
            }
        }
        Self{
            data:matrix,
            rows,
            cols,
        }
    }

    /// Returns row count
    pub fn row(&self) -> usize{
        self.rows.clone()
    }

    /// Returns column count
    pub fn col(&self) -> usize{
        self.cols.clone()
    }

    /// Returns column as Vector with index (index starts with 0)
    ///
    /// # Example
    /// ```
    /// use tensors::linalg::{Matrix, Vector};
    /// use tensors::matrix;
    /// let example = matrix![[1,2],
    ///                     [3,4]];
    /// let col:Vector<i32> = example.get_col(0);// [1 3]
    /// ```
    pub fn get_col(&self, index: usize) -> Vector<T> {
        if index >= self.cols {
            panic!("!!!Index {} more then columns count {}!!!", index, self.cols);
        }
        let mut vector = Vec::with_capacity(self.rows);
        for i in 0..self.rows{
            let index_col = i*self.cols + index;
            vector.push(self.data[index_col]);
        }
        Vector::from(vector)
    }

    /// Returns row as Vector with index (index starts with 0)
    ///
    /// # Example
    /// ```
    /// use tensors::linalg::{Matrix, Vector};
    /// use tensors::matrix;
    /// let example = matrix![[1,2],
    ///                     [3,4]];
    /// let col:Vector<i32> = example.get_row(0);//[1 2]
    /// ```
    pub fn get_row(&self, index:usize) -> Vector<T>{
        if index >= self.cols {
            panic!("!!!Index {} more then rows count {}!!!", index, self.rows);
        }
        let start_index = index * self.cols;
        let end_index = start_index + self.cols;

        Vector::from(self.data[start_index..end_index].to_vec())
    }


    /// Transpose matrix
    ///
    /// # Example
    ///
    /// ```
    /// use tensors::matrix;
    /// use tensors::linalg::Matrix;
    /// let example = matrix![[1,2],
    ///                     [3,4]];
    /// let example = example.transpose();
    /// //[1 3]
    /// //[2 4]
    /// ```
    pub fn transpose(self) -> Self {
        let mut result = Self::from_num(T::default(), self.cols, self.rows);
        for i in 0..self.cols{
            for j in 0..self.rows{
                result[[i, j]] = self[[j, i]];
            }
        }
        result
    }
}

impl<T:Num> Index<[usize;2]> for Matrix<T> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let [i, j] = index;
        if i >= self.rows || j >= self.cols{
            panic!("!!!Matrix index out of bounds!!!");
        }

        &self.data[(self.cols * i) + j]
    }
    
}

impl<T:Num> IndexMut<[usize; 2]> for Matrix<T>{
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        let [i, j] = index;
        if i >= self.rows || j >= self.cols{
            panic!("!!!Matrix index out of bounds!!!");
        }

        &mut self.data[(self.cols * i) + j]
    }
}

impl<T:Num> Add<&Matrix<T>> for Matrix<T>{
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols{
            panic!("!!!Matrix dimensions do not match!!!");
        }
        let mut result = Self::from_num(T::default(), self.rows, self.cols);
        for i in 0..self.rows{
            for j in 0..self.cols{
                result[[i, j]] = self[[i, j]] + rhs[[i, j]];
            }
        }
        result
    }
}

impl<T:Num> Add<T> for Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, rhs:T) -> Self::Output {
        let mut result = Self::from_num(T::default(), self.rows, self.cols);
        for i in 0..self.rows{
            for j in 0..self.cols{
                result[[i, j]] = self[[i, j]] + rhs;
            }
        }
        result
    }
}

impl<T:Num> Sub<&Matrix<T>> for &Matrix<T>{
    type Output = Matrix<T>;
    fn sub(self, rhs: &Matrix<T>) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols{
            panic!("!!!Matrix dimensions do not match!!!");
        }
        let mut result = Matrix::from_num(T::default(), self.rows, self.cols);
        for i in 0..self.rows{
            for j in 0..self.cols{
                result[[i, j]] = self[[i, j]] - rhs[[i, j]];
            }
        }
        result
    }
}

impl<T:Num> Sub<T> for Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, rhs:T) -> Self::Output {
        let mut result = Self::from_num(T::default(), self.rows, self.cols);
        for i in 0..self.rows{
            for j in 0..self.cols{
                result[[i, j]] = self[[i, j]] - rhs;
            }
        }
        result
    }
}

impl<T:Num> Mul<&Matrix<T>> for Matrix<T>{
    type Output = Matrix<T>;
    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        if self.cols != rhs.rows{
            panic!("!!!Matrix amount of columns 1st matrix does not equals to amount of rows of the 2nd one!!!")
        }
        let mut result = Self::from_num(T::default(), self.rows, rhs.cols);
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut sum = T::default();
                for k in 0..self.cols{
                    sum += self[[i, k]] * rhs[[k, j]];
                }

                result[[i,j]] = sum;
            }
        }
        result
    }
}

impl<T:Num> Mul<T> for Matrix<T>{
    type Output = Matrix<T>;
    fn mul(self, rhs: T) -> Self::Output {
        let mut result = Self::from_num(T::default(), self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[[i,j]] = self[[i,j]] * rhs;
            }
        }
        result
    }
}

impl<T:Num> From<Vec<Vec<T>>> for Matrix<T> {
    fn from(value: Vec<Vec<T>>) -> Self {
        let rows = value.len();
        let cols = value.first().map_or(0, |row | row.len());

        for row in value.iter().skip(1){
            if row.len() != cols{
                panic!("!!!All columns must be equal!!!")
            }
        }

        if rows != 0 && cols == 0 {
            panic!("!!!Invalid matrix dimensions. Multiple empty rows!!!");
        }

        let data = value.into_iter().flatten().collect();
        Self::new(data, rows, cols)
    }
}

impl<T:Num> From<Vec<T>> for Matrix<T>{
    fn from(value: Vec<T>) -> Self {
        Self{
            data:value.clone(),
            rows:1,
            cols:value.len(),
        }
    }
}

impl<T:Num> From<Vector<T>> for Matrix<T> {
    fn from(value: Vector<T>) -> Self {
        let vector:Vec<T> = value.into();
        Self{
            data:vector.clone(),
            rows:1,
            cols:vector.len(),
        }
    }
}

impl<T:Num> From<Vec<Vector<T>>> for Matrix<T>{
    fn from(value: Vec<Vector<T>>) -> Self {
        let mut vector:Vec<Vec<T>> = vec![];
        for i in value{
            vector.push(i.into());
        }
        Self::from(vector)
    }
}

impl<T:Num> Display for Matrix<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut vectors = vec![];
        for i in 0..self.rows{
            vectors.push(self.get_row(i));
        }
        let vectors = vectors
            .iter()
            .map(|x| format!("{x}"))
            .collect::<Vec<_>>()
            .join(",\n");
        write!(f, "[{}]", vectors)
    }
}

#[cfg(test)]
mod tests{
    use crate::linalg::matrix::*;

    #[test]
    fn macro_test(){
        let a = matrix!([1,2,3],[1,2,3],[1,2,3]);
        println!("{}", a);
    }

    #[test]
    fn create_matrix(){
        let a = Matrix::from_num(10, 2,2);

        println!("{a}");
    }

    #[test]
    fn single_matrix(){
        let a:Matrix<f64> = Matrix::single(2, 2);
        let b = matrix![[1.0,0.0],
            [0.0,1.0]];
        println!("{}", a);
        assert_eq!(a, b);
    }

    #[test]
    fn from_vector(){
        let a = Vector::from_num(5,5);
        let b = Matrix::from(a);
        let a = matrix!([5,5,5,5,5]);
        assert_eq!(a, b);
    }

    #[test]
    fn get_cols(){
        let a = matrix![
            [1,2,3,4],
            [1,2,3,4],
            [1,2,3,4]];
        let b = Vector::from_num(4, 3);
        assert_eq!(b, a.get_col(3));
        let c = Vector::from_num(10, 2);
    }

    #[test]
    fn get_rows(){
        let a = matrix![
            [1,2,3,4],
            [1,2,3,4],
            [1,2,3,4]];
        let b = Vector::from(vec![1,2,3,4]);
        assert_eq!(a.get_row(2), b)
    }

    #[test]
    fn transpose_matrix(){
        let a = vec![1,2,3, 1,2,3, 1,2,3];
        let a = Matrix::new(a, 3, 3);
        let a = a.transpose();
        println!("{}", a);
    }

    #[test]
    fn add_matrix(){
        let a = Matrix::from_num(1,1,1);
        let a1 = Matrix::from_num(2,1,1);

        let ans = Matrix::from_num(3,1,1);
        assert_eq!(ans, a+&a1);

    }

    #[test]
    fn add_matrix_and_num(){
        let ans = Matrix::from_num(3,1,1);
        let b = Matrix::from_num(2,1,1);

        assert_eq!(ans, b+1);
    }

    #[test]
    fn mul_matrix(){
        let a = Matrix::new(vec![0,1,-1,0],2,2);
        let a1 = Matrix::new(vec![0,1,-1,0],2,2);

        let ans = Matrix::new(vec![-1,0,0,-1],2,2);
        assert_eq!(ans, a*&a1);

    }

    #[test]
    fn test_mul_f64() {
        let a = matrix![
            [5.0, 2.0, 1.0, 9.0],
            [0.0, 3.0, 11.0, 17.0],
            [5.0, 8.0, 2.0, 3.0],
        ];

        let b = matrix![
            [1.0, 3.0, 5.0],
            [-2.0, 1.0, 3.0],
            [-3.0, 1.0, 3.0],
            [0.0, 2.0, 2.0],
        ];

        let mul = matrix![
            [-2.0, 36.0, 52.0],
            [-39.0, 48.0, 76.0],
            [-17.0, 31.0, 61.0]
        ];

        assert_eq!(a * &b, mul);
    }
}