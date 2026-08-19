use rand::random;
use std::cmp::min;
use rayon::prelude::*;
use crate::{Float, Num};
use std::ops::{Index, IndexMut};
use std::fmt::{Display, Formatter};
use crate::linalg::{Tensor, Vector};
use rayon::prelude::IntoParallelRefMutIterator;
use rand::distributions::{Distribution, Standard};

/// Matrix definition
///
/// # Example
/// ```
/// use tensorrs::linalg::Matrix;
/// use tensorrs::matrix;
///
/// let matrix_a = matrix![[1,2,3],
///                     [4,5,6],
///                     [7,8,9]];
/// //this will create matrix
/// //[[1 2 3]
/// //[4 5 6]
/// //[7 8 9]]
/// let matrix_b = Matrix::new(vec![1,2,3,4,5,6,7,8,9], 3, 3);// same as matrix a
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

/// A `Matrix` represents a two-dimensional mathematical structure consisting of rows and columns,
/// used for various mathematical operations, including linear algebra.
///
/// Reference: [skyl4b](https://github.com/TheAlgorithms/Rust/blob/master/src/math/matrix_ops.rs)
///
/// The `Matrix` struct is implemented using a vector, which means it is not a simple struct.
/// As a result, all mathematical operations are implemented without borrowing.
///
/// When performing operations, ensure to use a reference for the second operand(and for the first):
///
/// # Example
/// ```rust
/// use tensorrs::linalg::Matrix;
///
/// let a = Matrix::from_num(0, 2, 2);
/// let b = Matrix::from_num(1, 2, 2);
///
/// &a + &b; // Correct
/// // a + b;  // Incorrect
/// ```
///
/// All Matrix operations
///
/// | Name | Operation | Example |
/// |------|-----------|---------|
/// | Plus | + | `&matrix1 + &matrix2` |
/// | Plus assign| += | `&mut matrix1 += &matrix2`|
/// | Minus | - | `&matrix1 - &matrix2` |
/// | Minus assign| -=| `&mut matrix1 -= &matrix2` |
/// | Mul | * | `&matrix1 * &matrix2` |
/// | Mul assign| *= | `&mut matrix1 *= &matrix2` |
/// | Hadamard(element-wise)| & | `&matrix & &matrix2` |
/// | Hadamard(element-wise) assign| &=| `&mut matrix &= &matrix2` |
#[derive(PartialEq, Eq, Debug)]
pub struct Matrix<T: Num> {
    pub(crate) data: Vec<T>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl<T: Num> Matrix<T> {
    /// Create matrix from vector and usize, usize
    /// 
    /// can panic if length of vector does not equal rows * cols
    /// # Example
    /// ```
    /// use tensorrs::linalg::Matrix;
    /// let example = Matrix::new(vec![0, -1, -1, 0], 2, 2);
    /// // Will create matrix
    /// // [0 -1]
    /// // [-1 0]
    /// ```
    pub fn new(data: Vec<T>, rows: usize, cols: usize) -> Self {
        if data.len() != rows * cols {
            panic!("!!!Inconsistent data and dimensions combination for matrix!!!")
        }
        Self { data, rows, cols }
    }

    /// Create Matrix of zeros
    ///
    /// shape:
    /// [rows, cols]
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Matrix;
    /// let a:Matrix<f64> = Matrix::zeros([3, 2]);
    /// // matrix![{0, 0},
    /// //         {0, 0},
    /// //         {0, 0}]
    /// ```
    pub fn zeros(shape: [usize; 2]) -> Self {
        Self {
            data: vec![T::default(); shape[0] * shape[1]],
            rows: shape[0],
            cols: shape[1],
        }
    }

    /// Create matrix from number where all value will be equal num
    ///
    /// # Example
    ///
    /// ```
    /// use tensorrs::linalg::Matrix;
    /// let matrix_a = Matrix::from_num(1, 2, 2);
    /// //[1 1]
    /// //[1 1]
    /// ```
    pub fn from_num(num: T, rows: usize, cols: usize) -> Self {
        Self {
            data: vec![num; rows * cols],
            rows,
            cols,
        }
    }

    /// Creates Matrix from function that takes two arguments
    /// 
    /// # Example
    /// ```
    /// use tensorrs::linalg::Matrix;
    /// use tensorrs::matrix;
    /// let matrix = Matrix::<i32>::from_fn(3, 3, |i, j| (i * 10 + j) as i32);
    ///
    /// assert_eq!(matrix, matrix![[0, 1, 2], [10, 11, 12], [20, 21, 22]]);
    /// ```
    pub fn from_fn<F>(rows: usize, cols: usize, f: F) -> Self
    where
        F: Fn(usize, usize) -> T + Sync + Send {
        let mut data = vec![T::default(); rows*cols];

        data.par_iter_mut().enumerate().for_each(|(idx, item)| {
            let i = idx / cols;
            let j = idx % cols;
            *item = f(i, j);
        });

        Self {
            data,
            rows,
            cols,
        }
    }

    /// Safe creating of matrix from tensor
    /// 
    /// # Exmple
    /// ```
    /// use tensorrs::linalg::{Matrix, Tensor};
    /// use tensorrs::tensor;
    /// use tensorrs::matrix;
    /// 
    /// let tensor = tensor![[1,2,3], [1,2,3]];
    /// let matrix = Matrix::try_from(tensor).unwrap();
    /// assert_eq!(matrix![[1,2,3], [1,2,3]], matrix);
    /// 
    /// ```
    pub fn try_from(value: Tensor<T>) -> Result<Self, &'static str> {
        if value.shape.len() != 2 {
            return Err("Shape size must be 2");
        }
        Ok(Matrix::new(value.packed_data(), value.shape[0], value.shape[1]))
    }

    /// Creating Matrix with diagonal values of vector
    /// 
    /// Can panic if length of vector not equal to min(rows, cols)
    /// # Example
    /// ```
    /// use tensorrs::linalg::{Matrix, Vector};
    /// use tensorrs::matrix;
    /// 
    /// let a = Vector::from(vec![1,2,3]);
    /// 
    /// let b = Matrix::from_diag(a, 3, 3);
    /// 
    /// assert_eq!(b, matrix![[1, 0, 0],
    ///                       [0, 2, 0],
    ///                       [0, 0, 3]])
    /// ```
    pub fn from_diag(data:Vector<T>, rows:usize, cols:usize) -> Matrix<T> {
        assert_eq!(data.length, min(rows, cols),
                   "!!!The length of the data vector ({})\
                    must be equal to the minimum of rows ({}) and cols ({})!!!",
                   data.length,
                   rows,
                   cols
        );
        let mut mx_data = vec![T::default(); cols * rows];
        for i in 0..min(cols, rows) {
            let index = (cols * i) + i;
            mx_data[index] = data[i];
        }
        Matrix{
            data:mx_data,
            rows,
            cols
        }
    }

    /// Creates a identity matrix
    ///
    /// need to explicit write a data type
    ///
    /// # Example
    /// ```
    /// use tensorrs::DataType;
    /// use tensorrs::linalg::Matrix;
    /// let a:Matrix<f64> = Matrix::identity(DataType::f64(), 2, 2);
    /// // will create matrix
    /// // [1 0]
    /// // [0 1]
    /// ```
    pub fn identity(_: T, rows: usize, cols: usize) -> Self {
        let mut matrix = vec![T::default(); rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                if i == j {
                    matrix[i * rows + j] = T::from(1);
                }
            }
        }
        Self {
            data: matrix,
            rows,
            cols,
        }
    }

    ///Return data from matrix
    ///
    /// # Example
    ///
    ///```
    /// use tensorrs::linalg::Matrix;
    /// let a = Matrix::from_num(10, 2, 1);
    /// let a = a.get_data();
    /// // should return vec![10, 10]
    /// ```
    pub fn get_data(&self) -> Vec<T> {
        self.data.clone()
    }

    /// Returns shape of matrix
    ///
    /// # Example
    ///
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    ///
    /// let a = matrix![[1], [2]];
    /// println!("SHAPE:{:?}", a.shape());//[2, 1]
    /// ```
    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    /// Returns amount of rows
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns amount of columns
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns sum of all elements of matrix
    pub fn sum(&self) -> T {
        let mut sum = T::default();
        for i in self.data.iter() {
            sum += *i;
        }
        sum
    }

    /// Returns column as Vector with index (index starts with 0)
    /// 
    /// Can panic if index get out of bounds
    /// # Example
    /// ```
    /// use tensorrs::linalg::{Matrix, Vector};
    /// use tensorrs::matrix;
    /// let example = matrix![[1,2],
    ///                     [3,4]];
    /// let col:Vector<i32> = example.get_col(0);// [1 3]
    /// ```
    pub fn get_col(&self, index: usize) -> Vector<T> {
        assert!(
            index < self.cols,
            "!!!Index:{} is greater than or equal to columns count:{}!!!",
            index,
            self.cols
        );
        let mut vector = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let index_col = i * self.cols + index;
            vector.push(self.data[index_col]);
        }
        Vector::from(vector)
    }

    /// Removes column from matrix
    /// 
    /// Can panic if index get out of bounds
    /// # Example
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// let example = matrix![[1., 2.], [3., 4.]];
    /// let rem_example = example.rem_col(1);
    /// assert_eq!(rem_example,
    ///          matrix![[1.], [3.]]);
    /// ```
    pub fn rem_col(&self, index: usize) -> Matrix<T> {
        assert!(index < self.cols, "!!!Column index out of bounds!!!");

        let mut new_data = self.data.clone();
        for i in (0..self.rows).rev() {
            let index = i * self.cols + index;
            new_data.remove(index);
        }

        Matrix {
            data: new_data,
            rows: self.rows,
            cols: self.cols - 1,
        }
    }

    /// Returns row as Vector with index (index starts with 0)
    ///
    /// Can panic if index get out of bounds
    /// # Example
    /// ```
    /// use tensorrs::linalg::{Matrix, Vector};
    /// use tensorrs::matrix;
    /// let example = matrix![[1,2],
    ///                     [3,4]];
    /// let col:Vector<i32> = example.get_row(1);//[1 2]
    /// ```
    pub fn get_row(&self, index: usize) -> Vector<T> {
        assert!(
            index < self.rows,
            "!!!Index:{} is greater than or equal to rows count:{}!!!",
            index,
            self.rows
        );
        let start_index = index * self.cols;
        let end_index = start_index + self.cols;

        Vector::from(self.data[start_index..end_index].to_vec())
    }

    /// Transpose matrix
    ///
    /// # Example
    ///
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// let matrix = matrix![[1,2],
    ///                     [3,4]];
    /// let example = matrix.transpose();
    /// //[1 3]
    /// //[2 4]
    /// ```
    pub fn transpose(&self) -> Self {
        let mut result = Self::from_num(T::default(), self.cols, self.rows);
        for i in 0..self.cols {
            for j in 0..self.rows {
                result[[i, j]] = self[[j, i]];
            }
        }
        result
    }

    /// Adds column at the end of Matrix
    ///
    /// Can panic if index get out of bounds
    /// # Example
    ///
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// let mut a = matrix![[1, 1],
    ///                      [1, 1]];
    ///
    /// a.add_column(vec![2,2]);
    /// // [[1,1,2]
    /// // [1,1,2]]
    /// ```
    pub fn add_column(&mut self, column: Vec<T>) {
        assert_eq!(
            column.len(),
            self.rows,
            "!!!The length of the Vec<T> is not equal to the size of the rows of the matrix!!!"
        );
        for i in 0..self.rows {
            self.data.insert((i + 1) * self.cols + i, column[i].clone());
        }
        self.cols += 1;
    }

    pub fn set_col(&mut self, col: usize, v: &Vector<T>) {
        assert!(col < self.cols, "Column index out of bounds");
        assert_eq!(
            v.len(),
            self.rows,
            "Vector length must match number of rows"
        );

        for i in 0..self.rows {
            self[[i, col]] = v[i];
        }
    }

    /// Adds row at the end of Matrix
    ///
    /// Can panic if index get out of bounds
    /// # Example
    ///
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// let mut a = matrix![[1, 1],
    ///                      [1, 1]];
    ///
    /// a.add_row(vec![2,2]);
    /// // [[1,1]
    /// // [1,1]
    /// // [2,2]]
    /// ```
    pub fn add_row(&mut self, row: Vec<T>) {
        assert_eq!(
            row.len(),
            self.cols,
            "!!!the length of the Vec<T> is not equal to the size of the columns of the matrix!!!"
        );
        for i in row {
            self.data.push(i)
        }
        self.rows += 1;
    }

    /// Gets new matrix with same data but other shape
    ///
    /// # Example
    ///
    /// ```
    ///
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// let a = matrix![[1.0, 2.0]];
    /// a.get_resize(1, 1);// will get matrix![[1.0]];
    /// a.get_resize(2, 2);// will get matrix![[1.0, 2.0], [0.0, 0.0]]
    /// ```
    pub fn get_resize(&self, new_row: usize, new_col: usize) -> Matrix<T> {
        let mut new_matrix = Matrix::from_num(T::default(), new_row, new_col);
        for i in 0..min(new_row, self.rows) {
            for j in 0..min(new_col, self.cols) {
                new_matrix[[i, j]] = self[[i, j]];
            }
        }

        new_matrix
    }

    /// Returns a new matrix where each cell is the sum of rows.
    /// 
    /// # Example
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// 
    /// let a = matrix![[1,2,3], [3,2,6]];
    /// assert_eq!(a.sum_rows(), matrix![[4,4,9]]);
    /// ```
    pub fn sum_rows(&self) -> Matrix<T> {
        let mut matrix = vec![T::default(); self.cols];
        for i in 0..self.cols {
            matrix[i] = self.get_col(i).sum_all();
        }
        Matrix::new(matrix, 1, self.cols)
    }

    /// Comparison of two matrices
    /// 
    /// panics if their shapes are different
    /// # Example
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// let a = matrix![[1, 3, -1]];
    /// let b = matrix![[0, 4, -1]];
    ///
    /// println!("{}", a.compare(b));//[{1 -1 0}]
    /// ```
    pub fn compare(&self, other: Matrix<T>) -> Matrix<T> {
        assert_eq!(
            self.shape(),
            other.shape(),
            "!!!Can't compare matrices different shapes!!!\n Matrix a:{:?}; Matrix b:{:?}",
            self.shape(),
            other.shape()
        );
        let mut comparisons = vec![T::default(); self.rows * self.cols];
        comparisons.par_iter_mut().enumerate().for_each(|(i, x)| {
            if self.data[i] > other.data[i] {
                *x = T::from(1);
            } else if self.data[i] < other.data[i] {
                *x = T::from(1).neg();
            }
        });
        Matrix::new(comparisons, self.rows, self.cols)
    }

    /// Comparison of matrix and number
    ///
    /// # Example
    ///```
    /// use tensorrs::linalg::Matrix;
    /// use tensorrs::matrix;
    /// let a = matrix![[5,0,-3]];
    /// println!("{}", a.compare_num(0));//[{1, 0 ,-1}]
    ///```
    pub fn compare_num(&self, other: T) -> Matrix<T> {
        Matrix::new(
            self.data
                .iter()
                .map(|x| {
                    if *x > other {
                        T::from(1)
                    } else if *x == other {
                        T::from(0)
                    } else {
                        T::from(1).neg()
                    }
                })
                .collect(),
            self.rows,
            self.cols,
        )
    }

    /// Hadamard product or element-wise product
    ///
    /// panics if their shapes are different
    /// # Example
    ///```
    /// use tensorrs::linalg::Matrix;
    /// use tensorrs::matrix;
    /// let a = matrix![[2, 3, 1], [0, 8, -2]];
    /// let b = matrix![[3, 1, 4], [7, 9, 5]];
    /// assert_eq!(&a & &b,
    ///    matrix![[6, 3, 4]
    ///            ,[0, 72, -10]]);
    /// assert_eq!(a.hadamard(&b),
    ///    matrix![[6, 3, 4]
    ///            ,[0, 72, -10]])
    /// ```
    ///
    /// Or
    pub fn hadamard(&self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(
            other.shape(),
            self.shape(),
            "!!!Shapes must be equal. Matrix A: {:?} Matrix B: {:?}!!!",
            self.shape(),
            other.shape()
        );
        let mut ans = vec![T::default(); self.data.len()];
        ans.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.data[i] * other.data[i];
        });
        Matrix::new(ans, self.rows, self.cols)
    }

    /// [Kronecker](https://en.wikipedia.org/wiki/Kronecker_product) product of two matrices
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Matrix;
    /// use tensorrs::matrix;
    ///
    /// let a = matrix![[1, 2], [3, 4]];
    /// let b = matrix![[0, 5], [6, 7]];
    ///
    /// assert_eq!(a.kronecker(&b),
    ///     matrix![[0, 5, 0, 10],
    ///             [6, 7,12, 14],
    ///             [0, 15, 0, 20],
    ///             [18, 21, 24, 28]])
    /// ```
    pub fn kronecker(&self, other: &Matrix<T>) -> Matrix<T> {
        let mut ans = vec![T::default(); self.data.len() * other.data.len()];
        ans.par_iter_mut().enumerate().for_each(|(index, x)| {
            let i = index / (self.cols * other.cols);
            let j = index % (self.cols * other.cols);
            let a_row = i / self.rows;
            let a_col = j / self.cols;
            let b_row = i % other.rows;
            let b_col = j % other.cols;

            *x = self.data[a_row * self.cols + a_col] * other.data[b_row * other.cols + b_col];
        });
        Matrix::new(ans, self.rows * other.rows, self.cols * other.cols)
    }

    /// Parallel mapping for each element
    ///
    ///
    /// # Example
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// let a = matrix![[1,2], [3,4]];
    /// let b = a.map(|x| x % 2);
    /// // [{1 0},
    /// // {1 0}]
    /// ```
    pub fn map<F>(&self, f: F) -> Matrix<T>
    where
        F: Fn(T) -> T + Sync + Send,
    {
        let new_data = self.data.clone().par_iter_mut().map(|x| f(*x)).collect();
        Matrix {
            data: new_data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Applies a binary function to corresponding elements of two matrices, producing a new matrix.
    ///
    /// /// # Example
    /// ```rust
    /// use tensorrs::linalg::Matrix;
    ///
    /// let a = Matrix::from_num(1, 2, 2);
    /// let b = Matrix::from_num(2, 2, 2);
    /// let result = a.zip_with(&b, |x, y| x + y);
    /// ```
    pub fn zip_with<F>(&self, other: &Matrix<T>, f: F) -> Matrix<T>
    where
        F: Fn(T, T) -> T + Sync + Send,
    {
        let new_data = self
            .data
            .clone()
            .par_iter_mut()
            .enumerate()
            .map(|(i, x)| f(*x, other.data[i]))
            .collect();
        Matrix {
            data: new_data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Returns a new matrix where each element is the maximum
    /// between number and the original matrix
    /// and the given number.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Matrix;
    /// use tensorrs::matrix;
    /// let a = matrix![[1.0, 2.0], [0.0, -1.0]];
    /// let b = a.max(0.);
    /// assert_eq!(
    ///     b,
    ///     matrix![[1.0, 2.0], [0.0, 0.0]]
    /// );
    /// ```
    pub fn max(&self, num: T) -> Matrix<T> {
        self.map(|x| if x > num { x } else { num })
    }

    /// Returns a new matrix where each element is the minimum
    /// between the corresponding element of the original matrix
    /// and the given number.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Matrix;
    /// use tensorrs::matrix;
    /// let a = matrix![[1.0, 2.0], [0.0, -1.0]];
    /// let b = a.min(0.);
    /// assert_eq!(
    ///     b,
    ///     matrix![[0.0, 0.0], [0.0, -1.0]]
    /// );
    /// ```
    pub fn min(&self, num: T) -> Matrix<T> {
        self.map(|x| if x > num { num } else { x })
    }

    /// Extracts the main diagonal from a matrix and returns it as a vector.
    ///
    /// The diagonal elements are taken from positions where row index equals column index (i, i).
    /// For non-square matrices, the diagonal length equals the smaller dimension of the matrix.
    /// 
    /// # Example
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::{Vector, Matrix};
    /// 
    /// let a = matrix![
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]];
    /// 
    /// assert_eq!(a.diag(), Vector::from(vec![1,5,9]));
    /// ```

    pub fn diag(&self) -> Vector<T> {
        let mut data = vec![T::default(); min(self.cols, self.rows)];
        for i in 0..min(self.cols, self.rows) {
            let index = (self.cols * i) + i;
            data[i] = self.data[index];
        }
        Vector::from(data)
    }

    /// Performs a valid convolution (without padding) of the matrix with the given kernel.
    /// 
    /// Panics if the kernel is larger than the matrix in either dimension.
    /// # Example
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    ///
    /// let input = matrix![
    ///     [1, 2, 3, 4],
    ///     [5, 6, 7, 8],
    ///     [9, 10, 11, 12],
    ///     [13, 14, 15, 16]
    /// ];
    ///
    /// let kernel = matrix![
    ///     [1, 0, -1],
    ///     [1, 0, -1],
    ///     [1, 0, -1]
    /// ];
    ///
    /// let result = input.conv(&kernel);
    /// // Result: [[-6, -6],
    /// // [-6, -6]]
    /// ```
    pub fn conv(&self, kernel: &Matrix<T>) -> Matrix<T> {
        assert!(
            kernel.rows <= self.rows && kernel.cols <= self.cols,
            "!!!Kernel size must be less than Matrix itself!!!\nMatrix size: {:?}, Kernel Size: {:?}",
            self.shape(),
            kernel.shape()
        );
        
        let output_rows = self.rows - kernel.rows + 1;
        let output_cols = self.cols - kernel.cols + 1;

        let mut result_data = vec![T::default(); output_cols * output_rows];

        result_data.par_chunks_mut(output_cols)
        .enumerate()
        .for_each(|(i, out_row)| {
            for j in 0..output_cols {
                let mut sum = T::default();

                for ki in 0..kernel.rows {
                    let matrix_row_base = (i + ki) * self.cols + j;
                    let kernel_row_base = ki * kernel.cols;

                    for kj in 0..kernel.cols {
                        sum += self.data[matrix_row_base + kj] * 
                        kernel.data[kernel_row_base + kj];
                    }
                }
                out_row[j] = sum;
            }
        });
        Matrix::new(result_data, output_rows, output_cols)
    }

    /// Performs a convolution with zero padding, preserving the input dimensions.
    /// 
    /// # Example
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    ///
    /// let input = matrix![
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ];
    ///
    /// let kernel = matrix![
    ///     [0, 1, 0],
    ///     [1, 2, 1],
    ///     [0, 1, 0]
    /// ];
    ///
    /// let result = input.conv_zero(&kernel);
    /// // Result: [[ 8, 14,  3],
    /// // [21, 30, 29],
    /// // [26, 37, 32]]
    /// ```
    pub fn conv_zero(&self, kernel: &Matrix<T>) -> Matrix<T> {
        let pad_rows = kernel.rows / 2;
        let pad_cols = kernel.cols / 2;
        
        let output_rows = self.rows;
        let output_cols = self.cols;

        let mut result_data = vec![T::default(); output_cols * output_rows];

        result_data.par_chunks_mut(output_cols)
        .enumerate()
        .for_each(|(i, out_row)| {
            for j in 0..output_cols {
                let mut sum = T::default();

                for ki in 0..kernel.rows {
                    for kj in 0..kernel.cols {
                        let mi = i as i32 + ki as i32 - pad_rows as i32;
                        let mj = j as i32 + kj as i32 - pad_cols as i32;
                        
                        if mi >= 0 && mi < self.rows as i32 && mj >= 0 && mj < self.cols as i32 {
                            let matrix_idx = mi as usize * self.cols + mj as usize;
                            let kernel_idx = ki * kernel.cols + kj;
                            sum = sum + self.data[matrix_idx] * kernel.data[kernel_idx];
                        }
                    }
                }
                out_row[j] = sum;
            }
        });
    
    Matrix::new(result_data, output_rows, output_cols)


    }

    /// Performs a convolution with mirror padding, preserving the input dimensions.
    /// 
    /// # Example
    /// /// # Example
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    ///
    /// let input = matrix![
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    /// ];
    ///
    /// let gaussian = matrix![
    ///     [1, 2, 1],
    ///     [2, 4, 2],
    ///     [1, 2, 1]
    /// ];
    ///
    /// // Output will be 3x3 with mirror padding at edges
    /// let blurred = input.conv_with_mirror_padding(&gaussian);
    /// // Result: [[48,  56,  64],
    /// // [72,  80,  88],
    /// // [96, 104, 112]]
    /// ```
    pub fn conv_with_mirror_padding(&self, kernel: &Matrix<T>) -> Matrix<T> {
        fn mirror_index(idx: i32, size: usize) -> usize {
            let size_i32 = size as i32;
            if idx < 0 {
                (-idx - 1) as usize % size
            } else if idx >= size_i32 {
                (2 * size_i32 - idx - 1) as usize % size
            } else {
                idx as usize
            }
        }

        let pad_rows = kernel.rows / 2;
        let pad_cols = kernel.cols / 2;
        
        let output_rows = self.rows;
        let output_cols = self.cols;

        let mut result_data = vec![T::default(); output_cols * output_rows];

        result_data.par_chunks_mut(output_cols)
            .enumerate()
            .for_each(|(i, out_row)| {
                for j in 0..output_cols {
                    let mut sum = T::default();

                    for ki in 0..kernel.rows {
                        for kj in 0..kernel.cols {
                            // Work out the coordinates with a mirror reflection
                            let mi = mirror_index(i as i32 + ki as i32 - pad_rows as i32, self.rows);
                            let mj = mirror_index(j as i32 + kj as i32 - pad_cols as i32, self.cols);
                            
                            let matrix_idx = mi * self.cols + mj;
                            let kernel_idx = ki * kernel.cols + kj;
                            
                            sum = sum + self.data[matrix_idx] * kernel.data[kernel_idx];
                        }
                    }
                    out_row[j] = sum;
                }
            });
        
        Matrix::new(result_data, output_rows, output_cols)
    }


    /// Subtracts a scalar value (lambda) from all diagonal elements of the matrix.
    /// 
    /// /// This operation modifies the matrix in-place by subtracting the given lambda
    /// value from each element on the main diagonal
    /// 
    /// # Examples
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// 
    /// let mut a = matrix![
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0],
    ///     [7.0, 8.0, 9.0]
    /// ];
    /// 
    /// a.set_lambda(2.0);
    /// 
    /// // After subtraction, the diagonal becomes: 
    /// // 1.0 - 2.0 = -1.0, 5.0 - 2.0 = 3.0, 9.0 - 2.0 = 7.0
    /// assert_eq!(a.diag().get_data(), vec![-1.0, 3.0, 7.0]);
    /// ```
    pub fn set_lambda(&mut self, lambda: T) {
        for i in 0..self.rows {
            self.data[i * self.cols + i] = self.data[i * self.cols + i] - lambda;
        }
    }

    pub fn hstack(&self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows, "!!!Rows must be same mx1: {}, mx2:{}!!!",
            self.rows, other.rows);
        let mut new_data = vec![T::default(); self.rows * (self.cols + other.cols)];

        new_data.par_chunks_mut(self.cols + other.cols)
        .enumerate()
        .for_each(|(row_idx, row_slice)| {
            for col in 0..self.cols {
                row_slice[col] = self[[row_idx, col]];
            }

            for col in 0..self.cols {
                row_slice[self.cols + col] = self[[row_idx, col]];
            }
        });

        Matrix::new(new_data, self.rows, self.cols + other.cols)
    }


    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }

    pub fn indexed_iter(&self) -> impl Iterator<Item = ((usize, usize), &T)> {
        (0..self.rows).flat_map( move |r| {
            (0..self.cols).map(move |c| ((r, c), &self.data[(self.cols * r) + c]))
        })
    }  

    pub fn rows_iter(&self) -> impl Iterator<Item = &[T]> {
        self.data.chunks(self.cols)
    }
    
    // A row iterator that hands out mutable access
    pub fn rows_iter_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        self.data.chunks_mut(self.cols)
    }

    pub fn filter<P>(&self, predicate: P) -> impl Iterator<Item = &T> 
    where
        P: FnMut(&&T) -> bool,
    {
        self.iter().filter(predicate)
    }
}

impl<T: Num> Index<[usize; 2]> for Matrix<T> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let [i, j] = index;
        if i >= self.rows || j >= self.cols {
            panic!(
                "!!!Matrix index out of bounds!!! Got [{i}, {j}] but excepted less than [{}, {}]",
                self.rows, self.cols
            );
        }

        &self.data[(self.cols * i) + j]
    }
}

impl<T: Num> IndexMut<[usize; 2]> for Matrix<T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        let [i, j] = index;
        assert!(
            i < self.rows && j < self.cols,
            "!!!Matrix index out of bounds: i = {}, j = {}, rows = {}, cols = {}!!!",
            i,
            j,
            self.rows,
            self.cols
        );

        &mut self.data[(self.cols * i) + j]
    }
}

impl<'a, T: Num> IntoIterator for &'a Matrix<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T: Num> IntoIterator for &'a mut Matrix<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<T: Num> From<Vec<Vec<T>>> for Matrix<T> {
    fn from(value: Vec<Vec<T>>) -> Self {
        let rows = value.len();
        let cols = value.first().map_or(0, |row| row.len());

        for row in value.iter().skip(1) {
            assert_eq!(row.len(), cols, "!!!All columns must be equal!!!");
        }

        assert!(
            !(rows != 0 && cols == 0),
            "!!!Invalid matrix dimensions. Multiple empty rows!!!"
        );

        let data = value.into_iter().flatten().collect();
        Self::new(data, rows, cols)
    }
}

impl<T: Num> From<Vec<T>> for Matrix<T> {
    fn from(value: Vec<T>) -> Self {
        Self {
            data: value.clone(),
            rows: 1,
            cols: value.len(),
        }
    }
}

impl<T: Num> From<Vector<T>> for Matrix<T> {
    fn from(value: Vector<T>) -> Self {
        let vector: Vec<T> = value.into();
        Self {
            data: vector.clone(),
            rows: 1,
            cols: vector.len(),
        }
    }
}

impl<T: Num> From<Tensor<T>> for Matrix<T> {
    fn from(value: Tensor<T>) -> Self {
        assert_eq!(value.shape.len(), 2, "!!!Shape size must be 2!!!");
        Self {
            data: value.packed_data(),
            rows: value.shape[0],
            cols: value.shape[1],
        }
    }
}

impl<T: Num> From<Vec<Vector<T>>> for Matrix<T> {
    fn from(value: Vec<Vector<T>>) -> Self {
        let mut vector: Vec<Vec<T>> = vec![];
        for i in value {
            vector.push(i.into());
        }
        Self::from(vector)
    }
}

impl<T: Num> Display for Matrix<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut vectors = vec![];
        for i in 0..self.rows {
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

impl<T: Num> Clone for Matrix<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

// Float Number implementation
impl<T: Float> Matrix<T> {
    /// Creates a matrix with random numbers(between 0 and 1)
    /// This is achieved using the [Box-Muller transform](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform), which generates normally distributed random numbers
    /// from uniformly distributed random numbers.
    pub fn randn(row: usize, col: usize) -> Self
    where
        Standard: Distribution<T>,
    {
        Self {
            data: vec![T::default(); row * col]
                .iter()
                .map(|_| {
                    (-T::from(2) * random::<T>().ln()).sqrt() // Bpx - Muller Method
                        * (T::from(2) * T::pi() * random::<T>()).cos()
                })
                .collect(),
            rows: row,
            cols: col,
        }
    }

    pub fn rand(rows: usize, cols: usize) -> Self
    where 
        Standard: Distribution<T>,
    {
        Self { data: vec![T::default(); rows * cols]
                .iter()
                .map(|_| {
                    random::<T>()
                })
                .collect(),
             rows, cols }
    }

    /// Natural logarithm for all elements
    pub fn ln(&self) -> Self {
        self.map(|x| x.ln())
    }

    /// Finds the norm of Matrix in any power
    pub fn norm(&self, p: T) -> T {
        assert!(p >= T::one(), "!!!Number p:{} must be positive!!!", p);
        let mut norm = T::default();

        if p == T::one() {
            let mut max_num = self.get_col(0).abs_sum();

            for i in 1..self.cols {
                let sum = self.get_col(i).abs_sum();
                max_num = if sum > max_num { sum } else { max_num };
            }
            max_num
        } else if p == T::from(2) {
            let mut sum_of_squares = T::default();
            for x in &self.data {
                sum_of_squares += x.powf(T::from(2));
            }
            return sum_of_squares.sqrt();
        } else {
            for i in &self.data {
                norm += i.powf(p);
            }
            norm.powf(T::one() / p)
        }
    }

    /// Finds the norm of matrix in infinite power
    pub fn norm_inf(self) -> T {
        let mut max_num = self.get_row(0).abs_sum();

        for i in 1..self.rows {
            let sum = self.get_row(i).abs_sum();
            max_num = if sum > max_num { sum } else { max_num };
        }
        max_num
    }

    /// Finds determinant of matrix
    ///
    /// Using Gauss Method (<https://en.wikipedia.org/wiki/Gaussian_elimination>)
    pub fn det(&self) -> T {
        assert_eq!(
            self.rows, self.cols,
            "!!!The determinant is defined only for square matrices!!!"
        );

        let mut matrix = self.clone();
        let mut det = 1.into();
        for i in 0..self.rows {
            for j in (i + 1)..self.rows {
                let coefficient = matrix[[j, i]] / matrix[[i, i]];
                for k in i..self.rows {
                    matrix[[j, k]] = matrix[[j, k]] - coefficient * matrix[[i, k]];
                }
            }
            det = det * matrix[[i, i]];
        }
        det
    }

    //Need to optimize
    /// Finds an inverse matrix
    ///
    /// Will throw Err if matrix is not square or singular
    pub fn inv(&self) -> Result<Matrix<T>, &'static str> {
        if self.rows != self.cols {
            return Err("Matrix is not invertible. Matrix must be square.");
        }
        let n = self.rows;
        let mut augmented_matrix = self.clone();
        let mut inv_matrix = Matrix::identity(T::default(), self.rows, self.rows);

        // Forward elimination
        for k in 0..n {
            let diagonal = augmented_matrix.data[k * n + k];
            if diagonal == T::default() {
                return Err("Matrix is singular.");
            }

            for j in 0..n {
                augmented_matrix.data[k * n + j] = augmented_matrix.data[k * n + j] / diagonal;
                inv_matrix.data[k * n + j] = inv_matrix.data[k * n + j] / diagonal;
            }

            for i in 0..n {
                if i == k {
                    continue;
                }

                let factor = augmented_matrix.data[i * n + k];
                for j in 0..n {
                    let help_aug = augmented_matrix.data[k * n + j];
                    augmented_matrix.data[i * n + j] -= factor * help_aug;
                    let help_inv = inv_matrix.data[k * n + j];
                    inv_matrix.data[i * n + j] -= factor * help_inv;
                }
            }
        }

        Ok(inv_matrix)
    }

    pub fn qr(&self) -> (Matrix<T>, Matrix<T>) {
        let n = self.rows;
        let mut q = Matrix::from_num(T::default(), n, n);
        let mut r = q.clone();

        let mut v: Vec<Vector<T>> = (0..n)
            .map(|j| self.get_col(j))
            .collect();

        for i in 0..n {
            for j in 0..i {
                let rij = q.get_col(j).scalar(&v[i]);
                r[[j, i]] = rij;
                v[i] -= &(q.get_col(j) * rij);
            }

            let norm = v[i].length();
            r[[i, i]] = norm;

            q.set_col(i, &(&v[i] / norm));
        }
        (q, r)
    }

    /// Computes the eigenvalues of a **square** matrix.
    /// # Examples
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// 
    /// let a = matrix![
    ///     [2.0, 1.0],
    ///     [1.0, 2.0]
    /// ];
    /// 
    /// let eigenvalues = a.eig();
    /// // For this matrix, eigenvalues should be 1 and 3
    /// ```
    pub fn eig(&self) -> Vector<T> {
        assert_eq!(self.rows, self.cols, "!!!Matrix must be square!!!");

        let n = self.rows;
        let mut a = self.clone();

        let max_iter = 1000;
        let eps = T::from_f64(1e-10);

        for _ in 0..max_iter {
            let (q, r) = a.qr();
            a = r * &q;

            // the convergence check
            let mut off_diag_norm = T::default();
            for i in 0..n {
                for j in 0..i {
                    off_diag_norm = off_diag_norm + a[[i, j]].abs();
                }
            }

            if off_diag_norm < eps {
                break;
            }
        }

        let mut eigenvalues = Vec::with_capacity(n);
        for i in 0..n {
            eigenvalues.push(a[[i, i]]);
        }

        Vector::from(eigenvalues)
    }


    /// Finds orthonormal eigen vectors
    /// 
    /// Will return matrix where each row is eigenvector
    ///
    /// Can panic if matrix is not square
    /// 
    /// # Example
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// 
    /// let a = matrix![
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 2.0, 1.0],
    ///     [0.0, 1.0, 2.0]];
    /// 
    /// println!("{}", a.eig_vectors());
    /// // [{1 0 0},                     ≈ v₂ = [1, 0, 0]
    /// // {0 0.707106781, -0.707106781}, ≈ v₃ = [0, 1/√2, -1/√2]
    /// //{0 0.707106781, 0.707106781,}]  ≈ v₁ = [0, 1/√2, 1/√2]
    /// ```
    pub fn eig_vectors(&self) -> Matrix<T>{
        assert_eq!(self.rows, self.cols, "!!!Matrix must be square!!!");

        let n = self.rows;
        let mut a = self.clone();
        let mut v = Matrix::identity(T::one(), n, n);

        let max_iter = 1000;
        let eps = T::from_f64(1e-10);

        for _ in 0..max_iter {
            let (q, r) = a.qr();
            a = r * &q;
            v = v * &q;

            let mut off_diag = T::default();
            for i in 0..n {
                for j in 0..i {
                    off_diag = off_diag + a[[i, j]].abs();
                }
            }

            if off_diag < eps {
                break;
            }
        }

        v
    }

    pub fn schur(&self, iters: usize) -> (Matrix<T>, Matrix<T>) {
        assert_eq!(self.rows, self.cols, "Matrix must be square.");

        let n = self.rows;

        let mut a = self.clone();
        let mut q_total = Matrix::identity(T::one(), n, n);

        for _ in 0..iters {
            let (q, r) = a.qr();

            // A_{k+1} = R Q
            a = &r * &q;

            q_total = &q_total * &q;
        }

        (q_total, a)
    }

    fn powf_schur_blocks(&self, exp: T) -> Matrix<T> {
        assert_eq!(self.rows, self.cols);

        let n = self.rows;
        let mut result = Matrix::from_num(T::default(), n, n);

        let mut i = 0;
        while i < n {
            // Check the 2x2 block (in Schur form this is a block with complex eigenvalues)
            if i + 1 < n && self[[i + 1, i]].abs() > T::from_f64(1e-10) {
                // A 2x2 block in Schur form
                let a = self[[i, i]];
                let b = self[[i, i + 1]];
                let c = self[[i + 1, i]];
                let d = self[[i + 1, i + 1]];

                // In Schur form a 2x2 block has the shape [a, b; c, d] with c != 0
                // and corresponds to a pair of complex-conjugate eigenvalues
                
                // Eigenvalues: lambda = alpha +- i*beta, where alpha = (a+d)/2, beta = sqrt(-bc)
                let alpha = (a + d) / T::from_usize(2);
                let beta = (-b * c).sqrt();
                
                // The modulus and the argument of a complex number
                let r = (alpha * alpha + beta * beta).sqrt();
                let theta = beta.atan2(alpha);
                
                let r_pow = r.powf(exp);
                let new_theta = theta * exp;
                
                let new_alpha = r_pow * new_theta.cos();
                let new_beta = r_pow * new_theta.sin();
                
                result[[i, i]] = new_alpha;
                result[[i, i + 1]] = -new_beta;
                result[[i + 1, i]] = new_beta;
                result[[i + 1, i + 1]] = new_alpha;

                i += 2;
            } else {
                result[[i, i]] = self[[i, i]].powf(exp);
                i += 1;
            }
        }

        result
    }

    pub fn schur_with_convergence(&self, max_iters: usize, eps: T) -> (Matrix<T>, Matrix<T>) {
        assert_eq!(self.rows, self.cols, "Matrix must be square.");

        let n = self.rows;
        let mut a = self.clone();
        let mut q = Matrix::identity(T::one(), n, n);

        for _ in 0..max_iters {
            // Work out the shift (using the last diagonal element)
            let mu = a[[n - 1, n - 1]];
            
            // A shifted QR decomposition: A - mu*I = QR
            let mut a_shifted = a.clone();
            for i in 0..n {
                a_shifted[[i, i]] = a_shifted[[i, i]] - mu;
            }
            
            let (q_k, r) = a_shifted.qr();
            
            // A_{k+1} = RQ + μI
            a = &r * &q_k;
            for i in 0..n {
                a[[i, i]] = a[[i, i]] + mu;
            }
            
            // Accumulating the transformations
            q = &q * &q_k;

            // The convergence check (the off-diagonal elements are small)
            let mut converged = true;
            for i in 0..n {
                for j in 0..i {
                    if a[[i, j]].abs() > eps {
                        converged = false;
                        break;
                    }
                }
                if !converged { break; }
            }
            
            if converged {
                // println!("Schur converged after {} iterations", iter);
                break;
            }
        }

        (q, a)
    }

    fn diagonalize(&self) -> Option<(Matrix<T>, Vector<T>, Matrix<T>)> {
        assert_eq!(self.rows, self.cols, "Matrix must be square.");

        let _n = self.rows;

        let eigen_values = self.eig();
        let p_matrix = self.eig_vectors();

        if let Ok(p_inv) = p_matrix.inv() {
            Some((p_matrix.clone(), eigen_values, p_inv))
        } else {
            None
        }
    }

    pub fn powf(&self, exp: T) -> Matrix<T> {
        assert_eq!(self.rows, self.cols, "!!!Matrix must be square.!!!");

        let n = self.rows;

        if exp == T::default() {
            return Matrix::identity(T::one(), n, n);
        }
        if exp == T::one() {
            return self.clone();
        }


        if let Some((p, d, p_inv)) = self.diagonalize() {
            let d_pow = Matrix::from_diag(d.map_vec(|x| x.powf(exp)), n, n);
            &p * &(&d_pow * &p_inv)
        } else {
            let (q, t) = self.schur_with_convergence(200, T::from_f64(1e-10));
            let t_pow = t.powf_schur_blocks(exp);
            &q * &t_pow * &q.transpose()
        }
    }

    /// Computes the mean and variance of all matrix elements in a single pass.
    ///
    /// # Examples
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    ///
    /// let m = matrix![
    ///     [1.0f32, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0]
    /// ];
    ///
    /// let (mean, variance) = m.mean_var();
    /// assert_eq!(mean, 3.5);
    /// assert!((variance - 2.916666).abs() < 1e-6);
    /// ```
    pub fn mean_var(&self) -> (T, T) {
        let rows = self.rows;
        let cols = self.cols;

        let (sum, sum_eq) = self.data
            .par_chunks(cols)
            .map(|chunk| {
                let mut chunk_sum = T::default();
                let mut chunk_sum_eq = T::default();

                for &value in chunk {
                    chunk_sum += value;
                    chunk_sum_eq += value * value;
                }

                (chunk_sum, chunk_sum_eq)
            })
            .reduce(|| (T::default(), T::default()),
             |(sum1, sq1), (sum2, sq2)| (sum1 + sum2, sq1 + sq2)
            );

            let n = T::from_usize(rows * cols);

            let mean = sum / n;

            let variance = (sum_eq / n) - mean * mean;
            (mean, variance)
    }

    /// Computes the variance of all matrix elements.
    ///
    /// # Examples
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    ///
    /// let m = matrix![
    ///     [1.0f32, 2.0],
    ///     [3.0, 4.0]
    /// ];
    ///
    /// let variance = m.var();
    /// assert!((variance - 1.25).abs() < 1e-10);
    /// ```
    pub fn var(&self) -> T {
        self.mean_var().1
    }

    /// Computes the mean (average) of all matrix elements.
    ///
    /// # Examples
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// 
    /// let m = matrix![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0],
    ///     [5.0, 6.0]
    /// ];
    ///
    /// let mean = m.mean();
    /// assert_eq!(mean, 3.5);
    /// ```
    pub fn mean(&self) -> T {
        self.mean_var().0
    }

    /// Computes the mean of each row in the matrix.
    ///
    /// # Examples
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::{Vector, Matrix};
    ///
    /// let m = matrix![
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0],
    ///     [7.0, 8.0, 9.0]
    /// ];
    ///
    /// let row_means = m.mean_rows();
    /// let expected = Vector::from(vec![2.0, 5.0, 8.0]);
    /// assert_eq!(row_means, expected);
    /// ```
    pub fn mean_rows(&self) -> Vector<T> {
        let rows = self.rows;
        let cols = self.cols;
        
        let row_means: Vec<T> = (0..rows)
            .into_par_iter()
            .map(|row_idx| {
                let start = row_idx * cols;
                let end = start + cols;
                let mut sum = T::default();
                
                for i in start..end {
                    sum += self.data[i];
                }
                
                sum / T::from_usize(cols)
            })
            .collect();

        Vector::from(row_means)
    }

    /// Computes the mean of each column in the matrix.
    ///
    /// # Examples
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::{Vector, Matrix};
    ///
    /// let m = matrix![
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0]
    /// ];
    ///
    /// let col_means = m.mean_cols();
    /// let expected = Vector::from(vec![2.5, 3.5, 4.5]);
    /// assert_eq!(col_means, expected);
    /// ```
    pub fn mean_cols(&self) -> Vector<T> {
        let rows = self.rows;
        let cols = self.cols;
        
        let zero_vec = vec![T::default(); cols];
        
        let column_sums: Vec<T> = (0..rows)
            .into_par_iter()
            .fold(
                || zero_vec.clone(),
                |mut acc, row_idx| {
                    let start = row_idx * cols;
                    for col_idx in 0..cols {
                        acc[col_idx] += self.data[start + col_idx];
                    }
                    acc
                }
            )
            .reduce(
                || vec![T::default(); cols],
                |mut acc, sums| {
                    for i in 0..cols {
                        acc[i] += sums[i];
                    }
                    acc
                }
            );
        
        let column_means: Vec<T> = column_sums
            .into_iter()
            .map(|sum| sum / T::from_usize(rows))
            .collect();

        Vector::from(column_means)
    }

    /// Computes the variance of each row in the matrix.
    ///
    /// # Examples
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::{Vector, Matrix};
    ///
    /// let m = matrix![
    ///     [1.0f32, 2.0, 3.0],
    ///     [4.0, 4.0, 4.0]
    /// ];
    ///
    /// let row_vars = m.var_rows();
    /// let expected = Vector::from(vec![2.0/3.0, 0.0]);
    /// assert!((row_vars[0] - expected[0]).abs() < 1e-10);
    /// ```
    pub fn var_rows(&self) -> Vector<T> {
        let rows = self.rows;
        let cols = self.cols;
        
        let row_means: Vec<T> = (0..rows)
            .into_par_iter()
            .map(|row_idx| {
                let start = row_idx * cols;
                let end = start + cols;
                let mut sum = T::default();
                
                for i in start..end {
                    sum += self.data[i];
                }
                
                sum / T::from_usize(cols)
            })
            .collect();
        
        let row_vars: Vec<T> = (0..rows)
            .into_par_iter()
            .map(|row_idx| {
                let start = row_idx * cols;
                let end = start + cols;
                let mean = row_means[row_idx];
                
                let mut sum_sq = T::default();
                for i in start..end {
                    let diff = self.data[i] - mean;
                    sum_sq += diff * diff;
                }
                
                sum_sq / T::from_usize(cols)
            })
            .collect();

        Vector::from(row_vars)
    }
    
    /// Computes the variance of each column in the matrix.
    ///
    /// # Examples
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::{Vector, Matrix};
    ///
    /// let m = matrix![
    ///     [1.0f32, 4.0],
    ///     [2.0, 4.0],
    ///     [3.0, 4.0]
    /// ];
    ///
    /// let col_vars = m.var_cols();
    /// let expected = Vector::from(vec![2.0/3.0, 0.0]);
    /// assert!((col_vars[0] - expected[0]).abs() < 1e-10);
    /// ```
    pub fn var_cols(&self) -> Vector<T> {
        let rows = self.rows;
        let cols = self.cols;
        
        // The column means first
        // let col_means = self.mean_cols();
        
        // For the variance, Welford's online algorithm
        let variances: Vec<T> = (0..cols)
            .into_par_iter()
            .map(|col_idx| {
                let mut mean = T::default();
                let mut m2 = T::default();
                
                for row_idx in 0..rows {
                    let value = self.data[row_idx * cols + col_idx];
                    let delta = value - mean;
                    mean += delta / (T::from_usize(row_idx) + T::one());
                    let delta2 = value - mean;
                    m2 += delta * delta2;
                }
                
                m2 / T::from_usize(rows)
            })
            .collect();

        Vector::from(variances)
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::Matrix;
    use crate::linalg::Vector;
    use crate::{vector, DataType};
    use std::time::Instant;

    #[test]
    fn powf_matrix() {
        let a = Matrix::from_diag(vector![1.0, 4.0, 9.0], 3, 3);

        println!("{}", a.powf(0.5));

        let a = matrix![
            [2.0, 1.0],
            [1.0, 2.0]
        ];
        
        let result = a.powf(0.5);
        
        let result_squared: Matrix<f64> = &result * &result;
        println!("{}", result_squared.map(|x| x.round()));

        let a = matrix![
            [4.0, 1.0],
            [0.0, 9.0]
        ];
        
        let result = a.powf(1.0);
        println!("{result}");

        let a = matrix![
            [4.0, 1.0],
            [0.0, 9.0]
        ];
        
        let result = a.powf(0.0);

        println!("{}", result);

        let a = matrix![
            [1.0, 1.0],
            [0.0, 1.0]
        ];

        println!("{}", a.powf(3.0))
    }

    #[test]
    fn from_fn() {
        let matrix = Matrix::<i32>::from_fn(3, 3, |i, j| (i * 10 + j) as i32);

        assert_eq!(matrix.data, vec![0, 1, 2, 10, 11, 12, 20, 21, 22]);
    }

    #[test]
    fn mat_and_scalar() {
        let a = matrix![[1.0]];
        let b = matrix![[1.0, 2.0], [3.0, 4.0]];
        println!("{}", b * &a);
    }
    #[test]
    fn mat_and_num() {
        let a = matrix![[1, 2], [1, 2]];
        let b = matrix![[2, 1], [3, 4]];
        println!("{}", 1 * &a * &b);
    }

    #[test]
    fn calc_num_test() {
        let a = matrix![[1]];
        let b = 10;
        assert_eq!(b * a, matrix![[10]]);

        let a = matrix![[1]];
        let b = 10;
        assert_eq!(b - a, matrix![[9]]);

        let a = matrix![[1]];
        let b = 10;
        assert_eq!(b + a, matrix![[11]]);
    }
    #[test]
    fn mul_test() {
        let a = matrix![[1, 2, 3], [4, 5, 6]];
        let b = matrix![[1, 2], [3, 4], [5, 6]];
        let ans = matrix![[22, 28], [49, 64]];
        assert_eq!(ans, a.clone() * &b);
        let ans = matrix![[9, 12, 15], [19, 26, 33], [29, 40, 51]];
        assert_eq!(ans, b * &a);

        let a = matrix![[0, -1], [1, 0]];
        let b = matrix![[0, 1], [-1, 0]];
        assert_eq!(Matrix::identity(DataType::i32(), 2, 2), a * &b)
    }

    #[test]
    fn mul_max_vector() {
        let a = matrix![[0, 0, 0], [1, 1, 1], [2, 2, 2]];
        let b = vector![1, 2, 3];
        assert_eq!(vector![0, 6, 12], a * &b);
    }

    #[test]
    fn resize_matrix() {
        let a = matrix![[1.0, 1.0], [2.0, 3.0]];
        let a_bigger = matrix![[1.0, 1.0, 0.0], [2.0, 3.0, 0.0]];
        let a_less = matrix![[1.0]];

        assert_eq!(a.get_resize(2, 3), a_bigger);
        assert_eq!(a.get_resize(1, 1), a_less);
    }

    #[test]
    fn parallel_computation() {
        let num = 2usize;
        let a = Matrix::identity(DataType::i16(), num, num);
        let b = Matrix::from_num(1i16, num, num);
        //parallel
        let start_time = Instant::now();
        let _ans = a * &b;
        let elapsed_time = start_time.elapsed();
        println!("Time: {} micros", elapsed_time.as_micros());

        let num = 2usize;
        let mut a = Matrix::identity(DataType::i16(), num, num);
        let b = Matrix::from_num(10i16, num, num);

        let start_time = Instant::now();
        a *= &b;
        let elapsed_time = start_time.elapsed();
        println!("Time: {} micros", elapsed_time.as_micros());
    }
    #[test]
    fn det_test() {
        let a = matrix![[3.0, 7.0], [1.0, -4.0]];

        assert_eq!(-19.0, a.det());
    }

    #[test]
    fn inv_matrix() {
        let a = matrix![[1.0, 2.0], [3.0, 4.0]];
        let b = a.inv().unwrap();
        let single = Matrix::identity(DataType::f64(), 2, 2);
        assert_eq!(single, a * &b);
    }

    #[test]
    fn inv_error() {
        let a = Matrix::from_num(3.0, 1, 2);
        assert!(a.inv().is_err());
    }

    #[test]
    fn add_col() {
        let mut a = matrix![[1, 2, 3], [3, 4, 5]];
        let column = vec![7, 8];
        a.add_column(column);
        let right = matrix![[1, 2, 3, 7], [3, 4, 5, 8]];
        assert_eq!(a, right);
    }

    #[test]
    fn add_row() {
        let mut a = matrix![[1, 2, 3], [4, 5, 6]];
        let row = vec![7, 8, 9];
        a.add_row(row);
        let right = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        assert_eq!(a, right);
    }

    #[test]
    #[should_panic]
    fn add_col_err() {
        let mut a = matrix![[1, 2, 3], [3, 4, 5]];
        let column = vec![7, 8, 9];
        a.add_column(column);
    }

    #[test]
    #[should_panic]
    fn add_row_err() {
        let mut a = matrix![[1, 2, 3], [4, 5, 6]];
        let row = vec![7, 8, 9, 10];
        a.add_row(row);
    }

    #[test]
    fn one_multi() {
        let one = Matrix::identity(1.0, 3, 3);
        let some = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let same = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        assert_eq!(some * &one, same);
    }

    #[test]
    fn matrix_norm_one() {
        let a = matrix![[-3.0, 5.0, 7.0], [2.0, 6.0, 4.0], [0.0, 2.0, 8.0]];
        assert_eq!(19.0, a.norm(1.0));
    }
    #[test]
    fn matrix_norm_inf() {
        let a = matrix![[-3.0, 5.0, 7.0], [2.0, 6.0, 4.0], [0.0, 2.0, 8.0]];
        assert_eq!(15.0, a.norm_inf());
    }

    #[test]
    fn matrix_norm_swap() {
        let a = matrix![[1.0], [3.0]];
        assert_eq!(4.0, a.norm(1.0));
    }

    #[test]
    fn matrix_norm_inf_swap() {
        let a = matrix![[1.0, 3.0]];
        assert_eq!(4.0, a.norm_inf());
    }

    #[test]
    fn macro_test() {
        let a = matrix!([1, 2, 3], [1, 2, 3], [1, 2, 3]);
        let b = Matrix::new(vec![1, 2, 3, 1, 2, 3, 1, 2, 3], 3, 3);
        assert_eq!(b, a);
    }

    #[test]
    fn create_matrix() {
        let _a = Matrix::from_num(10, 2, 2);
    }

    #[test]
    fn sum_matrix() {
        let a = matrix![[1.0, 1.0], [1.0, 1.0]];
        assert_eq!(4.0, a.sum());
    }

    #[test]
    fn single_matrix() {
        let a: Matrix<f64> = Matrix::identity(1.0, 2, 2);
        let b = matrix![[1.0, 0.0], [0.0, 1.0]];
        assert_eq!(a, b);
    }

    #[test]
    fn from_vector() {
        let a = Vector::from_num(5, 5);
        let b = Matrix::from(a);
        let a = matrix!([5, 5, 5, 5, 5]);
        assert_eq!(a, b);
    }

    #[test]
    fn get_cols() {
        let a = matrix![[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]];
        let b = Vector::from_num(4, 3);
        assert_eq!(b, a.get_col(3));
    }

    #[test]
    fn get_rows() {
        let a = matrix![[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]];
        let b = Vector::from(vec![1, 2, 3, 4]);
        assert_eq!(a.get_row(2), b)
    }

    #[test]
    fn transpose_matrix() {
        let a = Matrix::identity(1, 3, 3);
        let a = a.transpose();
        let b = Matrix::identity(1, 3, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn add_matrix() {
        let a = Matrix::from_num(1, 1, 1);
        let a1 = Matrix::from_num(2, 1, 1);

        let ans = Matrix::from_num(3, 1, 1);
        assert_eq!(ans, a + &a1);
    }

    #[test]
    fn add_matrix_and_num() {
        let ans = Matrix::from_num(3, 1, 1);
        let b = Matrix::from_num(2, 1, 1);

        assert_eq!(ans, b + 1);
    }

    #[test]
    fn mul_matrix() {
        let a = Matrix::new(vec![0, 1, -1, 0], 2, 2);
        let a1 = Matrix::new(vec![0, 1, -1, 0], 2, 2);

        let ans = Matrix::new(vec![-1, 0, 0, -1], 2, 2);
        assert_eq!(ans, a * &a1);
    }

    #[test]
    fn sub_matrix() {
        let a = matrix![[1.0, 2.0]];
        let b = matrix![[1.0, 2.0]];

        let ans = matrix![[0.0, 0.0]];

        assert_eq!(a - &b, ans);
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

        let mul = matrix![[-2.0, 36.0, 52.0], [-39.0, 48.0, 76.0], [-17.0, 31.0, 61.0]];

        assert_eq!(a * &b, mul);
    }

    #[test]
    fn mul_many_times() {
        let mut a = matrix![[2, 0], [0, 2]];
        let b = matrix![[2, 0], [0, 2]];

        a = a * &b;
        a = a * &b;
        assert_eq!(b * 2 * 2, a);
    }

    #[test]
    fn sub_many_matrix() {
        let mut a = matrix![[2.0, 2.0]];
        let b = matrix![[1.0, 1.0]];

        let ans = matrix![[0.0, 0.0]];

        a = a - &b;
        a = a - &b;
        assert_eq!(ans, a);
    }

    #[test]
    fn add_assign_many_times() {
        let mut a = Matrix::from_num(0.0, 1, 2);
        let mut b = a.clone();
        b += 1.0;
        b += 1f64;
        a += &b;
        a += &b;
        let answer = matrix![[4.0, 4.0]];
        assert_eq!(a, answer)
    }

    #[test]
    fn sub_assign_many_times() {
        let mut a = Matrix::from_num(2.0, 1, 2);
        let mut b = a.clone();
        b -= 0.5;
        b -= 0.5f64;
        a -= &b;
        a -= &b;
        let answer = matrix![[0.0, 0.0]];
        assert_eq!(a, answer)
    }

    #[test]
    fn mul_assign_many_times() {
        let mut one = matrix![[1.0, 0.0], [0.0, 1.0]];
        let mut imaginary = matrix![[0.0, 1.0], [1.0, 0.0]];
        imaginary *= -1.0;
        imaginary *= -1.0;

        one *= &imaginary;
        one *= &imaginary;

        let answer = Matrix::identity(DataType::f64(), 2, 2);
        assert_eq!(one, answer);
    }

    #[test]
    fn math_help() {
        let mut a = matrix![[9.0, -3.0, 1.0], [4.0, -2.0, 1.0], [16.0, -4.0, 1.0]];
        let b = matrix![[-5.0], [-4.0], [-4.0]];
        a = a.inv().unwrap();
        println!("{}", a);
        println!("{}", a * &b);
    }

    #[test]
    fn cramer_test() {
        let a = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 1.0]];
        let ans = matrix![[14.0, 32.0, 50.0]].transpose();

        let det = a.det();

        let deters = vec![
            matrix![[14.0, 2.0, 3.0], [32.0, 5.0, 6.0], [50.0, 8.0, 1.0]].det(),
            matrix![[1.0, 14.0, 3.0], [4.0, 32.0, 6.0], [7.0, 50.0, 1.0]].det(),
            matrix![[1.0, 2.0, 14.0], [4.0, 5.0, 32.0], [7.0, 8.0, 50.0]].det(),
        ];

        let mut b = Vector::from_num(0f64, 3);
        for i in 0..deters.len() {
            let b_int = (deters[i] / det) as i32;
            b[i] = b_int as f64;
        }

        assert_eq!(b, (a.inv().unwrap() * &ans).get_col(0));
    }

    #[test]
    fn sum_of_rows() {
        let a = matrix![[1, 2, 3], [4, 5, 6]];
        println!("{}", a.sum_rows());
    }

    #[test]
    fn comparisons() {
        let a = matrix![[-1, 0, 1]];
        assert_eq!(a.compare(Matrix::from_num(0, 1, 3)), matrix![[-1, 0, 1]]);
    }

    #[test]
    fn comparisons_num() {
        let a = matrix![[-1, 0, 1]];
        assert_eq!(a.compare_num(0), matrix![[-1, 0, 1]])
    }

    #[test]
    fn map_test() {
        let a = matrix![[1, 2], [3, 4]];
        let b = a.map(|x| x % 2);
        println!("{b}");
    }

    #[test]
    fn num_mat_mul() {
        let a = matrix![[1.0, 2.0, 3.0]];
        let b = matrix![[2.0, 4.0, 6.0]];
        assert_eq!(2.0 * &a, b);
        assert_eq!(a * 2.0, b);
    }

    #[test]
    fn randn() {
        let a: Matrix<f64> = Matrix::randn(2, 2);
        println!("{a}");
    }

    #[test]
    fn rem_col() {
        let a = matrix![[1, 2, 3], [4, 5, 6]];
        println!("{}", a.rem_col(2));
    }

    #[test]
    fn mul_a() {
        let mut a = matrix![[1.0, 2.0], [3.0, 4.0]];
        let b = Matrix::from_num(1.0, 2, 3);

        a *= &b;
        println!("{a}");
    }

    #[test]
    fn diag_test() {
        let r = 3;
        let c = 1;
        let a = Matrix::new((0..(r*c) as i32).collect(), r, c);
        println!("{a}");
        println!("{}", a.diag());
    }

    #[test]
    fn diag_mx() {
        let v = vector![1, 2, 3];
        let b = Matrix::from_diag(v, 4, 3);
        println!("{b}");
    }

    #[test]
    fn sub_mx() {
        let a = matrix![[1,2,3]];
        println!("{}", 1 - a);
    }

    #[test]
    fn div_mx() {
        let mut a = matrix![[1.0,2.0,4.0]];
        println!("{}", &a / 2.0);
        println!("{}", 2.0 / &a);
        a /= 2.0;
        println!("{a}");

    }

    #[test]
    fn test_conv_3x3_identity_kernel() {
        let matrix = matrix![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];

        let kernel = matrix![[1.0]]; // a 1x1 identity kernel

        let result = matrix.conv(&kernel);
        
        assert_eq!(result.rows, 3);
        assert_eq!(result.cols, 3);
        assert_eq!(result.data, matrix.data);
    }

    #[test]
    fn test_conv_edge_detection_sobel() {
        let matrix = matrix![
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ];

        let sobel_x = matrix![
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ];

        let result = matrix.conv(&sobel_x);
        
        println!("Sobel X result:");
        println!("{}", result);
        
        // Check the sizes
        assert_eq!(result.rows, 3);
        assert_eq!(result.cols, 3);
        
        // In uniform regions it has to be 0
        assert_eq!(result.data[1 * result.cols + 1], -3.0);
    }

    #[test]
    fn iters() {
        let a: Matrix<f32> = Matrix::randn(5, 3);
        println!("{a}");
        for i in a.indexed_iter() {
            println!("{:?}", i);
        }
    }
}
