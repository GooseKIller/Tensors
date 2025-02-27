use crate::linalg::{Tensor, Vector};
use crate::{Float, Num};
use rand::distributions::{Distribution, Standard};
use rand::random;
use rayon::prelude::IntoParallelRefMutIterator;
use rayon::prelude::*;
use std::cmp::min;
use std::fmt::{Display, Formatter};
use std::ops::{Index, IndexMut};

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
/// When performing operations, ensure to use a reference for the second operand:
///
/// # Example
/// ```rust
/// use tensors::linalg::Matrix;
///
/// let a = Matrix::from_num(0, 2, 2);
/// let b = Matrix::from_num(1, 2, 2);
///
/// a + &b; // Correct
/// // a + b;  // Incorrect
/// ```
///
/// All Matrix operations
///
/// | name |operation|
/// |------|---------|
/// | plus | + |
/// |plus assign| += |
/// |minus | - |
/// |minus assign| -=|
/// |mul| * |
/// |mul assign| *= |
/// |hadamard(element-wise)| & |
/// |hadamard(element-wise) assign| &=|
#[derive(PartialEq, Eq, Debug)]
pub struct Matrix<T: Num> {
    pub(crate) data: Vec<T>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl<T: Num> Matrix<T> {
    /// Create matrix from vector and usize, usize
    /// # Example
    /// ```
    /// use tensors::linalg::Matrix;
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
    /// use tensors::linalg::Matrix;
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
    /// use tensors::linalg::Matrix;
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

    pub fn try_from(value: Tensor<T>) -> Result<Self, &'static str> {
        if value.shape.len() != 2 {
            return Err("Shape size must be 2");
        }
        Ok(Matrix::new(value.data, value.shape[0], value.shape[1]))
    }

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
    /// need to implement type
    ///
    /// # Example
    /// ```
    /// use tensors::DataType;
    /// use tensors::linalg::Matrix;
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
    /// use tensors::linalg::Matrix;
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
    /// use tensors::matrix;
    /// use tensors::linalg::Matrix;
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
    pub fn sum(self) -> T {
        let mut sum = T::default();
        for i in self.data {
            sum += i;
        }
        sum
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
    /// # Example
    /// ```
    /// use tensors::matrix;
    /// use tensors::linalg::Matrix;
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
    /// # Example
    /// ```
    /// use tensors::linalg::{Matrix, Vector};
    /// use tensors::matrix;
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
    /// use tensors::matrix;
    /// use tensors::linalg::Matrix;
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

    pub fn size(&self) -> [usize; 2] {
        [self.rows.clone(), self.cols.clone()]
    }

    /// Adds column at the end of Matrix
    ///
    /// # Example
    ///
    /// ```
    /// use tensors::matrix;
    /// use tensors::linalg::Matrix;
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
            "!!!the length of the Vec<T> is not equal to the size of the rows of the matrix!!!"
        );
        for i in 0..self.rows {
            self.data.insert((i + 1) * self.cols + i, column[i].clone());
        }
        self.cols += 1;
    }

    /// Adds row at the end of Matrix
    ///
    /// # Example
    ///
    /// ```
    /// use tensors::matrix;
    /// use tensors::linalg::Matrix;
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
    /// use tensors::matrix;
    /// use tensors::linalg::Matrix;
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

    pub fn sum_rows(&self) -> Matrix<T> {
        let mut matrix = vec![T::default(); self.cols];
        for i in 0..self.cols {
            matrix[i] = self.get_col(i).sum();
        }
        Matrix::new(matrix, 1, self.cols)
    }

    /// Comparison of two matrices
    ///
    /// panics if their shapes are different
    /// # Example
    /// ```
    /// use tensors::matrix;
    /// use tensors::linalg::Matrix;
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
        for i in 0..self.data.len() {
            if self.data[i] > other.data[i] {
                comparisons[i] = T::from(1);
            } else if self.data[i] < other.data[i] {
                comparisons[i] = T::from(1).neg();
            }
        }
        Matrix::new(comparisons, self.rows, self.cols)
    }

    /// Comparison of matrix and number
    ///
    /// # Example
    ///```
    /// use tensors::linalg::Matrix;
    /// use tensors::matrix;
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
    /// # Example
    ///```
    /// use tensors::linalg::Matrix;
    /// use tensors::matrix;
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
    /// use tensors::linalg::Matrix;
    /// use tensors::matrix;
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
    /// use tensors::matrix;
    /// use tensors::linalg::Matrix;
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
    /// use tensors::linalg::Matrix;
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
    /// use tensors::linalg::Matrix;
    /// use tensors::matrix;
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
    /// use tensors::linalg::Matrix;
    /// use tensors::matrix;
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

    /// makes diagonal from matrix
    pub fn diag(&self) -> Vector<T> {
        let mut data = vec![T::default(); min(self.cols, self.rows)];
        for i in 0..min(self.cols, self.rows) {
            let index = (self.cols * i) + i;
            data[i] = self.data[index];
        }
        Vector::from(data)
    }

    /// Function for saving
    pub(crate) fn data_as_string(&self) -> String {
        self.data
            .iter()
            .map(|x| format!("{x}"))
            .collect::<Vec<_>>()
            .join(" ")
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
        assert_eq!(value.shape.len(), 2, "Shape size must be 2");
        Self {
            data: value.data,
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
    pub fn randn(row: usize, col: usize) -> Self
    where
        Standard: Distribution<T>,
    {
        Self {
            data: vec![T::default(); row * col]
                .iter()
                .map(|_| {
                    (-T::from(2) * random::<T>().ln()).sqrt()
                        * (T::from(2) * T::pi() * random::<T>()).cos()
                })
                .collect(),
            rows: row,
            cols: col,
        }
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
}

#[cfg(test)]
mod tests {
    use crate::linalg::matrix::*;
    use crate::{vector, DataType};
    use std::time::Instant;

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
}
