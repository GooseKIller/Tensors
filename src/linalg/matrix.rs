use std::cmp::min;
use std::fmt::{Display, Formatter};
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use rayon::prelude::{IntoParallelRefMutIterator};
use rayon::prelude::*;
use crate::linalg::{Tensor, Vector};
use crate::{Float, Num};

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

///reference by (skyl4b)<https://github.com/TheAlgorithms/Rust/blob/master/src/math/matrix_ops.rs>
///
///Also since the matrix is implemented using vector, it is not simple struct, then all
///mathematical methods are realized without borrowing
///
///So you should use & with the second part of calculation
/// # Example
///
/// ```
/// use tensors::linalg::Matrix;
///
/// let a = Matrix::from_num(0, 2, 2);
/// let b = Matrix::from_num(1, 2, 2);
///
/// a + &b; // correct
/// // a + b;  // incorrect
/// ```
#[derive(PartialEq, Eq, Debug)]
pub struct Matrix<T:Num>{
    pub(crate) data: Vec<T>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
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

    pub fn try_from(value: Tensor<T>) -> Result<Self, &'static str> {
        if value.shape.len() != 1{
            return Err("Shape size must be 2")
        }
        Ok(Matrix::new(value.data,
                       value.shape[0],
                       value.shape[1]))
    }

    /// Creates a single matrix
    ///
    /// need to implement type
    ///
    /// # Example
    /// ```
    /// use tensors::DataType;
    /// use tensors::linalg::Matrix;
    /// let a:Matrix<f64> = Matrix::single(DataType::f64(), 2, 2);
    /// // will create matrix
    /// // [1 0]
    /// // [0 1]
    /// ```
    pub fn single(_: T, rows:usize, cols:usize) -> Self{
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
    pub fn get_data(&self) -> Vec<T>{
        self.data.clone()
    }

    /// Returns row count
    pub fn row(&self) -> usize{
        self.rows.clone()
    }

    /// Returns column count
    pub fn col(&self) -> usize{
        self.cols.clone()
    }

    pub fn sum(self) -> T{
        let mut sum = T::default();
        for i in self.data{
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
        if index >= self.cols {
            panic!("!!!Index:{} more or equal then columns count:{}!!!", index, self.cols);
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
    /// let col:Vector<i32> = example.get_row(1);//[1 2]
    /// ```
    pub fn get_row(&self, index:usize) -> Vector<T>{
        if index >= self.rows {
            panic!("!!!Index:{} greater or equal then rows count:{}.!!!", index, self.rows);
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

    pub fn size(&self) -> [usize; 2]{
        [self.rows.clone(), self.cols.clone()]
    }

    pub fn add_column(&mut self, column: Vec<T>) {
        if column.len() != self.rows{
            panic!("!!!the length of the Vec<T> is not equal to the size of the rows of the matrix!!!")
        }
        for i in 0..self.rows {
            self.data.insert((i + 1) * self.cols + i, column[i].clone());
        }
        self.cols += 1;
    }

    pub fn add_row(&mut self, row: Vec<T>) {
        if row.len() != self.cols{
            panic!("!!!the length of the Vec<T> is not equal to the size of the columns of the matrix!!!")
        }
        for i in row{
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
    pub fn get_resize(&self, new_row:usize, new_col:usize) -> Matrix<T>{
        let mut new_matrix = Matrix::from_num(T::default(), new_row, new_col);
        for i in 0..min(new_row, self.rows){
            for j in 0..min(new_col, self.cols){
                new_matrix[[i, j]] = self[[i, j]];
            }
        }

        new_matrix
    }
}

impl<T:Num> Index<[usize;2]> for Matrix<T> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let [i, j] = index;
        if i >= self.rows || j >= self.cols{
            panic!("!!!Matrix index out of bounds!!! Got [{i}, {j}] but excepted less than [{}, {}]", self.rows, self.cols);
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

impl<T:Num> Neg for Matrix<T>{
    type Output = Matrix<T>;
    fn neg(self) -> Self::Output {
        let mut data = vec![T::default(); self.rows*self.cols];
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x = -self.data[i];
        });
        Matrix::new(data, self.rows, self.cols)
    }
}

impl<T:Num> Add<&Matrix<T>> for Matrix<T>{
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols{
            panic!("!!!Matrix dimensions do not match!!!");
        }
        /*
        let mut result = Self::from_num(T::default(), self.rows, self.cols);
        for i in 0..self.rows{
            for j in 0..self.cols{
                result[[i, j]] = self[[i, j]] + rhs[[i, j]];
            }
        }
        result*/
        let mut data = vec![T::default(); self.rows*self.cols];
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.data[i] + rhs.data[i];
        });
        Matrix::new(data, self.rows, self.cols)
    }
}

impl<T:Num> AddAssign<&Matrix<T>> for Matrix<T>{
    fn add_assign(&mut self, rhs: &Matrix<T>) {
        if self.rows != rhs.rows || self.cols != rhs.cols{
            panic!("!!!Matrix dimensions do not match!!!");
        }
        /*
        for i in 0..self.rows{
            for j in 0..self.cols{
                self[[i, j]] += rhs[[i, j]];
            }
        }*/
        self.data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x += rhs.data[i];
        });
    }
}

impl<T:Num> Add<T> for Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, rhs:T) -> Self::Output {
        /*
        let mut result = Self::from_num(T::default(), self.rows, self.cols);
        for i in 0..self.rows{
            for j in 0..self.cols{
                result[[i, j]] = self[[i, j]] + rhs;
            }
        }
        result*/
        let mut data = vec![T::default(); self.rows*self.cols];
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.data[i] + rhs;
        });
        Matrix::new(data, self.rows, self.cols)
    }
}

impl<T:Num> AddAssign<T> for Matrix<T>{
    fn add_assign(&mut self, rhs: T) {
        /*
        for i in 0..self.rows{
            for j in 0..self.cols{
                self[[i, j]] += rhs;
            }
        }*/
        self.data.par_iter_mut().for_each(|x| {
            *x += rhs;
        });
    }
}

impl<T:Num> Sub<&Matrix<T>> for Matrix<T>{
    type Output = Matrix<T>;
    fn sub(self, rhs: &Matrix<T>) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols{
            panic!("!!!Matrix dimensions do not match!!!");
        }
        /*
        let mut result = Vec::with_capacity(self.rows*self.cols);
        for i in 0..self.rows*self.cols{
                result.push(self.data[i] - rhs.data[i]);
        }
        Matrix::new(result, self.rows, self.cols)
        */
        let mut data = vec![T::default(); self.rows*self.cols];
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.data[i] - rhs.data[i];
        });
        Matrix::new(data, self.rows, self.cols)
    }
}

impl<T:Num> SubAssign<&Matrix<T>> for Matrix<T>{
    fn sub_assign(&mut self, rhs: &Matrix<T>) {
        if rhs.rows == 1 && rhs.cols == 1{
            *self -= rhs.data[0];
            return;
        }
        if self.rows != rhs.rows || self.cols != rhs.cols{
            panic!("!!!Matrix dimensions do not match!!!");
        }
        self.data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x -= rhs.data[i];
        });
        /*
        for i in 0..self.rows{
            for j in 0..self.cols{
                self[[i, j]] -= rhs[[i, j]];
            }
        }*/
    }
}

impl<T:Num> Sub<T> for Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, rhs:T) -> Self::Output {
        /*
        let mut result = Self::from_num(T::default(), self.rows, self.cols);
        for i in 0..self.rows{
            for j in 0..self.cols{
                result[[i, j]] = self[[i, j]] - rhs;
            }
        }
        result*/
        let mut data = vec![T::default(); self.rows*self.cols];
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.data[i] - rhs;
        });
        Matrix::new(data, self.rows, self.cols)
    }
}

impl<T:Num> SubAssign<T> for Matrix<T>{
    fn sub_assign(&mut self, rhs: T) {
        self.data.par_iter_mut().for_each( |x| {
            *x -= rhs
        });
        /*
        for i in 0..self.rows{
            for j in 0..self.cols{
                self[[i, j]] -= rhs;
            }
        }*/
    }
}

impl<T:Num> Mul<&Vector<T>> for Matrix<T>{
    type Output = Vector<T>;
    fn mul(self, rhs: &Vector<T>) -> Self::Output {
        if self.cols != rhs.length {
            panic!("!!!Matrix amount of columns != Vector lengths\n\
             Matrix cols {} Vector cols {}!!!", self.cols, rhs.length)
        }
        let mut data = vec![T::default(); self.rows];
        data.par_iter_mut().enumerate().for_each(|(index, x)| {
            for i in 0..self.cols{
                *x += self[[index, i]] * rhs[i];
            }
        });
        Vector::from(data)
    }
}

impl<T:Num> Mul<&Vector<T>> for &Matrix<T>{
    type Output = Vector<T>;
    fn mul(self, rhs: &Vector<T>) -> Self::Output {
        if self.cols != rhs.length {
            panic!("!!!Matrix amount of columns != Vector lengths\n\
             Matrix cols {} Vector cols {}!!!", self.cols, rhs.length)
        }
        let mut data = vec![T::default(); self.rows];
        data.par_iter_mut().enumerate().for_each(|(index, x)| {
            for i in 0..self.cols{
                *x += self[[index, i]] * rhs[i];
            }
        });
        Vector::from(data)
    }
}

impl<T:Num> Mul<&Matrix<T>> for Matrix<T>{
    type Output = Matrix<T>;
    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        if self.cols != rhs.rows{
            panic!("!!!Matrix amount of columns 1st matrix != to amount of rows of the 2nd one!!!\n\
             Matrix cols: {} Other Matrix rows: {}",self.rows, self.cols)
        }

        let mut data = vec![T::default() ; self.rows*rhs.cols];
        data.par_iter_mut().enumerate().for_each(|(index, x)| {
            let (i, j)  = (index / rhs.cols, index % rhs.cols);
            let mut sum = T::default();
            for k in 0..self.cols {
                sum += self[[i, k]] * rhs[[k, j]];
            }
            *x = sum;
        });
        Self::new(data, self.rows, rhs.cols)
    }
}

impl<T:Num> Mul<&Matrix<T>> for &Matrix<T>{
    type Output = Matrix<T>;
    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        if self.cols != rhs.rows{
            panic!("!!!Matrix amount of columns 1st matrix != to amount of rows of the 2nd one!!!\n\
             Matrix cols: {} Other Matrix rows: {}",self.rows, self.cols)
        }

        let mut data = vec![T::default() ; self.rows*rhs.cols];
        data.par_iter_mut().enumerate().for_each(|(index, x)| {
            let (i, j)  = (index / rhs.cols, index % rhs.cols);
            let mut sum = T::default();
            for k in 0..self.cols {
                sum += self[[i, k]] * rhs[[k, j]];
            }
            *x = sum;
        });
        Matrix::new(data, self.rows, rhs.cols)
    }
}

impl<T:Num> MulAssign<&Matrix<T>> for Matrix<T>{
    /// WARNING if it is not square matrix sizes will change
    fn mul_assign(&mut self, rhs: &Matrix<T>) {
        if self.cols != rhs.rows{
            panic!("!!!Matrix amount of columns 1st matrix does not equals to amount of rows of the 2nd one!!!")
        }
        let mut result = Self::from_num(T::default(), self.rows, rhs.cols);
        result.data.par_iter_mut().enumerate().for_each(|(index, x)| {
            let (i, j)  = (index / rhs.cols, index % rhs.cols);
            let mut sum = T::default();
            for k in 0..self.cols {
                sum += self[[i, k]] * rhs[[k, j]];
            }
            *x = sum;
        });
        *self = result;
    }
}

impl<T:Num> Mul<T> for Matrix<T>{
    type Output = Matrix<T>;
    fn mul(self, rhs: T) -> Self::Output {
        /*
        let mut result = Self::from_num(T::default(), self.rows, self.cols);
        for i in 0..(self.rows*self.cols) {
            result.data[i] = self.data[i] * rhs;
        }
        result*/
        let mut data = vec![T::default(); self.rows*self.cols];
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.data[i] * rhs;
        });
        Matrix::new(data, self.rows, self.cols)
    }
}

impl<T:Num> MulAssign<T> for Matrix<T>{
    fn mul_assign(&mut self, rhs: T) {
        self.data.par_iter_mut().for_each(|x| {
            *x = *x * rhs;
        });
        /*
        for i in 0..self.rows {
            for j in 0..self.cols {
                self[[i,j]] = self[[i,j]] * rhs;
            }
        }*/
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

impl<T:Num> From<Tensor<T>> for Matrix<T>  {
    fn from(value: Tensor<T>) -> Self{
        if value.shape.len() != 2{
            panic!("Shape size must be 2")
        }
        Self{
            data: value.data,
            rows:value.shape[0],
            cols:value.shape[1]
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

impl <T:Num> Clone for Matrix<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            rows: self.rows.clone(),
            cols: self.cols.clone()
        }
    }
}


// Float Number implementation
impl<T:Float> Matrix<T> {
    pub fn norm(self, p:T) -> T{
        let one = 1.into();
        if p < one{
            panic!("!!!Number p:{} must be positive!!!", p);
        }
        let mut norm = T::default();

        if p == one{
            let mut max_num = self.get_col(0).abs_sum();

            for i in 1..self.cols{
                let sum = self.get_col(i).abs_sum();
                max_num = if sum > max_num {
                    sum
                } else {
                    max_num
                };
            }
            max_num

        } else {
            for i in self.data{
                norm += i.powf(p);
            }
            norm.powf(one/p)
        }
    }

    pub fn norm_inf(self) -> T{
        let mut max_num = self.get_row(0).abs_sum();

        for i in 1..self.rows{
            let sum = self.get_row(i).abs_sum();
            max_num = if sum > max_num {
                sum
            } else {
                max_num
            };
        }
        max_num
    }

    /// Finds determinant of matrix
    ///
    /// Using Gauss Method (<https://en.wikipedia.org/wiki/Gaussian_elimination>)
    ///
    /// O(N^3)
    pub fn det(&self) -> T{
        if self.rows != self.cols {
            panic!("!!!The determinant is defined only for square matrices!!!")
        }

        let mut matrix = self.clone();
        let mut det = 1.into();
        for i in 0..self.rows{
            for j in (i+1)..self.rows{
                let coefficient = matrix[[j, i]]/matrix[[i, i]];
                for k in i..self.rows{
                    matrix[[j, k]] = matrix[[j, k]] - coefficient * matrix[[i, k]];
                }
            }
            det = det * matrix[[i, i]];
        }
        det
    }

    //Need to optimize
    pub fn inv(&self) -> Result<Matrix<T>, &'static str>{
        if self.rows != self.cols{
            return Err("Matrix is not invertible");
        }
        let n = self.rows;
        let mut augmented_matrix = self.clone();
        let mut inv_matrix = Matrix::single(T::default(), self.rows, self.rows);

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

/*
impl<T:Num> Iterator for Matrix<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.data.pop()
    }
    
}*/

#[cfg(test)]
mod tests{
    use std::io;
    use crate::{DataType, vector};
    use crate::linalg::matrix::*;
    use std::time::Instant;

    #[test]
    fn mul_test(){
        let a = matrix![[1,2,3],[4,5,6]];
        let b = matrix![[1,2],[3,4],[5,6]];
        let ans = matrix![[22,28],[49,64]];
        assert_eq!(ans, a.clone() * &b);
        let ans = matrix![[9,12,15],[19,26,33],[29,40,51]];
        assert_eq!(ans, b * &a);

        let a = matrix![[0,-1],[1,0]];
        let b = matrix![[0,1],[-1,0]];
        assert_eq!(Matrix::single(DataType::i32(), 2, 2), a * &b)
    }

    #[test]
    fn mul_max_vector(){
        let a = matrix![[0,0,0],[1,1,1],[2,2,2]];
        let b = vector![1,2,3];
        assert_eq!(vector![0, 6, 12], a * &b);
    }

    #[test]
    fn resize_matrix(){
        let a = matrix![[1.0, 1.0],
                                    [2.0, 3.0]];
        let a_bigger = matrix![[1.0, 1.0, 0.0],
                                    [2.0, 3.0, 0.0]];
        let a_less = matrix![[1.0]];

        assert_eq!(a.get_resize(2, 3), a_bigger);
        assert_eq!(a.get_resize(1, 1), a_less);

    }


    #[test]
    fn parallel_computation(){
        let num = 2usize;
        let a = Matrix::single(DataType::i16(), num, num);
        let b = Matrix::from_num(1i16, num, num);
        //parallel
        let start_time = Instant::now();
        let _ans = a * &b;
        let elapsed_time = start_time.elapsed();
        println!("Time: {} micros", elapsed_time.as_micros());

        let num = 2usize;
        let mut a = Matrix::single(DataType::i16(), num, num);
        let b = Matrix::from_num(10i16, num, num);

        let start_time = Instant::now();
        a *= &b;
        let elapsed_time = start_time.elapsed();
        println!("Time: {} micros", elapsed_time.as_micros());

    }
    #[test]
    fn det_test(){
        let a = matrix![[3.0, 7.0],
                        [1.0, -4.0]];

        assert_eq!(-19.0, a.det());
    }

    #[test]
    fn inv_matrix(){
        let a = matrix![[1.0, 2.0],
                                    [3.0, 4.0]];
        let b = a.inv().unwrap();
        let single = Matrix::single(DataType::f64(), 2, 2);
        assert_eq!(single, a* &b);
    }

    #[test]
    fn inv_error(){
        let a = Matrix::from_num(3.0, 1, 2);
        assert!(a.inv().is_err());
    }

    #[test]
    fn add_col(){
        let mut a = matrix![[1,2,3],
            [3,4,5]];
        let column = vec![7,8];
        a.add_column(column);
        let right = matrix![[1,2,3,7],[3,4,5,8]];
        assert_eq!(a, right);
    }

    #[test]
    fn add_row(){
        let mut a = matrix![[1,2,3],
            [4,5,6]];
        let row = vec![7,8,9];
        a.add_row(row);
        let right = matrix![[1,2,3],[4,5,6],[7,8,9]];
        assert_eq!(a, right);
    }

    #[test]
    #[should_panic]
    fn add_col_err(){
        let mut a = matrix![[1,2,3],
            [3,4,5]];
        let column = vec![7,8, 9];
        a.add_column(column);
    }

    #[test]
    #[should_panic]
    fn add_row_err(){
        let mut a = matrix![[1,2,3],
            [4,5,6]];
        let row = vec![7,8,9,10];
        a.add_row(row);
    }

    #[test]
    fn one_multi(){
        let one = Matrix::single(1.0,3, 3);
        let some = matrix![[1.0,2.0,3.0],
            [4.0,5.0,6.0],
            [7.0,8.0,9.0]];
        let same = matrix![[1.0,2.0,3.0],
            [4.0,5.0,6.0],
            [7.0,8.0,9.0]];
        assert_eq!(some * &one, same);

    }

    #[test]
    fn matrix_norm_one(){
        let a = matrix![[-3.0, 5.0, 7.0],
                                    [2.0, 6.0, 4.0],
                                    [0.0, 2.0, 8.0]];
        assert_eq!(19.0, a.norm(1.0));

    }
    #[test]
    fn matrix_norm_inf(){
        let a = matrix![[-3.0, 5.0, 7.0],
                                    [2.0, 6.0, 4.0],
                                    [0.0, 2.0, 8.0]];
        assert_eq!(15.0, a.norm_inf());
    }

    #[test]
    fn matrix_norm_swap(){
        let a = matrix![[1.0],
                                    [3.0]];
        assert_eq!(4.0, a.norm(1.0));
    }

    #[test]
    fn matrix_norm_inf_swap(){
        let a = matrix![[1.0, 3.0]];
        assert_eq!(4.0, a.norm_inf());
    }

    #[test]
    fn macro_test(){
        let a = matrix!([1,2,3],[1,2,3],[1,2,3]);
        let b= Matrix::new(vec![1,2,3,1,2,3,1,2,3], 3, 3);
        assert_eq!(b, a);
    }

    #[test]
    fn create_matrix(){
        let a = Matrix::from_num(10, 2,2);
    }

    #[test]
    fn sum_matrix(){
        let a = matrix![[1.0, 1.0],
                                    [1.0, 1.0]];
        assert_eq!(4.0, a.sum());
    }

    #[test]
    fn single_matrix(){
        let a:Matrix<f64> = Matrix::single(1.0,2, 2);
        let b = matrix![[1.0,0.0],
            [0.0,1.0]];
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
        let a = Matrix::single(1, 3, 3);
        let a = a.transpose();
        let b = Matrix::single(1, 3, 3);
        assert_eq!(a, b);
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
    fn sub_matrix(){
        let a = matrix![[1.0, 2.0]];
        let b = matrix![[1.0, 2.0]];

        let ans = matrix![[0.0, 0.0]];

        assert_eq!(a-&b, ans);
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

    #[test]
    fn mul_many_times(){
        let mut a = matrix![[2,0],
                                        [0,2]];
        let b = matrix![[2,0],
                                    [0,2]];

        a = a * &b;
        a = a * &b;
        assert_eq!(b*2*2, a);
    }

    #[test]
    fn sub_many_matrix(){
        let mut a = matrix![[2.0, 2.0]];
        let b = matrix![[1.0, 1.0]];

        let ans = matrix![[0.0, 0.0]];

        a = a - &b;
        a = a - &b;
        assert_eq!(ans, a);
    }

    #[test]
    fn add_assign_many_times(){
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
    fn sub_assign_many_times(){
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
    fn mul_assign_many_times(){
        let mut one = matrix![[1.0, 0.0],
                                [0.0,1.0]];
        let mut imaginary = matrix![[0.0, 1.0],
                                    [1.0, 0.0]];
        imaginary *= -1.0;
        imaginary *= -1.0;

        one *= &imaginary;
        one *= &imaginary;

        let answer = Matrix::single(DataType::f64(), 2, 2);
        assert_eq!(one, answer);
    }

    #[test]
    fn math_help(){
        let mut a = matrix![[9.0, -3.0, 1.0],
                                    [4.0, -2.0, 1.0],
                                    [16.0, -4.0, 1.0]];
        let b = matrix![[-5.0],
                                    [-4.0],
                                    [-4.0]];
        a = a.inv().unwrap();
        println!("{}", a);
        println!("{}", a * &b);
    }

    #[test]
    fn cramer_test(){
        let a = matrix![[1.0,2.0,3.0],
                                    [4.0,5.0,6.0],
                                    [7.0,8.0,1.0]];
        let ans = matrix![[14.0,32.0,50.0]].transpose();

        let det = a.det();

        let deters = vec![
        matrix![[14.0,2.0,3.0],
                [32.0,5.0,6.0],
                [50.0,8.0,1.0]].det(),
        matrix![[1.0,14.0,3.0],
                [4.0,32.0,6.0],
                [7.0,50.0,1.0]].det(),
        matrix![[1.0,2.0,14.0],
                [4.0,5.0,32.0],
                [7.0,8.0,50.0]].det()];

        let mut b = Vector::from_num(0f64, 3);
        for i in 0..deters.len(){
            let b_int = (deters[i]/det) as i32;
            b[i] = b_int as f64;
        }

        assert_eq!(b, (a.inv().unwrap() * &ans).get_col(0));
    }

    #[test]
    fn focus_wierd(){
        let mut val = vec![0;5*5];
        for i in 0..(5*5){
            val[i] = (i as i32)+11;
        }
        let matrix = Matrix::new(val,5,5);
        println!("{}", matrix);
        println!("Remember your numbers\n\n\n\n\n\n\n\n\n");
        let matrix = matrix.transpose();
        println!("{matrix}");
        for i in 0..5{
            println!("which column has your {}st number", i+1);
            let mut str_index = String::new();
            io::stdin().read_line(&mut str_index).expect("Enter a number");
            let index:usize = match str_index.trim().parse() {
                Ok(num) => num,
                Err(_) => {
                    println!("Ошибка: Введите корректное целое число!");
                    return;
                }
            };
            println!("Your number is {}", matrix[[i, index-1]]);
        }
    }
}
