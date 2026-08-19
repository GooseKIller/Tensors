use std::cmp::min;
use rayon::prelude::*;
use crate::{Float, Num, linalg::{Matrix, Tensor, Vector}};
use rayon::prelude::IntoParallelRefMutIterator;

impl<T: Num> Matrix<T> {
    /// Safe version of matrix creation
    /// Create matrix from vector and usize, usize
    /// 
    /// # Example
    /// ```
    /// use tensorrs::linalg::Matrix;
    /// let example = Matrix::try_new(vec![0, -1, -1, 0], 2, 2);
    /// // Will create matrix
    /// // Ok([0 -1]
    /// // [-1 0])
    /// ```
    pub fn try_new(data: Vec<T>, rows: usize, cols: usize) -> Result<Self, String> {
        if data.len() != rows * cols {
            return Err("!!!Inconsistent data and dimensions combination for matrix!!!".to_string());
        }
        Ok(Self { data, rows, cols })
    }

    /// Creating Matrix with diagonal values of vector
    /// 
    /// # Example
    /// ```
    /// use tensorrs::linalg::{Matrix, Vector};
    /// use tensorrs::matrix;
    /// 
    /// let a = Vector::from(vec![1,2,3]);
    /// 
    /// let b = Matrix::try_from_diag(a, 3, 3).expect("Failed to create diagonal matrix");
    /// 
    /// assert_eq!(b, matrix![[1, 0, 0],
    ///                       [0, 2, 0],
    ///                       [0, 0, 3]])
    /// ```
    pub fn try_from_diag(data:Vector<T>, rows:usize, cols:usize) -> Result<Matrix<T>, String> {
        if data.length != min(rows, cols) {
        return Err(format!(
            "!!!The length of the data vector ({}) must be equal to the minimum of rows ({}) and cols ({})!!!",
            data.length, rows, cols
        ));
    }
        let mut mx_data = vec![T::default(); cols * rows];
        for i in 0..min(cols, rows) {
            let index = (cols * i) + i;
            mx_data[index] = data[i];
        }
        Ok(Matrix{
            data:mx_data,
            rows,
            cols
        })
    }

    /// Returns column as Vector with index (index starts with 0)
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::{Matrix, Vector};
    /// use tensorrs::matrix;
    /// 
    /// let example = matrix![[1,2],
    ///                       [3,4]];
    /// let col: Vector<i32> = example.try_get_col(0).expect("Failed to get column"); // [1 3]
    /// ```
    ///
    /// # Example with error handling
    /// ```
    /// use tensorrs::linalg::Matrix;
    /// use tensorrs::matrix;
    /// 
    /// let example = matrix![[1,2],
    ///                       [3,4]];
    /// match example.try_get_col(5) {
    ///     Ok(col) => println!("Column: {:?}", col),
    ///     Err(e) => println!("Error: {}", e), // Will print: "Column index 5 out of bounds for matrix with 2 columns"
    /// }
    /// ```
    pub fn try_get_col(&self, index: usize) -> Result<Vector<T>, String> {
        if index >= self.cols {
            return Err(format!(
                "!!!Index:{} is greater than or equal to columns count:{}!!!",
                index,
                self.cols));
        }
        let mut vector = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let index_col = i * self.cols + index;
            vector.push(self.data[index_col]);
        }
       Ok(Vector::from(vector))
    }

    /// Removes column from matrix
    ///
    /// # Example
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// 
    /// let example = matrix![[1., 2.], [3., 4.]];
    /// let rem_example = example.try_rem_col(1).expect("Failed to remove column");
    /// assert_eq!(rem_example, matrix![[1.], [3.]]);
    /// ```
    ///
    /// # Example with error handling
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// 
    /// let example = matrix![[1., 2.], [3., 4.]];
    /// match example.try_rem_col(2) {
    ///     Ok(matrix) => println!("New matrix: {:?}", matrix),
    ///     Err(e) => println!("Error: {}", e), // Will print: "Column index 2 out of bounds for matrix with 2 columns"
    /// }
    /// ```
    pub fn try_rem_col(&self, index: usize) -> Result<Matrix<T>, String> {
        if index >= self.cols {
            return Err(format!(
                "!!!Column index {} out of bounds for matrix with {} columns!!!",
                index, self.cols
            ));
        }
        
        if self.cols == 1 {
            return Err(format!(
                "!!!Cannot remove column from matrix with only 1 column (index: {})!!!",
                index
            ));
        }

        let mut new_data = self.data.clone();
        for i in (0..self.rows).rev() {
            let remove_index = i * self.cols + index;
            new_data.remove(remove_index);
        }

        Ok(Matrix {
            data: new_data,
            rows: self.rows,
            cols: self.cols - 1,
        })
    }

    pub fn try_set_col(&mut self, col: usize, v: &Vector<T>) -> Result<(), String> {
        if col >= self.cols {
            return Err(format!(
                "Column index {} out of bounds (cols = {})",
                col, self.cols
            ));
        }

        if v.len() != self.rows {
            return Err(format!(
                "Vector length {} does not match matrix rows {}",
                v.len(),
                self.rows
            ));
        }

        for i in 0..self.rows {
            self[[i, col]] = v[i];
        }

        Ok(())
    }

    /// Returns row as Vector with index (index starts with 0)
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::{Matrix, Vector};
    /// use tensorrs::matrix;
    /// 
    /// let example = matrix![[1,2],
    ///                       [3,4]];
    /// let row: Vector<i32> = example.try_get_row(1).expect("Failed to get row"); // [3 4]
    /// ```
    ///
    /// # Example with error handling
    /// ```
    /// use tensorrs::linalg::Matrix;
    /// use tensorrs::matrix;
    /// 
    /// let example = matrix![[1,2],
    ///                       [3,4]];
    /// match example.try_get_row(2) {
    ///     Ok(row) => println!("Row: {:?}", row),
    ///     Err(e) => println!("Error: {}", e), // Will print: "Row index 2 out of bounds for matrix with 2 rows"
    /// }
    /// ```
    pub fn try_get_row(&self, index: usize) -> Result<Vector<T>, String> {
        if index >= self.rows {
            return Err(format!(
                "!!!Row index {} out of bounds for matrix with {} rows!!!",
                index, self.rows
            ));
        }
        
        let start_index = index * self.cols;
        let end_index = start_index + self.cols;

        Ok(Vector::from(self.data[start_index..end_index].to_vec()))
    }

    /// Adds column at the end of Matrix
    ///
    /// # Example
    ///
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// let mut a = matrix![[1, 1],
    ///                     [1, 1]];
    ///
    /// a.try_add_column(vec![2,2]).expect("Failed to add column");
    /// // [[1,1,2]
    /// // [1,1,2]]
    /// ```
    ///
    /// # Example with error handling
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// 
    /// let mut a = matrix![[1, 1],
    ///                     [1, 1]];
    /// 
    /// match a.try_add_column(vec![2]) {
    ///     Ok(_) => println!("Column added successfully"),
    ///     Err(e) => println!("Error: {}", e), // Will print: "!!!The length of the Vec<T> (1) is not equal to the size of the rows of the matrix (2) !!!"
    /// }
    /// ```
    pub fn try_add_column(&mut self, column: Vec<T>) -> Result<(), String> {
        if column.len() != self.rows {
            return Err(format!(
                "!!!The length of the Vec<T> ({}) is not equal to the size of the rows of the matrix ({}) !!!",
                column.len(),
                self.rows
            ));
        }
        
        let mut new_data = Vec::with_capacity(self.data.len() + self.rows);
        
        for row in 0..self.rows {
            let row_start = row * self.cols;
            let row_end = row_start + self.cols;
            
            new_data.extend_from_slice(&self.data[row_start..row_end]);
            new_data.push(column[row].clone());
        }
        
        self.data = new_data;
        self.cols += 1;
        
        Ok(())
    }

    /// Adds row at the end of Matrix
    ///
    /// # Example
    ///
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// let mut a = matrix![[1, 1],
    ///                     [1, 1]];
    ///
    /// a.try_add_row(vec![2,2]).expect("Failed to add row");
    /// // [[1,1]
    /// // [1,1]
    /// // [2,2]]
    /// ```
    ///
    /// # Example with error handling
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// 
    /// let mut a = matrix![[1, 1],
    ///                     [1, 1]];
    /// 
    /// match a.try_add_row(vec![2]) {
    ///     Ok(_) => println!("Row added successfully"),
    ///     Err(e) => println!("Error: {}", e), 
    ///     // Will print: "!!!The length of the Vec<T> (1) is not equal to the size of the columns of the matrix (2)!!!"
    /// }
    /// ```
    pub fn try_add_row(&mut self, row: Vec<T>) -> Result<(), String> {
        if row.len() != self.cols {
            return Err(format!(
                "!!!The length of the Vec<T> ({}) is not equal to the size of the columns of the matrix ({})!!!",
                row.len(),
                self.cols
            ));
        }
        
        for i in row {
            self.data.push(i)
        }
        self.rows += 1;
        
        Ok(())
    }

    /// Safe comparison of two matrices
    ///
    /// Returns `Ok(Matrix<T>)` if shapes match, `Err` otherwise.
    /// 
    /// # Example
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// let a = matrix![[1, 3, -1]];
    /// let b = matrix![[0, 4, -1]];
    /// let c = matrix![[0, 4, -1, 5]];
    ///
    /// // Successful comparison
    /// match a.try_compare(b) {
    ///     Ok(result) => println!("Comparison result: {}", result),
    ///     Err(e) => println!("Error: {}", e),
    /// }
    ///
    /// // Failed comparison (different shapes)
    /// match a.try_compare(c) {
    ///     Ok(result) => println!("Comparison result: {}", result),
    ///     Err(e) => println!("Error: {}", e), // Will print error about shape mismatch
    /// }
    /// ```
    pub fn try_compare(&self, other: Matrix<T>) -> Result<Matrix<T>, String>{
        if self.shape() != other.shape() {
            return Err(format!(
                "!!!Can't compare matrices with different shapes!!!\n Matrix a:{:?}; Matrix b:{:?}",
                self.shape(),
                other.shape()
            ));
        }

        let mut comparisons = vec![T::default(); self.rows * self.cols];
        comparisons.par_iter_mut().enumerate().for_each(|(i, x)| {
            if self.data[i] > other.data[i] {
                *x = T::from(1);
            } else if self.data[i] < other.data[i] {
                *x = T::from(1).neg();
            }
        });
        
        Ok(Matrix::new(comparisons, self.rows, self.cols))
    }

    /// Safe Hadamard product or element-wise product
    ///
    /// Returns `Ok(Matrix<T>)` if shapes match, `Err` otherwise.
    /// 
    /// # Example
    ///```
    /// use tensorrs::linalg::Matrix;
    /// use tensorrs::matrix;
    /// 
    /// let a = matrix![[2, 3, 1], [0, 8, -2]];
    /// let b = matrix![[3, 1, 4], [7, 9, 5]];
    /// let c = matrix![[3, 1, 4]];
    /// 
    /// // Successful Hadamard product
    /// match a.try_hadamard(&b) {
    ///     Ok(result) => println!("Result: {:?}", result),
    ///     Err(e) => println!("Error: {}", e),
    /// }
    /// 
    /// // Failed Hadamard product (different shapes)
    /// match a.try_hadamard(&c) {
    ///     Ok(result) => println!("Result: {:?}", result),
    ///     Err(e) => println!("Error: {}", e),
    ///     // Will print: "!!!Shapes must be equal!!!\nMatrix A: (2 rows, 3 cols) Matrix B: (1 rows, 3 cols)"
    /// }
    /// ```
    pub fn try_hadamard(&self, other: &Matrix<T>) -> Result<Matrix<T>, String>{
        if self.rows != other.rows || self.cols != other.cols {
            return Err(format!(
                "!!!Shapes must be equal!!!\nMatrix A: ({} rows, {} cols) Matrix B: ({} rows, {} cols)",
                self.rows, self.cols, other.rows, other.cols
            ));
        }
        
        let mut ans = vec![T::default(); self.data.len()];
        ans.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.data[i] * other.data[i];
        });
        
        Ok(Matrix::new(ans, self.rows, self.cols))
    }


    /// Safe convolution operation on matrix with given kernel
    ///
    /// Returns `Ok(Matrix<T>)` if kernel size is valid, `Err` otherwise.
    ///
    /// # Example
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// 
    /// let matrix = matrix![[1, 2, 3, 4],
    ///                      [5, 6, 7, 8],
    ///                      [9, 10, 11, 12]];
    /// let valid_kernel = matrix![[1, 0],
    ///                            [0, 1]];
    /// let invalid_kernel = matrix![[1, 0, 0, 0, 0],
    ///                              [0, 1, 0, 0, 0],
    ///                              [0, 0, 1, 0, 0]];
    /// 
    /// // Successful convolution
    /// match matrix.try_conv(&valid_kernel) {
    ///     Ok(result) => println!("Convolution successful: {:?}", result),
    ///     Err(e) => println!("Error: {}", e),
    /// }
    /// 
    /// // Failed convolution (kernel too large)
    /// match matrix.try_conv(&invalid_kernel) {
    ///     Ok(result) => println!("Convolution successful: {:?}", result),
    ///     Err(e) => println!("Error: {}", e),
    ///     // Will print: "!!!Kernel size must be less than or equal to Matrix size!!!\nMatrix: (3 rows, 4 cols), Kernel: (3 rows, 5 cols)"
    /// }
    /// ```
    pub fn try_conv(&self, kernel: &Matrix<T>) -> Result<Matrix<T>, String> {
        if kernel.rows > self.rows || kernel.cols > self.cols {
            return Err(format!(
                "!!!Kernel size must be less than or equal to Matrix size!!!\nMatrix: ({} rows, {} cols), Kernel: ({} rows, {} cols)",
                self.rows, self.cols, kernel.rows, kernel.cols
            ));
        }

        if kernel.rows == 0 || kernel.cols == 0 {
            return Err(format!(
                "!!!Kernel must have non-zero dimensions!!!\nKernel size: ({} rows, {} cols)",
                kernel.rows, kernel.cols
            ));
        }

        let output_rows = self.rows - kernel.rows + 1;
        let output_cols = self.cols - kernel.cols + 1;

        if output_rows == 0 || output_cols == 0 {
            return Err(format!(
                "!!!Resulting convolution would have zero dimensions!!!\nThis happens when kernel is too large for the matrix"
            ));
        }

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
        
        Ok(Matrix::new(result_data, output_rows, output_cols))
    }


    /// Safe set element in matrix
    ///
    /// Returns `Ok(())` if indices are valid, `Err` otherwise.
    ///
    /// # Example
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// 
    /// let mut matrix = matrix![[1, 2, 3],
    ///                          [4, 5, 6]];
    /// 
    /// // Valid set
    /// match matrix.try_set([1, 2], 10) {
    ///     Ok(_) => println!("Value set successfully"),
    ///     Err(e) => println!("Error: {}", e),
    /// }
    /// 
    /// // Invalid set
    /// match matrix.try_set([2, 0], 10) {
    ///     Ok(_) => println!("Value set successfully"),
    ///     Err(e) => println!("Error: {}", e),
    /// }
    /// ```
    pub fn try_set(&mut self, index: [usize; 2], value: T) -> Result<(), String> {
        let [i, j] = index;
        
        if i >= self.rows {
            return Err(format!(
                "!!!Matrix index out of bounds!!! Row index {} out of bounds for matrix with {} rows!!!",
                i, self.rows
            ));
        }
        
        if j >= self.cols {
            return Err(format!(
                "!!!Matrix index out of bounds!!! Column index {} out of bounds for matrix with {} columns!!!",
                j, self.cols
            ));
        }
        
        let idx = (self.cols * i) + j;
        self.data[idx] = value;
        
        Ok(())
    }

    /// Safe get element with bounds checking
    ///
    /// Returns `Some(&T)` if indices are valid, `None` otherwise.
    /// This is a lighter alternative to `try_get` that doesn't provide error messages.
    ///
    /// # Example
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// 
    /// let matrix = matrix![[1, 2, 3],
    ///                      [4, 5, 6]];
    /// 
    /// if let Some(value) = matrix.get([1, 2]) {
    ///     println!("Value: {}", value);
    /// } else {
    ///     println!("Index out of bounds");
    /// }
    /// ```
    pub fn get(&self, index: [usize; 2]) -> Option<&T> {
        let [i, j] = index;
        
        if i < self.rows && j < self.cols {
            Some(&self.data[(self.cols * i) + j])
        } else {
            None
        }
    }

    /// Safe mutable get element with bounds checking
    ///
    /// Returns `Some(&mut T)` if indices are valid, `None` otherwise.
    pub fn get_mut(&mut self, index: [usize; 2]) -> Option<&mut T> {
        let [i, j] = index;
        
        if i < self.rows && j < self.cols {
            Some(&mut self.data[(self.cols * i) + j])
        } else {
            None
        }
    }


    pub fn try_from_vec_vec(value: Vec<Vec<T>>) -> Result<Self, String> {
        let rows = value.len();
        
        if rows == 0 {
            return Ok(Self::new(Vec::new(), 0, 0));
        }
        
        let first_row_len = value[0].len();
        
        for (i, row) in value.iter().enumerate().skip(1) {
            if row.len() != first_row_len {
                return Err(format!(
                    "!!!All rows must have the same length!!! Row 0 has {} elements, row {} has {} elements",
                    first_row_len, i, row.len()
                ));
            }
        }
        
        if first_row_len == 0 && rows > 1 {
            return Err("!!!Invalid matrix dimensions: multiple empty rows!!!".to_string());
        }
        
        if rows > 0 && value[0].is_empty() && rows > 1 {
            return Err("!!!Invalid matrix: first row is empty but there are multiple rows!!!".to_string());
        }
        
        let cols = first_row_len;
        let data = value.into_iter().flatten().collect();
        
        Ok(Self::new(data, rows, cols))
    }

    pub fn try_from_tensor(value: Tensor<T>) -> Result<Self, String> {
        if value.shape.len() != 2 {
            return Err("!!!Shape size must be 2!!!".to_string());
        }
        Ok(Self {
            data: value.packed_data(),
            rows: value.shape[0],
            cols: value.shape[1],
        })
    } 
}

impl<T: Float> Matrix<T> {
    /// Safe version of norm calculation
    ///
    /// Returns `Ok(T)` if p is positive and calculations succeed, `Err` otherwise.
    ///
    /// # Example
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// 
    /// let matrix = matrix![[1.0, 2.0], [3.0, 4.0]];
    /// 
    /// // Successful norm calculation
    /// match matrix.try_norm(2.0) {
    ///     Ok(norm) => println!("Frobenius norm: {}", norm),
    ///     Err(e) => println!("Error: {}", e),
    /// }
    /// 
    /// // Invalid p value
    /// match matrix.try_norm(0.5) {
    ///     Ok(norm) => println!("Norm: {}", norm),
    ///     Err(e) => println!("Error: {}", e),
    ///     // Will print: "!!!Number p must be >= 1!!! Got: 0.5"
    /// }
    /// 
    /// // Negative p value
    /// match matrix.try_norm(-1.0) {
    ///     Ok(norm) => println!("Norm: {}", norm),
    ///     Err(e) => println!("Error: {}", e),
    /// }
    /// ```
    pub fn try_norm(&self, p: T) -> Result<T, String>{
        // Check that p >= 1
        if p < T::one() {
            return Err(format!(
                "!!!Number p must be >= 1!!! Got: {}",
                p
            ));
        }

        // Special handling for p = 1 (the largest column sum)
        if p == T::one() {
            let mut max_num = match self.try_get_col(0) {
                Ok(col) => col.abs_sum(),
                Err(_) => T::default(), // when the matrix is empty
            };

            for i in 1..self.cols {
                if let Ok(col) = self.try_get_col(i) {
                    let sum = col.abs_sum();
                    if sum > max_num {
                        max_num = sum;
                    }
                }
            }
            return Ok(max_num);
        }

        // Special handling for p = 2 (the Frobenius norm)
        if p == T::from(2) {
            let mut sum_of_squares = T::default();
            for x in &self.data {
                sum_of_squares += x.powf(T::from(2));
            }
            return Ok(sum_of_squares.sqrt());
        }

        // The general case for p > 1, p != 2
        let mut norm = T::default();
        for i in &self.data {
            norm += i.powf(p);
        }
        Ok(norm.powf(T::one() / p))
    }

    /// Safe determinant calculation
    pub fn try_det(&self) -> Result<T, String> {
        if self.rows != self.cols {
            return Err(format!(
                "!!!The determinant is defined only for square matrices!!!"
            ));
        }

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
        Ok(det)
    }

    /// Safe eigenvalue computation
    pub fn try_eig(&self) -> Result<Vector<T>, String> {
        if self.rows != self.cols {
            return Err("!!!Matrix must be square!!!".to_string());
        }

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

        Ok(Vector::from(eigenvalues))
    }
    
    /// Safe eigenvector computation
    pub fn try_eig_vector(&self) -> Result<Matrix<T>, String> {
        if self.rows != self.cols {
            return Err("!!!Matrix must be square!!!".to_string());
        }

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

        Ok(v)
    }
}

/// Safe Matrix multiplication
pub fn try_mtrxdot<T: Num>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>, String> {
    if a.cols != b.rows {
        return Err(format!("Matrix multiplication shape mismatch: left = ({},{}) right = ({},{})",
        a.rows, a.cols, b.rows, b.cols));
    }

    let rows = a.rows;
    let cols = b.cols;
    let k = a.cols;

    let small_threshold = 19 * 19;

    let mut rhs_t = vec![T::default(); cols * k];
    for p in 0..k {
        for j in 0..cols {
            rhs_t[j * k + p] = b.data[p * b.cols + j];
        }
    }

    let mut data = vec![T::default(); rows * cols];

    if rows * cols <= small_threshold {
        for i in 0..rows {
            let a_row_base = i * a.cols;
            let out_row_base = i * cols;
            for j in 0..cols {
                let mut sum = T::default();
                let rt_base = j * k;
                for p in 0..k {
                    sum += a.data[a_row_base + p] * rhs_t[rt_base + p];
                }
                data[out_row_base + j] = sum;
            }
        }
    } else {
        data.par_chunks_mut(cols).enumerate().for_each(|(i, out_row)| {
            let a_row_base = i * a.cols;
            for j in 0..cols {
                let mut sum = T::default();
                let rt_base = j * k;
                for p in 0..k {
                    sum += a.data[a_row_base + p] * rhs_t[rt_base + p];
                }
                out_row[j] = sum;
            }
        });
    }

    Ok(Matrix::new(data, rows, cols))
}

/// Safe matrix addition
pub fn try_add<T:Num>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>, String> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(format!(
            "!!!Matrix dimensions do not match!!!\nCannot add Matrix 1: {:?} and Matrix 2: {:?}",
            a.shape(), b.shape()
        ));
    }

    let mut data = vec![T::default(); a.rows * a.cols];
    data.par_iter_mut().enumerate().for_each(|(i, x)| {
        *x = a.data[i] + b.data[i];
    });
    Ok(Matrix::new(data, a.rows, a.cols))
}

/// Safe matrix subtraction
pub fn try_sub<T:Num>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>, String> {
    if !(a.rows == b.rows && a.cols == b.cols) {
        return Err(format!(
            "!!!Matrix dimensions do not match!!!\nMatrix 1: [{}, {}], Matrix 2: [{} {}]",
            a.rows,
            a.cols,
            b.rows,
            b.cols
        ));
    }
    let mut data = vec![T::default(); a.rows * a.cols];
    data.par_iter_mut().enumerate().for_each(|(i, x)| {
        *x = a.data[i] - b.data[i];
    });
    Ok(Matrix::new(data, a.rows, a.cols))
}

/// Safe Matrix and vector multiplication
pub fn try_mul_vec<T:Num>(a: &Matrix<T>, b: &Vector<T>) -> Result<Vector<T>, String> {
    if a.cols != b.length {
        return Err(format!(
            "!!!Matrix amount of columns != Vector length!!!\n\
    Matrix cols: {}, Vector length: {}!!!",
            a.cols, b.length
        ));
    }
    let mut data = vec![T::default(); a.rows];
    data.par_iter_mut().enumerate().for_each(|(index, x)| {
        for i in 0..a.cols {
            *x += a[[index, i]] * b[i];
        }
    });
    Ok(Vector::from(data))
}

#[cfg(test)]
mod test {
    use crate::{linalg::Matrix, matrix};

    #[test]
    fn create_matrix() {
        let _a = Matrix::try_new(vec![0, -1, -1, 0], 2, 2);
    }

    #[test]
    fn other_stuff() {
        let a = matrix![
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 2.0]];

        println!("{}", a.eig());
        println!("{}", a.eig_vectors());

        let b = matrix![[0.0, -1.0], [1.0, 0.0]];

        println!("{}", b.eig_vectors());
    }
}