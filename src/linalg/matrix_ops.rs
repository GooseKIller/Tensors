use crate::linalg::{Matrix, Vector};
use crate::Num;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::prelude::*;
use std::ops::{Add, AddAssign, BitAnd, BitAndAssign, Mul, MulAssign, Neg, Sub, SubAssign};

impl<T: Num> Neg for Matrix<T> {
    type Output = Matrix<T>;
    fn neg(self) -> Self::Output {
        let mut data = vec![T::default(); self.rows * self.cols];
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x = -self.data[i];
        });
        Matrix::new(data, self.rows, self.cols)
    }
}

impl<T: Num> Add<&Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "!!!Matrix dimensions do not match!!!\nCannot add Matrix 1: {:?} and Matrix 2: {:?}",
            self.shape(),
            rhs.shape()
        );
        let mut data = vec![T::default(); self.rows * self.cols];
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.data[i] + rhs.data[i];
        });
        Matrix::new(data, self.rows, self.cols)
    }
}

impl<T: Num> Add<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, rhs: &Matrix<T>) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "!!!Matrix dimensions do not match!!!\nCannot add Matrix 1: {:?} and Matrix 2: {:?}",
            self.shape(),
            rhs.shape()
        );
        let mut data = vec![T::default(); self.rows * self.cols];
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.data[i] + rhs.data[i];
        });
        Matrix::new(data, self.rows, self.cols)
    }
}

impl<T: Num> AddAssign<&Matrix<T>> for Matrix<T> {
    fn add_assign(&mut self, rhs: &Matrix<T>) {
        assert!(self.rows == rhs.rows && self.cols == rhs.cols,
                "!!!Matrix dimensions do not match!!!\nCannot add assign Matrix 1: {:?} and Matrix 2: {:?}",
                self.shape(), rhs.shape());

        self.data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x += rhs.data[i];
        });
    }
}

impl<T: Num> Add<T> for Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, rhs: T) -> Self::Output {
        let mut data = vec![T::default(); self.rows * self.cols];
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.data[i] + rhs;
        });
        Matrix::new(data, self.rows, self.cols)
    }
}

macro_rules! impl_add_for_types {
    ($($type:ty),*) => {
        $(
            impl Add<Matrix<$type>> for $type {
                type Output = Matrix<$type>;

                fn add(self, rhs: Matrix<$type>) -> Matrix<$type> {
                    rhs + self
                }
            }

            impl Add<&Matrix<$type>> for $type {
                type Output = Matrix<$type>;

                fn add(self, rhs: &Matrix<$type>) -> Matrix<$type> {
                    rhs.map(| x | x + self)
                }
            }
        )*
    };
}
impl_add_for_types!(i16, i32, i64, i128, f32, f64);

impl<T: Num> AddAssign<T> for Matrix<T> {
    fn add_assign(&mut self, rhs: T) {
        self.data.par_iter_mut().for_each(|x| {
            *x += rhs;
        });
    }
}

impl<T: Num> Sub<&Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, rhs: &Matrix<T>) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "!!!Matrix dimensions do not match!!!\nMatrix 1: [{}, {}], Matrix 2: [{} {}]",
            self.rows,
            self.cols,
            rhs.rows,
            rhs.cols
        );
        let mut data = vec![T::default(); self.rows * self.cols];
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.data[i] - rhs.data[i];
        });
        Matrix::new(data, self.rows, self.cols)
    }
}

impl<T: Num> Sub<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, rhs: &Matrix<T>) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "!!!Matrix dimensions do not match!!!\nMatrix 1: [{}, {}], Matrix 2: [{} {}]",
            self.rows,
            self.cols,
            rhs.rows,
            rhs.cols
        );
        let mut data = vec![T::default(); self.rows * self.cols];
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.data[i] - rhs.data[i];
        });
        Matrix::new(data, self.rows, self.cols)
    }
}

impl<T: Num> SubAssign<&Matrix<T>> for Matrix<T> {
    fn sub_assign(&mut self, rhs: &Matrix<T>) {
        if rhs.rows == 1 && rhs.cols == 1 {
            *self -= rhs.data[0];
            return;
        }
        assert!(self.rows == rhs.rows && self.cols == rhs.cols,
                "!!!Matrix dimensions do not match!!!\nCannot sub-assign Matrix 1: {:?} and Matrix 2: {:?}",
                self.shape(), rhs.shape());
        self.data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x -= rhs.data[i];
        });
    }
}

impl<T: Num> Sub<T> for Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, rhs: T) -> Self::Output {
        let mut data = vec![T::default(); self.rows * self.cols];
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.data[i] - rhs;
        });
        Matrix::new(data, self.rows, self.cols)
    }
}
macro_rules! impl_sub_for_types {
    ($($type:ty),*) => {
        $(
            impl Sub<Matrix<$type>> for $type {
                type Output = Matrix<$type>;

                fn sub(self, rhs: Matrix<$type>) -> Matrix<$type> {
                    let mut data = vec![self; rhs.rows * rhs.cols];
                    data.par_iter_mut().enumerate().for_each(|(i, x)| {
                        *x = self - rhs.data[i];
                    });
                    Matrix::new(data, rhs.rows, rhs.cols)
                }
            }

            impl Sub<&Matrix<$type>> for $type {
                type Output = Matrix<$type>;

                fn sub(self, rhs: &Matrix<$type>) -> Matrix<$type> {
                    let mut data = vec![self; rhs.rows * rhs.cols];
                    data.par_iter_mut().enumerate().for_each(|(i, x)| {
                        *x = self - rhs.data[i];
                    });
                    Matrix::new(data, rhs.rows, rhs.cols)
                }
            }
        )*
    };
}
impl_sub_for_types!(i16, i32, i64, i128, f32, f64);

impl<T: Num> SubAssign<T> for Matrix<T> {
    fn sub_assign(&mut self, rhs: T) {
        self.data.par_iter_mut().for_each(|x| *x -= rhs);
    }
}

impl<T: Num> Mul<&Vector<T>> for Matrix<T> {
    type Output = Vector<T>;
    fn mul(self, rhs: &Vector<T>) -> Self::Output {
        assert_eq!(
            self.cols, rhs.length,
            "!!!Matrix amount of columns != Vector length!!!\n\
    Matrix cols: {}, Vector length: {}!!!",
            self.cols, rhs.length
        );
        let mut data = vec![T::default(); self.rows];
        data.par_iter_mut().enumerate().for_each(|(index, x)| {
            for i in 0..self.cols {
                *x += self[[index, i]] * rhs[i];
            }
        });
        Vector::from(data)
    }
}

impl<T: Num> Mul<&Vector<T>> for &Matrix<T> {
    type Output = Vector<T>;
    fn mul(self, rhs: &Vector<T>) -> Self::Output {
        assert_eq!(
            self.cols, rhs.length,
            "!!!Matrix amount of columns != Vector length!!!\n\
    Matrix cols: {}, Vector length: {}!!!",
            self.cols, rhs.length
        );

        let mut data = vec![T::default(); self.rows];
        data.par_iter_mut().enumerate().for_each(|(index, x)| {
            for i in 0..self.cols {
                *x += self[[index, i]] * rhs[i];
            }
        });
        Vector::from(data)
    }
}

impl<T: Num> Mul<&Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        if rhs.rows == 1 && rhs.cols == 1 {
            return self * rhs.data[0];
        }
        assert_eq!(self.cols, rhs.rows,
                   "!!!Matrix amount of columns in the 1st matrix != amount of rows in the 2nd matrix!!!\n\
    Matrix shape: {:?}, Other Matrix shape: {:?}\nCan't multiply", self.size(), rhs.size());

        let mut data = vec![T::default(); self.rows * rhs.cols];
        data.par_iter_mut().enumerate().for_each(|(index, x)| {
            let (i, j) = (index / rhs.cols, index % rhs.cols);
            let mut sum = T::default();
            for k in 0..self.cols {
                sum += self[[i, k]] * rhs[[k, j]];
            }
            *x = sum;
        });
        Self::new(data, self.rows, rhs.cols)
    }
}

impl<T: Num> Mul<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(self.cols, rhs.rows, "!!!Matrix amount of columns in the 1st matrix != amount of rows in the 2nd matrix!!!\n\
    Matrix shape: {:?}, Other Matrix shape: {:?}\nCan't multiply", self.size(), rhs.size());

        let mut data = vec![T::default(); self.rows * rhs.cols];
        data.par_iter_mut().enumerate().for_each(|(index, x)| {
            let (i, j) = (index / rhs.cols, index % rhs.cols);
            let mut sum = T::default();
            for k in 0..self.cols {
                sum += self[[i, k]] * rhs[[k, j]];
            }
            *x = sum;
        });
        Matrix::new(data, self.rows, rhs.cols)
    }
}

impl<T: Num> MulAssign<&Matrix<T>> for Matrix<T> {
    /// WARNING if it is not square matrix sizes will change
    fn mul_assign(&mut self, rhs: &Matrix<T>) {
        assert_eq!(
            self.cols, rhs.rows,
            "!!!Matrix amount of columns in the 1st matrix \
                    does not equal the amount of rows in the 2nd matrix!!!\nCan't multiply"
        );

        let mut result = Self::from_num(T::default(), self.rows, rhs.cols);
        result
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, x)| {
                let (i, j) = (index / rhs.cols, index % rhs.cols);
                let mut sum = T::default();
                for k in 0..self.cols {
                    sum += self[[i, k]] * rhs[[k, j]];
                }
                *x = sum;
            });
        *self = result;
    }
}

/// Hadamard product
impl<T: Num> BitAnd<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;
    fn bitand(self, rhs: &Matrix<T>) -> Self::Output {
        self.hadamard(rhs)
    }
}

impl<T: Num> BitAnd<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn bitand(self, rhs: Matrix<T>) -> Self::Output {
        self.hadamard(&rhs)
    }
}

impl<T: Num> BitAnd<&Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn bitand(self, rhs: &Matrix<T>) -> Self::Output {
        self.hadamard(rhs)
    }
}

impl<T: Num> BitAndAssign<Matrix<T>> for Matrix<T> {
    fn bitand_assign(&mut self, rhs: Matrix<T>) {
        *self = self.hadamard(&rhs);
    }
}

impl<T: Num> BitAndAssign<&Matrix<T>> for Matrix<T> {
    fn bitand_assign(&mut self, rhs: &Matrix<T>) {
        *self = self.hadamard(rhs);
    }
}

impl<T: Num> Mul<T> for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, rhs: T) -> Self::Output {
        let mut data = vec![T::default(); self.rows * self.cols];
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.data[i] * rhs;
        });
        Matrix::new(data, self.rows, self.cols)
    }
}

// it's workaround, but it works
macro_rules! impl_mul_for_types {
    ($($type:ty),*) => {
        $(
            impl Mul<Matrix<$type>> for $type {
                type Output = Matrix<$type>;

                fn mul(self, matrix: Matrix<$type>) -> Matrix<$type> {
                    matrix * self
                }
            }
            impl Mul<&Matrix<$type>> for $type {
                type Output = Matrix<$type>;

                fn mul(self, matrix: &Matrix<$type>) -> Matrix<$type> {
                    matrix.clone() * self
                }
            }
        )*
    };
}
impl_mul_for_types!(i16, i32, i64, i128, f32, f64);
impl<T: Num> MulAssign<T> for Matrix<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.data.par_iter_mut().for_each(|x| {
            *x = *x * rhs;
        });
    }
}
