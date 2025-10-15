use crate::linalg::{Matrix, Vector};
use crate::Num;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::prelude::*;
use std::ops::{Add, AddAssign, BitAnd, BitAndAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};


pub fn mtrxdot<T: Num>(a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T>
{
    assert_eq!(
        a.cols, b.rows,
        "Matrix multiplication shape mismatch: left = ({},{}) right = ({},{})",
        a.rows, a.cols, b.rows, b.cols
    );

    let rows = a.rows;
    let cols = b.cols;
    let k = a.cols; // == b.rows

    // Порог — если результат меньше или равен, выполняем в одном потоке.
    // Настройте под свои данные/машину. Уменьшите порог -> чаще однопоточно.
    let small_threshold = 19 * 19;

    // Транспонируем b в rhs_t: rhs_t[j * k + p] == b.data[p * b.cols + j] == b[[p,j]]
    let mut rhs_t = vec![T::default(); cols * k];
    for p in 0..k {
        for j in 0..cols {
            rhs_t[j * k + p] = b.data[p * b.cols + j];
        }
    }

    let mut data = vec![T::default(); rows * cols];

    if rows * cols <= small_threshold {
        // Однопоточная версия для маленьких матриц
        for i in 0..rows {
            let a_row_base = i * a.cols;
            let out_row_base = i * cols;
            for j in 0..cols {
                let mut sum = T::default();
                let rt_base = j * k;
                // суммируем по p
                for p in 0..k {
                    // a[i,p] == a.data[a_row_base + p]
                    sum += a.data[a_row_base + p] * rhs_t[rt_base + p];
                }
                data[out_row_base + j] = sum;
            }
        }
    } else {
        // Параллельная версия: параллелим по строкам результата
        // par_chunks_mut разделяет `data` на срезы по `cols` элементов — каждая строка.
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

    Matrix::new(data, rows, cols)
}


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
        mtrxdot(&self, rhs)
        /*
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
        */
    }
}

impl<T: Num> Mul<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(self.cols, rhs.rows, "!!!Matrix amount of columns in the 1st matrix != amount of rows in the 2nd matrix!!!\n\
    Matrix shape: {:?}, Other Matrix shape: {:?}\nCan't multiply", self.size(), rhs.size());
        mtrxdot(self, rhs)
        /*
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
        */
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
        /*

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
        */
        *self = mtrxdot(&self, rhs)//result;
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

impl<T: Num> Div<T> for Matrix<T> {
    type Output = Matrix<T>;

    fn div(self, rhs: T) -> Self::Output {
        let mut data = vec![T::default(); self.rows * self.cols];
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.data[i] / rhs;
        });
        Matrix::new(data, self.rows, self.cols)
    }
}

impl<T: Num> Div<T> for &Matrix<T> {
    type Output = Matrix<T>;
    fn div(self, rhs: T) -> Self::Output {
        let mut data = vec![T::default(); self.rows * self.cols];
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.data[i] / rhs;
        });
        Matrix::new(data, self.rows, self.cols)
    }
}

macro_rules! impl_div_for_types {
    ($($type:ty),*) => {
        $(
            impl Div<Matrix<$type>> for $type {
                type Output = Matrix<$type>;

                fn div(self, rhs: Matrix<$type>) -> Matrix<$type> {
                    let mut data = vec![self; rhs.rows * rhs.cols];
                    data.par_iter_mut().enumerate().for_each(|(i, x)| {
                        *x = self / rhs.data[i];
                    });
                    Matrix::new(data, rhs.rows, rhs.cols)
                }
            }

            impl Div<&Matrix<$type>> for $type {
                type Output = Matrix<$type>;

                fn div(self, rhs: &Matrix<$type>) -> Matrix<$type> {
                    let mut data = vec![self; rhs.rows * rhs.cols];
                    data.par_iter_mut().enumerate().for_each(|(i, x)| {
                        *x = self / rhs.data[i];
                    });
                    Matrix::new(data, rhs.rows, rhs.cols)
                }
            }
        )*
    };
}
impl_div_for_types!(i16, i32, i64, i128, f32, f64);

impl<T:Num> DivAssign<T> for Matrix<T> {
    fn div_assign(&mut self, rhs: T) {
        self.data.par_iter_mut().for_each(|x| {*x = *x / rhs});
    }
}


#[cfg(test)]
mod tests {
    use crate::linalg::Matrix;


    #[test]
    fn mx_multiplication_test() {
        use std::time::Instant;
        use rayon::prelude::*;

        // helper: прогрев + среднее время из `runs` прогонов
        fn measure_avg<F: Fn() -> Matrix<f32>>(f: F, runs: usize) -> std::time::Duration {
            // прогрев
            let _ = f();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..runs {
                let t0 = Instant::now();
                let _res = f();
                total += t0.elapsed();
            }
            total / (runs as u32)
        }

        // reference: single-thread (transpose b, обычные циклы)
        fn mtrxdot_ref_st(a: &Matrix<f32>, b: &Matrix<f32>) -> Matrix<f32> {
            assert_eq!(a.cols, b.rows);
            let rows = a.rows;
            let cols = b.cols;
            let k = a.cols;

            let mut rhs_t = vec![0.0f32; cols * k];
            for p in 0..k {
                for j in 0..cols {
                    rhs_t[j * k + p] = b.data[p * b.cols + j];
                }
            }

            let mut data = vec![0.0f32; rows * cols];
            for i in 0..rows {
                let a_row_base = i * a.cols;
                let out_row_base = i * cols;
                for j in 0..cols {
                    let mut sum = 0.0f32;
                    let rt_base = j * k;
                    for p in 0..k {
                        sum += a.data[a_row_base + p] * rhs_t[rt_base + p];
                    }
                    data[out_row_base + j] = sum;
                }
            }
            Matrix::new(data, rows, cols)
        }

        // reference: always-parallel (par по строкам)
        fn mtrxdot_ref_par(a: &Matrix<f32>, b: &Matrix<f32>) -> Matrix<f32> {
            assert_eq!(a.cols, b.rows);
            let rows = a.rows;
            let cols = b.cols;
            let k = a.cols;

            let mut rhs_t = vec![0.0f32; cols * k];
            for p in 0..k {
                for j in 0..cols {
                    rhs_t[j * k + p] = b.data[p * b.cols + j];
                }
            }

            let mut data = vec![0.0f32; rows * cols];
            data.par_chunks_mut(cols).enumerate().for_each(|(i, out_row)| {
                let a_row_base = i * a.cols;
                for j in 0..cols {
                    let mut sum = 0.0f32;
                    let rt_base = j * k;
                    for p in 0..k {
                        sum += a.data[a_row_base + p] * rhs_t[rt_base + p];
                    }
                    out_row[j] = sum;
                }
            });

            Matrix::new(data, rows, cols)
        }

        // main loop: n = 5 .. 127
        let runs = 3usize;
        println!("n, lib_us, st_us, par_us, rayon_threads");
        for n in 5usize..128 {
            let a = Matrix::from_num((n as f32).powf(2.5), n, n);
            let b = Matrix::from_num(1.0f32, n, n);

            // measure library function (если доступна)
            let avg_lib = std::panic::catch_unwind(|| {
                measure_avg(|| crate::linalg::matrix_ops::mtrxdot(&a, &b), runs)
            });

            // measure references
            let avg_st = measure_avg(|| mtrxdot_ref_st(&a, &b), runs);
            let avg_par = measure_avg(|| mtrxdot_ref_par(&a, &b), runs);

            let rayon_threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "auto".into());

            match avg_lib {
                Ok(dur) => println!("{},{},{},{},{}", n, dur.as_micros(), avg_st.as_micros(), avg_par.as_micros(), rayon_threads),
                Err(_) => println!("{},{},{},{},{}", n, "na", avg_st.as_micros(), avg_par.as_micros(), rayon_threads),
            }
        }
    }

}
