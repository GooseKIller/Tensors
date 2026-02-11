use std::ops::{Add, Mul, Sub};
use rayon::prelude::*;
use crate::{Float, linalg::Matrix};

#[derive(Debug, Clone, Copy)]
struct Complex<T: Float> {
    pub re: T,
    pub img: T,
}

impl<T:Float> Complex<T> {
    fn new(re: T, img: T) -> Self {
        Self { re, img }
    }
    
    fn expi(theta: T) -> Self {
        Self { re: theta.cos(), img: theta.sin() }
    }
}

impl<T: Float> Add<Complex<T>> for Complex<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { re: self.re + rhs.re, img: self.img + rhs.img }
    }
}

impl<T: Float> Sub<Complex<T>> for Complex<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { re: self.re - rhs.re, img: self.img - rhs.img }
    }
}

impl<T:Float> Mul<Complex<T>> for Complex<T> {
    type Output = Self;

    fn mul(self, rhs: Complex<T>) -> Self::Output {
        Self {
            re: self.re * rhs.re - self.img * rhs.img,
            img: self.re * rhs.img + self.img * rhs.re,
        }
    }   
}

impl<T: Float> Mul<T> for Complex<T> {
    type Output = Complex<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Self { re: self.re * rhs, img: self.img * rhs }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PaddingMode {
    Valid,
    Zero(usize, usize),
    Mirror(usize, usize),
}

impl<T: Float> Matrix<T> {
    pub fn conv_fft(&self, kernel: &Matrix<T>) -> Matrix<T> {
        assert!(
            kernel.rows <= self.rows && kernel.cols <= self.cols,
            "Kernel size must be less than or equal to input size"
        );
        let output_rows = self.rows - kernel.rows + 1;
        let output_cols = self.cols - kernel.cols + 1;
        fft_convolution_2d(self, kernel, output_rows, output_cols, PaddingMode::Valid)
    }

    pub fn conv_zero_fft(&self, kernel: &Matrix<T>) -> Matrix<T> {
        let pad_rows = kernel.rows / 2;
        let pad_cols = kernel.cols / 2;
        fft_convolution_2d(self, kernel, self.rows, self.cols, PaddingMode::Zero(pad_rows, pad_cols))
    }

    pub fn conv_with_mirror_padding_fft(&self, kernel: &Matrix<T>) -> Matrix<T> {
        let pad_rows = kernel.rows / 2;
        let pad_cols = kernel.cols / 2;
        fft_convolution_2d(self, kernel, self.rows, self.cols, PaddingMode::Mirror(pad_rows, pad_cols))
    }

    pub fn smart_conv(&self, kernel: &Matrix<T>, mode: PaddingMode) -> Matrix<T> {
        const MS: [usize;5] = [64, 128, 256, 512, 1024];
        const KS: [usize;4] = [3, 16, 64, 128];

        let fft_table: [[Option<f64>;5];4] = [
            // K=3
            [Some(14.0), Some(27.0), Some(86.0), Some(383.0), Some(1611.0)],
            // K=16
            [Some(6.0), Some(22.0), Some(86.0), Some(367.0), Some(1610.0)],
            // K=64
            [Some(6.0), Some(22.0), Some(86.0), Some(362.0), Some(1613.0)],
            // K=128
            [None, Some(23.0), Some(85.0), Some(361.0), Some(1607.0)],
        ];

        // Direct times
        let direct_table: [[Option<f64>;5];4] = [
            // K=3
            [Some(0.0), Some(1.0), Some(3.0), Some(12.0), None],
            // K=16
            [Some(2.0), Some(13.0), Some(61.0), Some(258.0), None],
            // K=64
            [Some(0.0), Some(69.0), Some(604.0), Some(3322.0), None],
            // K=128
            [None, Some(0.0), Some(1102.0), Some(10026.0), None],
        ];

        let n = self.rows.max(self.cols);
        let k = kernel.rows.max(kernel.cols);

        let log_n = (n as f64).log2();
        let log_k = (k as f64).log2();

        let mut best_dist = f64::INFINITY;
        let mut best_fft: Option<f64> = None;
        let mut best_direct: Option<f64> = None;

        for (i, &ks) in KS.iter().enumerate() {
            for (j, &ms) in MS.iter().enumerate() {
                if fft_table[i][j].is_none() && direct_table[i][j].is_none() {
                    continue;
                }
                let d = (log_k - (ks as f64).log2()).powi(2) + (log_n - (ms as f64).log2()).powi(2);
                if d < best_dist {
                    best_dist = d;
                    best_fft = fft_table[i][j];
                    best_direct = direct_table[i][j];
                }
            }
        }

        let choose_fft = if let Some(fft_t) = best_fft {
            match best_direct {
                Some(direct_t) => {
                    fft_t > direct_t
                }
                None => {
                    true
                }
            }
        } else {
            let kernel_size = kernel.rows * kernel.cols;
            let matrix_size = self.rows * self.cols;
            kernel_size > 32 || matrix_size > 512*512
        };

        if choose_fft {
            match mode {
                PaddingMode::Valid => self.conv_fft(kernel),
                PaddingMode::Zero(_, _) => self.conv_zero_fft(kernel),
                PaddingMode::Mirror(_, _) => self.conv_with_mirror_padding_fft(kernel),
            }
        } else {
            match mode {
                PaddingMode::Valid => self.conv(kernel),
                PaddingMode::Zero(_, _) => self.conv_zero(kernel),
                PaddingMode::Mirror(_, _) => self.conv_with_mirror_padding(kernel)
            }
        }

    }
}

fn fft_convolution_2d<T: Float + Copy>(
    input: &Matrix<T>,
    kernel: &Matrix<T>,
    output_rows: usize,
    output_cols: usize,
    padding: PaddingMode,
) -> Matrix<T> {
    let fft_rows = (input.rows + kernel.rows - 1).next_power_of_two();
    let fft_cols = (input.cols + kernel.cols - 1).next_power_of_two();

    let mut input_buf = prepare_input(input, fft_rows, fft_cols, &padding);
    let mut kernel_buf = prepare_kernel(kernel, fft_rows, fft_cols);

    // forward 2D FFT (parallelized over rows and columns)
    fft_2d(&mut input_buf, fft_rows, fft_cols, false);
    fft_2d(&mut kernel_buf, fft_rows, fft_cols, false);

    // pointwise multiply (parallel)
    input_buf.par_iter_mut()
        .zip(kernel_buf.par_iter())
        .for_each(|(a, b)| *a = *a * *b);

    // inverse 2D FFT (parallelized)
    fft_2d(&mut input_buf, fft_rows, fft_cols, true);

    extract_result(&input_buf, fft_cols, output_rows, output_cols, &padding, fft_rows, fft_cols)
}

// Подготовка входа: T -> Complex<T>
fn prepare_input<T: Float + Copy>(
    input: &Matrix<T>,
    fft_rows: usize,
    fft_cols: usize,
    padding: &PaddingMode,
) -> Vec<Complex<T>> {
    let zero = T::default();
    let mut buffer = vec![Complex::new(zero, zero); fft_rows * fft_cols];

    match padding {
        PaddingMode::Valid => {
            for i in 0..input.rows {
                for j in 0..input.cols {
                    let val = input.data[i * input.cols + j];
                    buffer[i * fft_cols + j] = Complex::new(val, T::default());
                }
            }
        }
        PaddingMode::Zero(pad_rows, pad_cols) => {
            for i in 0..input.rows {
                for j in 0..input.cols {
                    let row = i + *pad_rows;
                    let col = j + *pad_cols;
                    if row < fft_rows && col < fft_cols {
                        let val = input.data[i * input.cols + j];
                        buffer[row * fft_cols + col] = Complex::new(val, T::default());
                    }
                }
            }
        }
        PaddingMode::Mirror(pad_rows, pad_cols) => {
            for r in 0..fft_rows {
                for c in 0..fft_cols {
                    let src_r = reflect_index((r as isize) - (*pad_rows as isize), input.rows);
                    let src_c = reflect_index((c as isize) - (*pad_cols as isize), input.cols);
                    let val = input.data[src_r * input.cols + src_c];
                    buffer[r * fft_cols + c] = Complex::new(val, T::default());
                }
            }
        }
    }
    buffer
}

fn reflect_index(mut k: isize, n: usize) -> usize {
    let n = n as isize;
    if n == 0 { return 0 }
    loop {
        if k < 0 {
            k = -k - 1;
        } else if k >= n {
            k = 2*n - k - 1;
        } else {
            break;
        }
    }
    k as usize
}

fn prepare_kernel<T: Float + Copy>(
    kernel: &Matrix<T>,
    fft_rows: usize,
    fft_cols: usize,
) -> Vec<Complex<T>> {
    let zero = T::default();
    let mut buffer = vec![Complex::new(zero, zero); fft_rows * fft_cols];

    let pad_rows = kernel.rows / 2;
    let pad_cols = kernel.cols / 2;

    for i in 0..kernel.rows {
        for j in 0..kernel.cols {
            let mirrored_i = kernel.rows - 1 - i;
            let mirrored_j = kernel.cols - 1 - j;
            let val = kernel.data[mirrored_i * kernel.cols + mirrored_j];
            let row = (fft_rows + i - pad_rows) % fft_rows;
            let col = (fft_cols + j - pad_cols) % fft_cols;
            buffer[row * fft_cols + col] = Complex::new(val, T::default());
        }
    }
    buffer
}

fn fft_2d<T: Float + Copy>(
    buffer: &mut [Complex<T>],
    rows: usize,
    cols: usize,
    inverse: bool,
) {
    buffer.par_chunks_mut(cols).for_each(|slice| {
        fft_1d_inplace(slice, inverse);
    });

    // 2) FFT по столбцам — параллельно вычисляем каждый столбец и собираем результаты
    // cols_results[c] будет Vec<Complex<T>> длины rows — результат для столбца c
    let cols_results: Vec<Vec<Complex<T>>> = (0..cols).into_par_iter().map(|c| {
        // собрать столбец в локальный буфер
        let mut col_buffer = vec![Complex::new(T::default(), T::default()); rows];
        for r in 0..rows {
            col_buffer[r] = buffer[r * cols + c];
        }
        // выполнить FFT для столбца (локально)
        fft_1d_inplace(&mut col_buffer, inverse);
        col_buffer
    }).collect();

    // 3) Копируем результаты обратно в buffer (последовательно — безопасно и быстро по сравнению с FFT)
    for c in 0..cols {
        for r in 0..rows {
            buffer[r * cols + c] = cols_results[c][r];
        }
    }
}

fn fft_1d_inplace<T: Float + Copy>(buf: &mut [Complex<T>], inverse: bool) {
    let n = buf.len();
    assert!(n.is_power_of_two(), "FFT length must be power of two, got {}", n);
    // bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }

    // main loops
    let mut len = 2;
    let pi = T::pi();
    while len <= n {
        let len_t = T::from_usize(len);
        let two = T::from_usize(2);
        let angle = if inverse {
            two * pi / len_t
        } else {
            -two * pi / len_t
        };
        let wlen = Complex::expi(angle);
        for i in (0..n).step_by(len) {
            let mut w = Complex::new(T::one(), T::default());
            let half = len / 2;
            for j in 0..half {
                let u = buf[i + j];
                let v = buf[i + j + half] * w;
                buf[i + j] = u + v;
                buf[i + j + half] = u - v;
                w = w * wlen;
            }
        }
        len <<= 1;
    }

    // NOTE: scaling (1/N) is done outside (in extract_result),
    // so we do not scale here.
}

fn extract_result<T: Float + Copy>(
    buffer: &[Complex<T>],
    stride: usize,
    output_rows: usize,
    output_cols: usize,
    padding: &PaddingMode,
    fft_rows: usize,
    fft_cols: usize,
) -> Matrix<T> {
    let denom = fft_rows * fft_cols;
    let scale = T::one() / T::from_usize(denom);
    let mut result_data = vec![T::default(); output_rows * output_cols];

    let (start_row, start_col) = match padding {
        PaddingMode::Valid => (0, 0),
        PaddingMode::Zero(pad_rows, pad_cols) => (*pad_rows, *pad_cols),
        PaddingMode::Mirror(pad_rows, pad_cols) => (*pad_rows, *pad_cols),
    };

    for i in 0..output_rows {
        for j in 0..output_cols {
            let buf_row = start_row + i;
            let buf_col = start_col + j;
            if buf_row < buffer.len() / stride && buf_col < stride {
                let idx = buf_row * stride + buf_col;
                let value = buffer[idx].re * scale;
                result_data[i * output_cols + j] = value;
            }
        }
    }

    Matrix::new(result_data, output_rows, output_cols)
}


#[cfg(test)]
mod tests {
    //use std::time::Instant;

    use crate::linalg::Matrix;
    /*
    #[test]
    fn test_fft_conv() {
        let a: Matrix<f64> = Matrix::from_num(1.0, 6, 6);
        let sobel = matrix![[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]];
        let fft_res: Matrix<f64> = a.conv_zero_fft(&sobel);
        let direct_res: Matrix<f64> = a.conv_zero(&sobel);
        println!("FFT: {}", fft_res);
        println!("Direct: {}", direct_res);

        for i in 0..fft_res.data.len() {
            let diff = (fft_res.data[i] - direct_res.data[i]).abs();
            assert!(diff < 1e-5, "Mismatch at {}: {} vs {}", i, fft_res.data[i], direct_res.data[i]);
        }
    }

    #[test]
    fn conv_time() {
        let matrix_size = 2*512usize;
        let kernel_size = 128usize;
        let a: Matrix<f32> = Matrix::randn(matrix_size, matrix_size);
        let b: Matrix<f32> = Matrix::randn(kernel_size, kernel_size);
        //parallel
        let start_time = Instant::now();
        let _ans = a.conv_fft(&b);
        let elapsed_time = start_time.elapsed();
        println!("With FFT Time: {} millis", elapsed_time.as_millis());

        let start_time = Instant::now();
        let _z = a.conv(&b);
        let elapsed_time = start_time.elapsed();
        println!("Without Time: {} millis", elapsed_time.as_millis());
    }

    #[test]
    fn generate_conv_decision_table() {
        println!("{:^10} | {:^12} | {:^10} | {:^10} | {:^8}", 
                "Kernel", "Matrix", "FFT(ms)", "Direct(ms)", "Faster");
        println!("{:-<60}", "");

        // Test different size combinations
        let sizes = [
            // Small kernels
            (3, 64),
            (3, 128),
            (3, 256),
            (3, 512),
            (3, 1024),
            
            // Medium kernels
            (16, 64),
            (16, 128),
            (16, 256),
            (16, 512),
            (16, 1024),
            
            // Large kernels
            (64, 64),
            (64, 128),
            (64, 256),
            (64, 512),
            (64, 1024),
            
            (128, 128),
            (128, 256),
            (128, 512),
            (128, 1024),
        ];

        for (kernel_size, matrix_size) in sizes.iter() {
            compare_methods(*kernel_size, *matrix_size);
        }
    }

    fn compare_methods(kernel_size: usize, matrix_size: usize) {
        let a: Matrix<f32> = Matrix::randn(matrix_size, matrix_size);
        let b: Matrix<f32> = Matrix::randn(kernel_size, kernel_size);
        
        // Time FFT convolution
        let fft_start = Instant::now();
        let _fft_result = a.conv_fft(&b);
        let fft_time = fft_start.elapsed().as_millis();
        
        // Time direct convolution (only if matrix is not too large)
        let direct_time = if matrix_size <= 512 { // Avoid extremely long computations
            let direct_start = Instant::now();
            let _direct_result = a.conv(&b);
            direct_start.elapsed().as_millis()
        } else {
            u128::MAX // Mark as too large for practical testing
        };
        
        // Determine which is faster
        let faster = if direct_time == u128::MAX {
            "FFT-only"
        } else if fft_time < direct_time {
            "FFT"
        } else {
            "Direct"
        };
        
        println!("{:^10} | {:^12} | {:^10} | {:^10} | {:^8}", 
                kernel_size, 
                format!("{}x{}", matrix_size, matrix_size),
                fft_time,
                if direct_time == u128::MAX { "N/A".to_string() } else { direct_time.to_string() },
                faster);
    }*/
    #[test]
    fn generate_conv_decision_table() {
        println!("{:^10} | {:^12} | {:^10} | {:^10} | {:^10} | {:^10}", 
                "Kernel", "Matrix", "FFT(ms)", "Direct(ms)", "Smart(ms)", "Chosen");
        println!("{:-<75}", "");

        // Test different size combinations
        let sizes = [
            // Small kernels
            (3, 64),
            (3, 128),
            (3, 256),
            (3, 512),
            (3, 1024),
            
            // Medium kernels
            (16, 64),
            (16, 128),
            (16, 256),
            (16, 512),
            (16, 1024),
            
            // Large kernels
            (64, 64),
            (64, 128),
            (64, 256),
            (64, 512),
            (64, 1024),
            
            (128, 128),
            (128, 256),
            (128, 512),
            (128, 1024),
        ];

        for (kernel_size, matrix_size) in sizes.iter() {
            compare_methods_with_smart(*kernel_size, *matrix_size);
        }
    }

    fn compare_methods_with_smart(kernel_size: usize, matrix_size: usize) {
        let a: Matrix<f32> = Matrix::randn(matrix_size, matrix_size);
        let b: Matrix<f32> = Matrix::randn(kernel_size, kernel_size);
        
        // Time FFT convolution
        let fft_start = std::time::Instant::now();
        let _fft_result = a.conv_fft(&b);
        let fft_time = fft_start.elapsed().as_millis();
        
        // Time direct convolution (only if matrix is not too large)
        let direct_time = if matrix_size <= 512 {
            let direct_start = std::time::Instant::now();
            let _direct_result = a.conv(&b);
            direct_start.elapsed().as_millis()
        } else {
            u128::MAX
        };

        // Time smart convolution
        let smart_start = std::time::Instant::now();
        let _smart_result = a.smart_conv(&b, crate::linalg::matrix::conv::PaddingMode::Zero(kernel_size/2, kernel_size/2));
        let smart_time = smart_start.elapsed().as_millis();

        // Determine which method smart chose
        let chosen = if smart_time == fft_time {
            "FFT"
        } else if smart_time == direct_time {
            "Direct"
        } else {
            "Smart" // если smart сам что-то смешанное делает (на всякий случай)
        };

        println!("{:^10} | {:^12} | {:^10} | {:^10} | {:^10} | {:^10}", 
                kernel_size, 
                format!("{}x{}", matrix_size, matrix_size),
                fft_time,
                if direct_time == u128::MAX { "N/A".to_string() } else { direct_time.to_string() },
                smart_time,
                chosen
        );
    }

}