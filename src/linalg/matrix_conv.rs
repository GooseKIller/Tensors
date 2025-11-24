use std::sync::Arc;
use rustfft::{FftPlanner, num_complex::Complex};
use crate::{Float, linalg::Matrix};

#[derive(Debug, Clone, Copy)]
enum PaddingMode {
    Valid,
    Zero(usize, usize),
    Mirror(usize, usize),
}

impl<T: Float + Copy> Matrix<T> {
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

    let mut planner = FftPlanner::new();
    let fft_forward_row = planner.plan_fft_forward(fft_cols);
    let fft_forward_col = planner.plan_fft_forward(fft_rows);
    let fft_inverse_row = planner.plan_fft_inverse(fft_cols);
    let fft_inverse_col = planner.plan_fft_inverse(fft_rows);

    let mut input_buf = prepare_input_f64(input, fft_rows, fft_cols, &padding);
    let mut kernel_buf = prepare_kernel_f64(kernel, fft_rows, fft_cols);

    fft_2d_f64(&mut input_buf, fft_rows, fft_cols, &fft_forward_row, &fft_forward_col);
    fft_2d_f64(&mut kernel_buf, fft_rows, fft_cols, &fft_forward_row, &fft_forward_col);

    // Поэлементное умножение
    for i in 0..fft_rows {
        for j in 0..fft_cols {
            let idx = i * fft_cols + j;
            input_buf[idx] = input_buf[idx] * kernel_buf[idx];
        }
    }

    fft_2d_f64(&mut input_buf, fft_rows, fft_cols, &fft_inverse_row, &fft_inverse_col);

    extract_result_f64(&input_buf, fft_cols, output_rows, output_cols, &padding, fft_rows, fft_cols)
}

// Подготовка входа: T → f64
fn prepare_input_f64<T: Float + Copy>(
    input: &Matrix<T>,
    fft_rows: usize,
    fft_cols: usize,
    padding: &PaddingMode,
) -> Vec<Complex<f64>> {
    let mut buffer = vec![Complex::new(0.0, 0.0); fft_rows * fft_cols];

    match padding {
        PaddingMode::Valid => {
            for i in 0..input.rows {
                for j in 0..input.cols {
                    let val = input.data[i * input.cols + j].to_f64();
                    buffer[i * fft_cols + j] = Complex::new(val, 0.0);
                }
            }
        }
        PaddingMode::Zero(pad_rows, pad_cols) => {
            for i in 0..input.rows {
                for j in 0..input.cols {
                    let row = i + *pad_rows;
                    let col = j + *pad_cols;
                    if row < fft_rows && col < fft_cols {
                        let val = input.data[i * input.cols + j].to_f64();
                        buffer[row * fft_cols + col] = Complex::new(val, 0.0);
                    }
                }
            }
        }
        PaddingMode::Mirror(pad_rows, pad_cols) => {
            // Базовое копирование
            for i in 0..input.rows {
                for j in 0..input.cols {
                    let row = i + *pad_rows;
                    let col = j + *pad_cols;
                    if row < fft_rows && col < fft_cols {
                        let val = input.data[i * input.cols + j].to_f64();
                        buffer[row * fft_cols + col] = Complex::new(val, 0.0);
                    }
                }
            }
            // TODO: Добавить зеркальное отражение
        }
    }
    buffer
}

// Подготовка ядра: отзеркаливание + T → f64 + shift
fn prepare_kernel_f64<T: Float + Copy>(
    kernel: &Matrix<T>,
    fft_rows: usize,
    fft_cols: usize,
) -> Vec<Complex<f64>> {
    let mut buffer = vec![Complex::new(0.0, 0.0); fft_rows * fft_cols];

    let pad_rows = kernel.rows / 2;
    let pad_cols = kernel.cols / 2;

    for i in 0..kernel.rows {
        for j in 0..kernel.cols {
            let mirrored_i = kernel.rows - 1 - i;
            let mirrored_j = kernel.cols - 1 - j;
            let val = kernel.data[mirrored_i * kernel.cols + mirrored_j].to_f64();
            let row = (fft_rows + i - pad_rows) % fft_rows;
            let col = (fft_cols + j - pad_cols) % fft_cols;
            buffer[row * fft_cols + col] = Complex::new(val, 0.0);
        }
    }
    buffer
}

// 2D FFT through f64
fn fft_2d_f64(
    buffer: &mut [Complex<f64>],
    rows: usize,
    cols: usize,
    fft_row: &Arc<dyn rustfft::Fft<f64>>,
    fft_col: &Arc<dyn rustfft::Fft<f64>>,
) {
    // FFT on rows
    for row in 0..rows {
        let slice = &mut buffer[row * cols..(row + 1) * cols];
        fft_row.process(slice);
    }

    // FFT on columns
    let mut col_buffer = vec![Complex::new(0.0, 0.0); rows];
    for col in 0..cols {
        for row in 0..rows {
            col_buffer[row] = buffer[row * cols + col];
        }
        fft_col.process(&mut col_buffer);
        for row in 0..rows {
            buffer[row * cols + col] = col_buffer[row];
        }
    }
}

// Извлечение результата: f64 → T
fn extract_result_f64<T: Float + Copy>(
    buffer: &[Complex<f64>],
    stride: usize,
    output_rows: usize,
    output_cols: usize,
    padding: &PaddingMode,
    fft_rows: usize,
    fft_cols: usize,
) -> Matrix<T> {
    let scale = 1.0 / (fft_rows as f64 * fft_cols as f64);
    let mut result_data = vec![T::from_f64(0.0); output_rows * output_cols];

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
                result_data[i * output_cols + j] = T::from_f64(value);
            }
        }
    }

    Matrix::new(result_data, output_rows, output_cols)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::{linalg::Matrix, matrix};

    #[test]
    fn test_fft_conv() {
        let a: Matrix<f32> = Matrix::from_num(1.0, 6, 6);
        let sobel = matrix![[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]];
        let fft_res: Matrix<f32> = a.conv_zero_fft(&sobel);
        let direct_res: Matrix<f32> = a.conv_zero(&sobel);
        println!("FFT: {}", fft_res);
        println!("Direct: {}", direct_res);

        for i in 0..fft_res.data.len() {
            let diff = (fft_res.data[i] - direct_res.data[i]).abs();
            assert!(diff < 1e-10, "Mismatch at {}: {} vs {}", i, fft_res.data[i], direct_res.data[i]);
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
    }
}