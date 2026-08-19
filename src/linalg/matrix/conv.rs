use crate::{
    Float,
    linalg::{Matrix, fft::{Complex, Twiddles, fft_nd, split_and_multiply, twiddles_for}},
};

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

    /// Picks between the direct and the FFT path, whichever should be faster.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::{Matrix, PaddingMode};
    ///
    /// let img: Matrix<f64> = Matrix::from_num(1.0, 64, 64);
    /// let kernel: Matrix<f64> = Matrix::from_num(1.0 / 9.0, 3, 3);
    ///
    /// let out = img.smart_conv(&kernel, PaddingMode::Valid);
    /// assert_eq!(out.shape(), [62, 62]);
    /// ```
    ///
    /// # Arguments
    /// * `kernel` — the kernel to convolve with.
    /// * `mode` — see [PaddingMode].
    ///
    /// # Notes
    /// The choice comes from comparing the work each path does — see
    /// [prefers_fft]. Both paths agree to within floating point error, so the
    /// decision only ever changes the running time.
    pub fn smart_conv(&self, kernel: &Matrix<T>, mode: PaddingMode) -> Matrix<T> {
        if prefers_fft(self.rows, self.cols, kernel.rows, kernel.cols) {
            match mode {
                PaddingMode::Valid => self.conv_fft(kernel),
                PaddingMode::Zero(_, _) => self.conv_zero_fft(kernel),
                PaddingMode::Mirror(_, _) => self.conv_with_mirror_padding_fft(kernel),
            }
        } else {
            match mode {
                PaddingMode::Valid => self.conv(kernel),
                PaddingMode::Zero(_, _) => self.conv_zero(kernel),
                PaddingMode::Mirror(_, _) => self.conv_with_mirror_padding(kernel),
            }
        }
    }
}

/// Reports whether the FFT path should beat the direct one for these sizes.
///
/// # Example
/// ```
/// use tensorrs::linalg::prefers_fft;
///
/// // a tiny kernel over a small image: the direct loop wins easily
/// assert!(!prefers_fft(64, 64, 3, 3));
/// // a large kernel is what the FFT is for
/// assert!(prefers_fft(256, 256, 64, 64));
/// ```
///
/// # Arguments
/// * `input_rows`, `input_cols` — the size of the input.
/// * `kernel_rows`, `kernel_cols` — the size of the kernel.
///
/// # Notes
/// The direct path does `output_cells * kernel_cells` multiply-adds, the FFT path
/// two transforms of the input rounded up to a power of two. Comparing the two
/// makes the rule independent of the machine it was measured on, unlike a table
/// of absolute timings; only one constant, the cost of a butterfly relative to a
/// multiply-add, carries any hardware assumption.
///
/// The rule picks the faster path on every size in `generate_conv_decision_table`,
/// covering kernels from `3x3` to `128x128` over inputs from `64x64` to
/// `1024x1024`. [Tensor::conv](crate::linalg::Tensor::conv) applies the same rule
/// at any rank.
pub fn prefers_fft(
    input_rows: usize,
    input_cols: usize,
    kernel_rows: usize,
    kernel_cols: usize,
) -> bool {
    crate::linalg::fft::prefers_fft_nd(&[input_rows, input_cols], &[kernel_rows, kernel_cols])
}

fn fft_convolution_2d<T: Float + Copy>(
    input: &Matrix<T>,
    kernel: &Matrix<T>,
    output_rows: usize,
    output_cols: usize,
    padding: PaddingMode,
) -> Matrix<T> {
    // The size of the circular convolution. Normally input + kernel - 1 is
    // needed so that the wrap-around cannot spoil the result. But Valid takes
    // only those outputs whose window lies wholly inside the input: with a
    // centred kernel the extracted region [(k-1)/2 .. L-k+(k-1)/2] touches
    // exactly the indices 0..L-1, so the wrap-around never reaches it. The size
    // of the input itself is therefore enough here - and that is precisely four
    // times less work whenever the side of the input is already a power of two.
    let (fft_rows, fft_cols) = match padding {
        PaddingMode::Valid => (
            input.rows.next_power_of_two(),
            input.cols.next_power_of_two(),
        ),
        _ => (
            (input.rows + kernel.rows - 1).next_power_of_two(),
            (input.cols + kernel.cols - 1).next_power_of_two(),
        ),
    };

    // The input and the kernel are both real, so they fit into a single complex
    // field: re = input, img = kernel. One forward transform instead of two -
    // the spectra are separated afterwards by Hermitian symmetry.
    let mut input_buf = prepare_packed(input, kernel, fft_rows, fft_cols, &padding);

    // both transforms share these, so they are built once
    let shape = [fft_rows, fft_cols];
    let twiddles: Vec<Twiddles<T>> = twiddles_for(&shape);
    let mut scratch = vec![Complex::zero(); fft_rows * fft_cols];

    // forward transform of the packed field
    fft_nd(&mut input_buf, &mut scratch, &shape, &twiddles, false);

    // split the two spectra apart and multiply them in one pass
    let mut input_buf = split_and_multiply(&input_buf, &shape);

    // and back
    fft_nd(&mut input_buf, &mut scratch, &shape, &twiddles, true);

    // prepare_kernel lays the mirrored kernel down shifted by -k/2, so the
    // wanted result of the circular convolution sits in the buffer at an offset
    // of (k-1)/2 rather than in the corner. For odd kernels that coincides with
    // k/2, for even ones it differs by one - which is exactly where the direct
    // and the FFT path used to diverge.
    let start_row = kernel.rows.saturating_sub(1) / 2;
    let start_col = kernel.cols.saturating_sub(1) / 2;

    extract_result(&input_buf, fft_cols, output_rows, output_cols, start_row, start_col, fft_rows, fft_cols)
}

/// Packs the input into the real part and the kernel into the imaginary one.
fn prepare_packed<T: Float + Copy>(
    input: &Matrix<T>,
    kernel: &Matrix<T>,
    fft_rows: usize,
    fft_cols: usize,
    padding: &PaddingMode,
) -> Vec<Complex<T>> {
    let mut buffer = prepare_input(input, fft_rows, fft_cols, padding);
    let kernel_buf = prepare_kernel(kernel, fft_rows, fft_cols);

    for (slot, k) in buffer.iter_mut().zip(kernel_buf.iter()) {
        slot.img = k.re;
    }

    buffer
}

// Preparing the input: T -> Complex<T>
fn prepare_input<T: Float + Copy>(
    input: &Matrix<T>,
    fft_rows: usize,
    fft_cols: usize,
    padding: &PaddingMode,
) -> Vec<Complex<T>> {
    let mut buffer = vec![Complex::zero(); fft_rows * fft_cols];

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
    let mut buffer = vec![Complex::zero(); fft_rows * fft_cols];

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


fn extract_result<T: Float + Copy>(
    buffer: &[Complex<T>],
    stride: usize,
    output_rows: usize,
    output_cols: usize,
    start_row: usize,
    start_col: usize,
    fft_rows: usize,
    fft_cols: usize,
) -> Matrix<T> {
    let denom = fft_rows * fft_cols;
    let scale = T::one() / T::from_usize(denom);
    let mut result_data = vec![T::default(); output_rows * output_cols];

    let buffer_rows = buffer.len() / stride;

    for i in 0..output_rows {
        for j in 0..output_cols {
            let buf_row = start_row + i;
            let buf_col = start_col + j;
            if buf_row < buffer_rows && buf_col < stride {
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
    
    #[test]
    fn fft_path_agrees_with_the_direct_one() {
        use crate::linalg::PaddingMode;

        fn grid(r: usize, c: usize) -> Matrix<f64> {
            Matrix::new((0..r * c).map(|i| (i + 1) as f64).collect(), r, c)
        }
        /// Relative difference: these grids reach ~1e6, so an absolute bound
        /// would be measuring the magnitude of the data rather than the error.
        fn max_diff(a: &Matrix<f64>, b: &Matrix<f64>) -> f64 {
            let (a, b) = (a.get_data(), b.get_data());
            let scale = a
                .iter()
                .fold(1.0f64, |m, v| if v.abs() > m { v.abs() } else { m });

            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs() / scale)
                .fold(0.0f64, |m, v| if v > m { v } else { m })
        }

        // both parities of the kernel: the centering shift in prepare_kernel is
        // k / 2, but the result sits at (k - 1) / 2, which only differ for even k
        for (ir, ic, kr, kc) in [
            (5, 5, 3, 3),
            (9, 7, 5, 3),
            (8, 8, 4, 4),
            (16, 16, 2, 2),
            (10, 10, 6, 4),
            (7, 7, 1, 1),
            // powers of two: Valid takes the reduced FFT size here
            (8, 8, 3, 3),
            (16, 8, 5, 5),
            (32, 32, 7, 7),
            // a kernel nearly as wide as the input leaves almost no valid region
            (8, 8, 7, 7),
            (9, 9, 8, 8),
            (4, 4, 4, 4),
            // one row or column
            (1, 16, 1, 3),
            (16, 1, 3, 1),
        ] {
            let img = grid(ir, ic);
            let kernel = grid(kr, kc);

            // a few ulps of an f64 FFT, with room for the accumulation over a
            // transform of this size
            let tol = 1e-10;

            assert!(max_diff(&img.conv(&kernel), &img.conv_fft(&kernel)) < tol,
                "Valid disagrees for a {ir}x{ic} input and a {kr}x{kc} kernel");
            assert!(max_diff(&img.conv_zero(&kernel), &img.conv_zero_fft(&kernel)) < tol,
                "Zero disagrees for a {ir}x{ic} input and a {kr}x{kc} kernel");
            assert!(max_diff(&img.conv_with_mirror_padding(&kernel),
                             &img.conv_with_mirror_padding_fft(&kernel)) < tol,
                "Mirror disagrees for a {ir}x{ic} input and a {kr}x{kc} kernel");

            // smart_conv picks a path on its own, so it has to match either way
            assert!(max_diff(&img.conv(&kernel),
                             &img.smart_conv(&kernel, PaddingMode::Valid)) < tol,
                "smart_conv disagrees for a {ir}x{ic} input and a {kr}x{kc} kernel");
        }
    }

    #[test]
    fn pi_is_accurate_enough_for_an_f64_fft() {
        use crate::Float;
        // a truncated pi poisons every twiddle factor
        assert!((f64::pi() - std::f64::consts::PI).abs() < 1e-15);
        assert!((f32::pi() - std::f32::consts::PI).abs() < 1e-7);
    }

    #[test]
    fn fft_path_agrees_with_the_direct_one_in_f32() {
        let img: Matrix<f32> = Matrix::new((0..64).map(|i| ((i % 7) as f32) * 0.25).collect(), 8, 8);
        let kernel: Matrix<f32> = Matrix::new(
            vec![0.5, -1.0, 0.25, 2.0, 0.0, -0.5, 1.0, 0.75, -0.25],
            3,
            3,
        );

        let direct = img.conv(&kernel).get_data();
        let fft = img.conv_fft(&kernel).get_data();

        for (d, f) in direct.iter().zip(fft.iter()) {
            assert!((d - f).abs() < 1e-3, "f32 paths disagree: {d} vs {f}");
        }
    }

    #[test]
    #[ignore = "benchmark: run with `cargo test --release -- --ignored --nocapture`"]
    fn generate_conv_decision_table() {
        use super::prefers_fft;
        use std::hint::black_box;
        use std::time::Instant;

        /// Best of `runs`, after one warm-up call. A single measurement catches
        /// the first-touch page faults and the rayon pool spinning up.
        fn best_ms(mut f: impl FnMut(), runs: usize) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..runs {
                let start = Instant::now();
                f();
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                if ms < best {
                    best = ms;
                }
            }
            best
        }

        println!("{:^8} | {:^11} | {:^10} | {:^10} | {:^7} | {:^8} | {:^7}",
                 "Kernel", "Matrix", "FFT(ms)", "Direct(ms)", "FFT/Dir", "prefers", "right?");
        println!("{:-<75}", "");

        let sizes = [
            (3, 64), (3, 128), (3, 256), (3, 512), (3, 1024),
            (16, 64), (16, 128), (16, 256), (16, 512), (16, 1024),
            (64, 64), (64, 128), (64, 256), (64, 512), (64, 1024),
            (128, 128), (128, 256), (128, 512), (128, 1024),
        ];

        for (k, n) in sizes {
            let a: Matrix<f32> = Matrix::randn(n, n);
            let b: Matrix<f32> = Matrix::randn(k, k);

            // both paths measured on the same mode, otherwise the FFT sizes differ
            let runs = if n >= 1024 { 3 } else { 5 };
            let fft = best_ms(|| { black_box(a.conv_fft(&b)); }, runs);
            let direct = best_ms(|| { black_box(a.conv(&b)); }, runs);

            let chose_fft = prefers_fft(n, n, k, k);
            let correct = chose_fft == (fft < direct);

            println!("{:^8} | {:^11} | {:^10.2} | {:^10.2} | {:^7.2} | {:^8} | {:^7}",
                     k, format!("{n}x{n}"), fft, direct, fft / direct,
                     if chose_fft { "FFT" } else { "direct" },
                     if correct { "yes" } else { "NO" });
        }
    }
}
