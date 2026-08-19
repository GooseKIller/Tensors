use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

use crate::{
    Float, Num,
    linalg::{
        Tensor, product,
        tensor::ops::PARALLEL_THRESHOLD,
        fft::{Complex, fft_nd, prefers_fft_nd, split_and_multiply, twiddles_for},
    },
};

/// What a convolution does at the edges of the input.
///
/// # Example
/// ```
/// use tensorrs::{tensor, linalg::{Padding, Tensor}};
///
/// let img = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
/// let blur: Tensor<f64> = Tensor::from_num(1.0 / 9.0, vec![3, 3]);
///
/// // Valid only keeps windows that fit: a 3x3 kernel leaves a single pixel
/// assert_eq!(img.conv(&blur, Padding::Valid).get_shape(), vec![1, 1]);
///
/// // the padded modes keep the size instead
/// assert_eq!(img.conv(&blur, Padding::Zeros).get_shape(), vec![3, 3]);
/// assert_eq!(img.conv(&blur, Padding::Mirror).get_shape(), vec![3, 3]);
/// ```
///
/// # Notes
/// The variants are named after what fills the border. Both padded modes keep the
/// output the same size as the input; they differ in what a window sees when it
/// hangs over the edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    /// No padding: only windows that fit entirely inside the input are taken, so
    /// each axis shrinks by `kernel - 1`.
    Valid,
    /// Zeros around the input. The output keeps the size of the input, at the
    /// cost of a dark border the filter can see.
    Zeros,
    /// The input mirrored around its edges. The output keeps the size of the
    /// input, and no edge is introduced that was not in the data.
    Mirror,
}

/// The padding that keeps the output the same size as the input.
///
/// Split unevenly for an even kernel, which has no centre to sit on.
fn size_preserving(kernel_shape: &[usize]) -> Vec<(usize, usize)> {
    kernel_shape
        .iter()
        .map(|&k| ((k - 1) / 2, k / 2))
        .collect()
}

impl<T: Float> Tensor<T> {
    /// Convolves the tensor with a kernel of the same rank.
    ///
    /// # Formula
    ///```math
    ///  y_{\mathbf{i}} = \sum_{\mathbf{u}} x_{\mathbf{i} + \mathbf{u}} \, k_{\mathbf{u}}
    ///```
    /// Where $`\mathbf{i}`$ and $`\mathbf{u}`$ run over all axes at once, so the
    /// same definition covers one, two and any number of dimensions
    ///
    /// # Example
    /// ```
    /// use tensorrs::{tensor, linalg::Padding};
    ///
    /// // a vertical edge detector over a horizontal ramp
    /// let img = tensor![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];
    /// let sobel = tensor![[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    ///
    /// assert_eq!(img.conv(&sobel, Padding::Valid).get_data(), vec![8.0]);
    ///
    /// // 1-D works the same way: a moving sum of three
    /// let signal = tensor![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let window = tensor![1.0, 1.0, 1.0];
    /// assert_eq!(signal.conv(&window, Padding::Valid).get_data(), vec![6.0, 9.0, 12.0, 15.0]);
    /// ```
    ///
    /// # Arguments
    /// * `kernel` — the kernel, of the same rank as the tensor and no larger on
    ///   any axis.
    /// * `padding` — see [Padding].
    ///
    /// # Panics
    /// If the ranks disagree, or the kernel is larger than the input on some axis.
    ///
    /// # Notes
    /// This is a cross-correlation: the kernel is *not* flipped, which is the
    /// convention every machine learning framework uses.
    ///
    /// The windows are strided views, so nothing is copied to build them. For a
    /// trainable convolution inside a model use [Conv2d](crate::nn::Conv2d),
    /// which goes through the autodiff graph.
    pub fn conv(&self, kernel: &Tensor<T>, padding: Padding) -> Tensor<T> {
        match padding {
            Padding::Valid => self.conv_dispatch(kernel),
            Padding::Zeros => self
                .pad_zero(&size_preserving(&kernel.shape))
                .conv_dispatch(kernel),
            Padding::Mirror => self
                .pad_mirror(&size_preserving(&kernel.shape))
                .conv_dispatch(kernel),
        }
    }

    /// Runs the convolution down whichever path should be faster.
    fn conv_dispatch(&self, kernel: &Tensor<T>) -> Tensor<T> {
        if prefers_fft_nd(&self.shape, &kernel.shape) {
            self.conv_fft(kernel)
        } else {
            self.conv_valid(kernel)
        }
    }

    /// Convolves through the frequency domain.
    ///
    /// A multiplication of spectra is a *circular* convolution, so the result
    /// normally needs a transform of `input + kernel - 1` to keep the wrap-around
    /// out of it. Here only the windows that fit inside the input are kept, and
    /// with the kernel centred their region touches indices `0..len` only — so a
    /// transform the size of the input is enough, which is four times less work
    /// whenever an axis already sits on a power of two.
    fn conv_fft(&self, kernel: &Tensor<T>) -> Tensor<T> {
        let ndim = self.shape.len();

        let fft_shape: Vec<usize> = self.shape.iter().map(|&d| d.next_power_of_two()).collect();
        let total = product(&fft_shape[..]);

        let mut strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * fft_shape[i + 1];
        }

        let mut buffer = vec![Complex::zero(); total];

        // the input goes into the real part, at the origin
        let src = self.packed_data();
        for (flat, &value) in src.iter().enumerate() {
            let mut rem = flat;
            let mut at = 0usize;
            for i in (0..ndim).rev() {
                at += (rem % self.shape[i]) * strides[i];
                rem /= self.shape[i];
            }
            buffer[at].re = value;
        }

        // the kernel goes into the imaginary part, mirrored so that the spectral
        // product comes out as a correlation, and shifted so that it sits centred
        let ker = kernel.packed_data();
        let mut coord = vec![0usize; ndim];

        for flat in 0..ker.len() {
            let mut rem = flat;
            for i in (0..ndim).rev() {
                coord[i] = rem % kernel.shape[i];
                rem /= kernel.shape[i];
            }

            let mut mirrored = 0usize;
            let mut kernel_stride = 1usize;
            for i in (0..ndim).rev() {
                mirrored += (kernel.shape[i] - 1 - coord[i]) * kernel_stride;
                kernel_stride *= kernel.shape[i];
            }

            let mut at = 0usize;
            for i in 0..ndim {
                let pad = kernel.shape[i] / 2;
                at += ((fft_shape[i] + coord[i] - pad) % fft_shape[i]) * strides[i];
            }

            buffer[at].img = ker[mirrored];
        }

        // one forward transform for both signals, one inverse for the product
        let twiddles = twiddles_for(&fft_shape);
        let mut scratch = vec![Complex::zero(); total];

        fft_nd(&mut buffer, &mut scratch, &fft_shape, &twiddles, false);
        let mut spectrum = split_and_multiply(&buffer, &fft_shape);
        fft_nd(&mut spectrum, &mut scratch, &fft_shape, &twiddles, true);

        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .zip(kernel.shape.iter())
            .map(|(&dim, &k)| dim - k + 1)
            .collect();

        // the centring shift puts the wanted region at (k - 1) / 2 on each axis,
        // and the 1/N of the inverse transform is folded in here
        let scale = T::one() / T::from_usize(total);
        let mut out_data = vec![T::default(); product(&out_shape[..])];

        if out_data.len() < PARALLEL_THRESHOLD {

            out_data.iter_mut().enumerate().for_each(|(flat, slot)| {
    let mut rem = flat;
    let mut at = 0usize;
    for i in (0..ndim).rev() {
        let c = rem % out_shape[i];
        rem /= out_shape[i];
        at += (c + (kernel.shape[i] - 1) / 2) * strides[i];
    }
    *slot = spectrum[at].re * scale;
});

        } else {

            out_data.par_iter_mut().enumerate().for_each(|(flat, slot)| {
    let mut rem = flat;
    let mut at = 0usize;
    for i in (0..ndim).rev() {
        let c = rem % out_shape[i];
        rem /= out_shape[i];
        at += (c + (kernel.shape[i] - 1) / 2) * strides[i];
    }
    *slot = spectrum[at].re * scale;
});

        }

        Tensor::new(out_data, out_shape)
    }

    /// Convolves without padding, taking only the windows that fit.
    ///
    /// The kernel offsets are the same for every output position, so they are
    /// computed once; the inner loop then walks a flat offset table instead of
    /// decoding coordinates per element.
    fn conv_valid(&self, kernel: &Tensor<T>) -> Tensor<T> {
        let ndim = self.shape.len();

        assert_eq!(ndim, kernel.shape.len(),
            "!!!conv(): the kernel has rank {} but the input has rank {ndim}!!!",
            kernel.shape.len());

        for (axis, (&dim, &k)) in self.shape.iter().zip(kernel.shape.iter()).enumerate() {
            assert!(k <= dim,
                "!!!conv(): the kernel is {k} wide on axis {axis} but the input is only {dim}!!!");
        }

        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .zip(kernel.shape.iter())
            .map(|(&dim, &k)| dim - k + 1)
            .collect();

        let src = self.packed_data();
        let ker = kernel.packed_data();

        // row-major strides of the packed input
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }

        // where each kernel element sits relative to the corner of a window
        let offsets: Vec<usize> = (0..ker.len())
            .map(|flat| {
                let mut rem = flat;
                let mut offset = 0;
                for i in (0..ndim).rev() {
                    offset += (rem % kernel.shape[i]) * strides[i];
                    rem /= kernel.shape[i];
                }
                offset
            })
            .collect();

        let mut out_data = vec![T::default(); product(&out_shape[..])];

        if out_data.len() < PARALLEL_THRESHOLD {

            out_data.iter_mut().enumerate().for_each(|(out_idx, slot)| {
            // the corner of this window, in the packed input
            let mut rem = out_idx;
            let mut corner = 0usize;
            for i in (0..ndim).rev() {
                corner += (rem % out_shape[i]) * strides[i];
                rem /= out_shape[i];
            }

            let mut acc = T::default();
            for (offset, &weight) in offsets.iter().zip(ker.iter()) {
                acc = acc + src[corner + offset] * weight;
            }

            *slot = acc;
        });

        } else {

            out_data.par_iter_mut().enumerate().for_each(|(out_idx, slot)| {
            // the corner of this window, in the packed input
            let mut rem = out_idx;
            let mut corner = 0usize;
            for i in (0..ndim).rev() {
                corner += (rem % out_shape[i]) * strides[i];
                rem /= out_shape[i];
            }

            let mut acc = T::default();
            for (offset, &weight) in offsets.iter().zip(ker.iter()) {
                acc = acc + src[corner + offset] * weight;
            }

            *slot = acc;
        });

        }

        Tensor::new(out_data, out_shape)
    }
}

impl<T: Num> Tensor<T> {
    /// Lays every sliding window out as a row — the `im2col` transform.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// // one image, one channel, 3x3
    /// let x = Tensor::new((1..=9).collect::<Vec<i32>>(), vec![1, 1, 3, 3]);
    /// let cols = x.unfold_2d((2, 2), (1, 1), (0, 0));
    ///
    /// // 2x2 windows at 4 positions, each flattened to 4 values
    /// assert_eq!(cols.get_shape(), vec![4, 4]);
    /// assert_eq!(cols.get_data()[..4], [1, 2, 4, 5]);
    /// ```
    ///
    /// # Arguments
    /// * `kernel` — the window size, `(height, width)`.
    /// * `stride` — the step between two windows, `(height, width)`.
    /// * `padding` — how many zero rows and columns to add on each side.
    ///
    /// # Returns
    /// A tensor of shape `[batch * out_h * out_w, channels * kernel_h * kernel_w]`,
    /// where `out = (size + 2 * padding - kernel) / stride + 1`.
    ///
    /// # Panics
    /// If the input is not 4-D (`[batch, channels, height, width]`), if a stride
    /// is zero, or if the padded input is smaller than the kernel.
    ///
    /// # Notes
    /// This is what turns a convolution into a matrix multiplication: multiplying
    /// the result by a weight matrix of shape `[channels * kh * kw, out_channels]`
    /// convolves every window with every filter in one [Tensor::matmul] call. That
    /// is how [Conv2d](crate::nn::Conv2d) is built.
    ///
    /// Windows overlap, so the result holds more values than the input — for a
    /// `3x3` kernel with stride 1 roughly nine times as many.
    pub fn unfold_2d(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Tensor<T> {
        let (batch, channels, height, width) = self.unfold_dims(kernel, stride, padding);
        let (kernel_h, kernel_w) = kernel;
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;

        let out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        let out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

        let patch = channels * kernel_h * kernel_w;
        let src = self.packed_data();
        let mut data = vec![T::default(); batch * out_h * out_w * patch];

        // every window lands in its own row, so the rows never race
        data.par_chunks_mut(patch).enumerate().for_each(|(row, out)| {
            let ow = row % out_w;
            let oh = (row / out_w) % out_h;
            let n = row / (out_h * out_w);

            for c in 0..channels {
                for i in 0..kernel_h {
                    let y = match (oh * stride_h + i).checked_sub(pad_h) {
                        Some(y) if y < height => y,
                        _ => continue, // inside the padding: stays zero
                    };

                    for j in 0..kernel_w {
                        let x = match (ow * stride_w + j).checked_sub(pad_w) {
                            Some(x) if x < width => x,
                            _ => continue,
                        };

                        out[c * kernel_h * kernel_w + i * kernel_w + j] =
                            src[((n * channels + c) * height + y) * width + x];
                    }
                }
            }
        });

        Tensor::new(data, vec![batch * out_h * out_w, patch])
    }

    /// Folds unfolded windows back into an image, summing wherever they overlap —
    /// the inverse of [Tensor::unfold_2d].
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let x = Tensor::new((1..=9).collect::<Vec<i32>>(), vec![1, 1, 3, 3]);
    /// let cols = x.unfold_2d((2, 2), (1, 1), (0, 0));
    ///
    /// let back = cols.fold_2d([1, 1, 3, 3], (2, 2), (1, 1), (0, 0));
    /// // the centre belongs to all four windows, so it comes back four times over
    /// assert_eq!(back.get_data(), vec![1, 4, 3, 8, 20, 12, 7, 16, 9]);
    /// ```
    ///
    /// # Arguments
    /// * `output_shape` — the `[batch, channels, height, width]` to fold back into.
    /// * `kernel`, `stride`, `padding` — the same values [Tensor::unfold_2d] was given.
    ///
    /// # Panics
    /// If `self` is not the 2-D result of an unfold with these parameters.
    ///
    /// # Notes
    /// Summing the overlaps is exactly what the backward pass of a convolution
    /// needs: a pixel read by several windows collects a gradient from each of them.
    /// It is *not* a right inverse — folding an unfold multiplies overlapping
    /// pixels by how many windows saw them.
    pub fn fold_2d(
        &self,
        output_shape: [usize; 4],
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Tensor<T> {
        let [batch, channels, height, width] = output_shape;
        let (kernel_h, kernel_w) = kernel;
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;

        assert!(stride_h > 0 && stride_w > 0, "!!!fold_2d(): stride must be positive!!!");

        let out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        let out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
        let patch = channels * kernel_h * kernel_w;

        assert_eq!(self.shape, vec![batch * out_h * out_w, patch],
            "!!!fold_2d(): shape {:?} does not match an unfold of {output_shape:?}!!!", self.shape);

        let src = self.packed_data();
        let mut data = vec![T::default(); batch * channels * height * width];

        // one image per task: overlaps are summed inside an image, never across
        data.par_chunks_mut(channels * height * width)
            .enumerate()
            .for_each(|(n, image)| {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let row = (n * out_h + oh) * out_w + ow;

                        for c in 0..channels {
                            for i in 0..kernel_h {
                                let y = match (oh * stride_h + i).checked_sub(pad_h) {
                                    Some(y) if y < height => y,
                                    _ => continue,
                                };

                                for j in 0..kernel_w {
                                    let x = match (ow * stride_w + j).checked_sub(pad_w) {
                                        Some(x) if x < width => x,
                                        _ => continue,
                                    };

                                    image[(c * height + y) * width + x] +=
                                        src[row * patch + c * kernel_h * kernel_w + i * kernel_w + j];
                                }
                            }
                        }
                    }
                }
            });

        Tensor::new(data, vec![batch, channels, height, width])
    }

    /// Validates the arguments of an unfold and returns `[batch, channels, height, width]`.
    fn unfold_dims(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> (usize, usize, usize, usize) {
        assert_eq!(self.shape.len(), 4,
            "!!!unfold_2d() expects [batch, channels, height, width], got {:?}!!!", self.shape);
        assert!(stride.0 > 0 && stride.1 > 0, "!!!unfold_2d(): stride must be positive!!!");

        let (height, width) = (self.shape[2], self.shape[3]);
        assert!(height + 2 * padding.0 >= kernel.0 && width + 2 * padding.1 >= kernel.1,
            "!!!unfold_2d(): a {:?} kernel does not fit a {height}x{width} input padded by {:?}!!!",
            kernel, padding);

        (self.shape[0], self.shape[1], height, width)
    }
}

#[cfg(test)]
mod tests{
    use crate::tensor;

    #[test]
    fn simple_conv() {
        let img = tensor![
            [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
            [0.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0],
            [1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0],
            [1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0],
            [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],
            [1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0],
            [1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0],
            [0.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0],
            [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
        ];

        let kernel = tensor![[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];

        println!("{}", img.conv(&kernel, crate::linalg::Padding::Valid));

        let w = img.window(&kernel.shape, &[1,1]).next().unwrap();
        println!("{:?}", w.mul_sum(&kernel));
    }

    #[test]
    fn padded_modes_keep_the_size_at_every_rank() {
        use crate::linalg::{Padding, Tensor};

        // an all-ones input convolved with an all-ones kernel: mirror padding of
        // a constant is that same constant, so every output equals the kernel size
        for shape in [vec![6], vec![5, 5], vec![4, 4, 4]] {
            let kernel_shape = vec![3; shape.len()];
            let cells = 3usize.pow(shape.len() as u32);

            let input: Tensor<f64> = Tensor::from_num(1.0, shape.clone());
            let kernel: Tensor<f64> = Tensor::from_num(1.0, kernel_shape.clone());

            let mirrored = input.conv(&kernel, Padding::Mirror);
            assert_eq!(mirrored.get_shape(), shape, "Mirror changed the shape of {shape:?}");
            assert!(mirrored.get_data().iter().all(|&v| v == cells as f64),
                "Mirror of a constant should stay constant for {shape:?}");

            let zeroed = input.conv(&kernel, Padding::Zeros);
            assert_eq!(zeroed.get_shape(), shape, "Zeros changed the shape of {shape:?}");
            // an interior position sees no padding, so it still sums the whole
            // kernel; index 1 on every axis is interior for a pad of 1
            let interior = vec![1usize; shape.len()];
            assert_eq!(zeroed.get(&interior), Some(&(cells as f64)),
                "the interior of a Zeros convolution should be untouched for {shape:?}");

            let valid = input.conv(&kernel, Padding::Valid);
            let expected: Vec<usize> = shape.iter().map(|&d| d - 2).collect();
            assert_eq!(valid.get_shape(), expected);
        }
    }

    #[test]
    fn an_even_kernel_still_keeps_the_size() {
        use crate::linalg::{Padding, Tensor};

        // an even kernel has no centre, so the padding is split unevenly
        let input: Tensor<f64> = Tensor::from_num(1.0, vec![7, 5]);

        for k in [2usize, 4, 6] {
            let kernel: Tensor<f64> = Tensor::from_num(1.0, vec![k, k]);
            assert_eq!(input.conv(&kernel, Padding::Zeros).get_shape(), vec![7, 5],
                "a {k}x{k} kernel changed the shape");
            assert_eq!(input.conv(&kernel, Padding::Mirror).get_shape(), vec![7, 5],
                "a {k}x{k} kernel changed the shape");
        }
    }

    #[test]
    fn agrees_with_the_matrix_implementation() {
        use crate::linalg::{Matrix, Padding, Tensor};

        let data: Vec<f64> = (0..25).map(|i| ((i % 7) as f64) - 3.0).collect();
        let kernel_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let tensor = Tensor::new(data.clone(), vec![5, 5]);
        let matrix = Matrix::new(data, 5, 5);
        let t_kernel = Tensor::new(kernel_data.clone(), vec![3, 3]);
        let m_kernel = Matrix::new(kernel_data, 3, 3);

        assert_eq!(tensor.conv(&t_kernel, Padding::Valid).get_data(),
                   matrix.conv(&m_kernel).get_data());
        assert_eq!(tensor.conv(&t_kernel, Padding::Zeros).get_data(),
                   matrix.conv_zero(&m_kernel).get_data());
        assert_eq!(tensor.conv(&t_kernel, Padding::Mirror).get_data(),
                   matrix.conv_with_mirror_padding(&m_kernel).get_data());
    }

    #[test]
    fn the_fft_path_agrees_with_the_direct_one_at_every_rank() {
        use crate::linalg::Tensor;

        fn ramp(shape: Vec<usize>) -> Tensor<f64> {
            let n: usize = shape.iter().product();
            Tensor::new((0..n).map(|i| ((i % 61) as f64) - 30.0).collect(), shape)
        }

        fn max_relative(a: &Tensor<f64>, b: &Tensor<f64>) -> f64 {
            let (a, b) = (a.get_data(), b.get_data());
            let scale = a
                .iter()
                .fold(1.0f64, |m, v| if v.abs() > m { v.abs() } else { m });

            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs() / scale)
                .fold(0.0f64, |m, v| if v > m { v } else { m })
        }

        let cases: Vec<(Vec<usize>, Vec<usize>)> = vec![
            // 1-D, both parities
            (vec![64], vec![5]),
            (vec![100], vec![8]),
            // 2-D, powers of two take the reduced transform size
            (vec![32, 32], vec![3, 3]),
            (vec![64, 64], vec![4, 4]),
            (vec![48, 40], vec![7, 5]),
            (vec![17, 23], vec![2, 6]),
            // 3-D and 4-D
            (vec![16, 16, 16], vec![3, 3, 3]),
            (vec![16, 8, 12], vec![4, 2, 3]),
            (vec![8, 8, 8, 8], vec![3, 3, 3, 3]),
            // a kernel nearly as wide as the input
            (vec![9, 9], vec![8, 8]),
            (vec![8], vec![8]),
        ];

        for (shape, kernel_shape) in cases {
            let x = ramp(shape.clone());
            let kernel = ramp(kernel_shape.clone());

            let direct = x.conv_valid(&kernel);
            let fft = x.conv_fft(&kernel);

            assert_eq!(direct.get_shape(), fft.get_shape(),
                "shapes differ for {shape:?} with a {kernel_shape:?} kernel");
            assert!(max_relative(&direct, &fft) < 1e-10,
                "values differ for {shape:?} with a {kernel_shape:?} kernel: {:e}",
                max_relative(&direct, &fft));
        }
    }

    #[test]
    fn the_dispatch_reaches_the_fft_path() {
        use crate::linalg::{Padding, Tensor};

        // a large kernel is where the FFT wins, so conv() has to route there and
        // still agree with the direct result
        let x: Tensor<f64> = Tensor::new(
            (0..256 * 256).map(|i| ((i % 61) as f64) - 30.0).collect(),
            vec![256, 256],
        );
        let kernel: Tensor<f64> = Tensor::new(
            (0..48 * 48).map(|i| ((i % 13) as f64) - 6.0).collect(),
            vec![48, 48],
        );

        assert!(crate::linalg::prefers_fft(256, 256, 48, 48),
            "the rule should send this size to the FFT");

        let dispatched = x.conv(&kernel, Padding::Valid);
        let direct = x.conv_valid(&kernel);

        let scale = direct
            .get_data()
            .iter()
            .fold(1.0f64, |m, v| if v.abs() > m { v.abs() } else { m });

        for (d, f) in direct.get_data().iter().zip(dispatched.get_data().iter()) {
            assert!((d - f).abs() / scale < 1e-10, "{d} vs {f}");
        }
    }
}
