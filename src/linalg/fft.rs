//! The internal core of the fast Fourier transform.
//!
//! Everything both the matrix and the tensor convolution need lives here: a
//! complex number, a table of twiddle factors, the one-dimensional transform and
//! its generalisation to any number of axes.

use std::ops::{Add, Mul, Sub};

use rayon::prelude::*;

use crate::Float;

/// A complex number, written out rather than pulled in as a dependency.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Complex<T: Float> {
    pub re: T,
    pub img: T,
}

impl<T: Float> Complex<T> {
    pub(crate) fn new(re: T, img: T) -> Self {
        Self { re, img }
    }

    /// `e^{i theta}`
    pub(crate) fn expi(theta: T) -> Self {
        Self { re: theta.cos(), img: theta.sin() }
    }

    /// A complex zero, the value every buffer starts from.
    pub(crate) fn zero() -> Self {
        Self { re: T::default(), img: T::default() }
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

impl<T: Float> Mul<Complex<T>> for Complex<T> {
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

/// Precomputed twiddle factors for one transform length.
///
/// The naive loop walks `w *= w_len` and accumulates a rounding error over the
/// whole stage; reading exact factors out of a table removes that, and removes
/// the `cos`/`sin` call per stage as well. One table serves every stage, since
/// the factor of stage `len` at index `j` is `table[j * (n / len)]`.
pub(crate) struct Twiddles<T: Float> {
    fwd: Vec<Complex<T>>,
    inv: Vec<Complex<T>>,
}

impl<T: Float> Twiddles<T> {
    pub(crate) fn new(n: usize) -> Self {
        let half = n / 2;
        let two_pi = T::from_usize(2) * T::pi();
        let n_t = T::from_usize(n);

        let mut fwd = Vec::with_capacity(half);
        let mut inv = Vec::with_capacity(half);

        for k in 0..half {
            let angle = two_pi * T::from_usize(k) / n_t;
            // forward uses e^{-i0}, the inverse is its conjugate
            fwd.push(Complex::expi(-angle));
            inv.push(Complex::expi(angle));
        }

        Self { fwd, inv }
    }

    pub(crate) fn get(&self, inverse: bool) -> &[Complex<T>] {
        if inverse { &self.inv } else { &self.fwd }
    }
}

/// In-place radix-2 Cooley-Tukey transform of a power-of-two buffer.
pub(crate) fn fft_1d_inplace<T: Float + Copy>(buf: &mut [Complex<T>], twiddles: &[Complex<T>]) {
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

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        // the factor of this stage at index j is twiddles[j * step]
        let step = n / len;

        for i in (0..n).step_by(len) {
            for j in 0..half {
                let w = twiddles[j * step];
                let u = buf[i + j];
                let v = buf[i + j + half] * w;
                buf[i + j] = u + v;
                buf[i + j + half] = u - v;
            }
        }
        len <<= 1;
    }

    // NOTE: scaling by 1/N is done by the caller, once, at extraction time.
}

/// Transposes `src` (`rows` x `cols`) into `dst` (`cols` x `rows`), tile by tile
/// so that both sides stay in cache.
pub(crate) fn transpose_into<T: Float + Copy>(
    src: &[Complex<T>],
    rows: usize,
    cols: usize,
    dst: &mut [Complex<T>],
) {
    const TILE: usize = 32;

    for row_block in (0..rows).step_by(TILE) {
        let row_end = (row_block + TILE).min(rows);
        for col_block in (0..cols).step_by(TILE) {
            let col_end = (col_block + TILE).min(cols);
            for r in row_block..row_end {
                for c in col_block..col_end {
                    dst[c * rows + r] = src[r * cols + c];
                }
            }
        }
    }
}

/// Transforms every axis of a buffer laid out row-major in `shape`.
///
/// A multi-dimensional transform is separable, so it is a one-dimensional
/// transform along each axis in turn. Rather than walking the non-final axes with
/// a stride, each pass transforms the *last* axis — which is contiguous — and then
/// rotates it to the front. After `ndim` passes every axis has been transformed
/// and the layout is back where it started.
pub(crate) fn fft_nd<T: Float + Copy>(
    buffer: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
    shape: &[usize],
    twiddles: &[Twiddles<T>],
    inverse: bool,
) {
    let ndim = shape.len();
    let total = buffer.len();

    for pass in 0..ndim {
        // the rotation walks the axes backwards, so pass `p` handles axis `ndim-1-p`
        let axis = ndim - 1 - pass;
        let len = shape[axis];

        if len > 1 {
            let tw = twiddles[axis].get(inverse);
            buffer
                .par_chunks_mut(len)
                .for_each(|line| fft_1d_inplace(line, tw));
        }

        // move the axis just transformed to the front
        transpose_into(buffer, total / len, len, scratch);
        buffer.copy_from_slice(scratch);
    }
}

/// Builds one twiddle table per axis.
pub(crate) fn twiddles_for<T: Float>(shape: &[usize]) -> Vec<Twiddles<T>> {
    shape.iter().map(|&n| Twiddles::new(n)).collect()
}

/// Splits the spectra of two real signals packed into one field, and multiplies
/// them together.
///
/// Both operands of a convolution are real, so they fit into a single complex
/// field as `z = a + i b` and need one forward transform instead of two. Hermitian
/// symmetry pulls them back apart: `A[k] = (Z[k] + conj(Z[-k])) / 2` and
/// `B[k] = (Z[k] - conj(Z[-k])) / 2i`.
pub(crate) fn split_and_multiply<T: Float + Copy>(
    spectrum: &[Complex<T>],
    shape: &[usize],
) -> Vec<Complex<T>> {
    let half = T::one() / T::from_usize(2);
    let ndim = shape.len();

    // row-major strides of the spectrum
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let mut out = vec![Complex::zero(); spectrum.len()];

    out.par_iter_mut().enumerate().for_each(|(idx, slot)| {
        // index -k, taken per axis
        let mut rem = idx;
        let mut mirror = 0usize;
        for i in (0..ndim).rev() {
            let coord = rem % shape[i];
            rem /= shape[i];
            mirror += ((shape[i] - coord) % shape[i]) * strides[i];
        }

        let z = spectrum[idx];
        let m = spectrum[mirror];

        // A = (Z[k] + conj(Z[-k])) / 2
        let a = Complex::new((z.re + m.re) * half, (z.img - m.img) * half);
        // B = (Z[k] - conj(Z[-k])) / 2i, and dividing by i is a turn of -90 degrees
        let b = Complex::new((z.img + m.img) * half, (m.re - z.re) * half);

        *slot = a * b;
    });

    out
}

/// How much a radix-2 butterfly costs relative to one multiply-add of the direct
/// path. A butterfly is a complex multiply plus an add and a subtract, and the
/// direct loop vectorizes better, so the ratio sits well above one. Calibrated
/// against `generate_conv_decision_table`.
const BUTTERFLY_COST: f64 = 6.0;

/// Multiply-adds below which the direct path is taken regardless: everything
/// still fits in cache, and both paths finish in a fraction of a millisecond.
const SMALL_PROBLEM: f64 = 1e6;

/// Reports whether the FFT path should beat the direct one for these shapes.
///
/// The direct path does `output_cells * kernel_cells` multiply-adds. The FFT path
/// does two transforms costing about `cells * sum(log2(m_i))` butterflies each,
/// where `m_i` is each input axis rounded up to a power of two.
///
/// Comparing the two makes the rule independent of the machine it was measured
/// on, unlike a table of absolute timings.
pub(crate) fn prefers_fft_nd(shape: &[usize], kernel_shape: &[usize]) -> bool {
    if shape.len() != kernel_shape.len() {
        return false;
    }
    if shape.iter().zip(kernel_shape).any(|(&d, &k)| k > d) {
        return false;
    }

    let out_cells: f64 = shape
        .iter()
        .zip(kernel_shape)
        .map(|(&d, &k)| (d - k + 1) as f64)
        .product();
    let kernel_cells: f64 = kernel_shape.iter().map(|&k| k as f64).product();
    let direct = out_cells * kernel_cells;

    // Below this the whole problem stays in cache and the direct loop runs at a
    // rate the flop count does not capture. Both paths finish in well under a
    // millisecond here, so the exact choice hardly matters - but the model alone
    // guesses wrong on the way in.
    if direct < SMALL_PROBLEM {
        return false;
    }

    let transform: Vec<usize> = shape.iter().map(|&d| d.next_power_of_two()).collect();
    let cells: f64 = transform.iter().map(|&m| m as f64).product();
    let stages: f64 = transform.iter().map(|&m| (m as f64).log2()).sum();

    BUTTERFLY_COST * cells * stages < direct
}
