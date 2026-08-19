use crate::{
    Float,
    activation::Module,
    autodiff::{AutoGrad, VarRef, max_axis_op, reshape_op, sum_axis_op, unfold_op},
};

/// Shared shape arithmetic of the pooling layers.
fn pooled_size(
    size: (usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
) -> (usize, usize) {
    (
        (size.0 - kernel.0) / stride.0 + 1,
        (size.1 - kernel.1) / stride.1 + 1,
    )
}

/// Lays every pooling window of every channel out as a row.
///
/// [unfold_op](crate::autodiff::unfold_op) mixes the channels into one patch,
/// which is what a convolution wants and a pooling layer does not: pooling treats
/// the channels independently. Folding the channels into the batch first gives
/// each of them its own windows.
fn windows_per_channel<T: Float>(
    x: &VarRef<T>,
    shape: &[usize],
    kernel: (usize, usize),
    stride: (usize, usize),
) -> VarRef<T> {
    let merged = reshape_op(x, vec![shape[0] * shape[1], 1, shape[2], shape[3]]);
    unfold_op(&merged, kernel, stride, (0, 0))
}

/// Checks the input of a pooling layer and returns `[batch, channels, height, width]`.
fn pool_dims<T: Float>(x: &VarRef<T>, name: &str) -> Vec<usize> {
    let shape = x.value().get_shape();
    assert_eq!(shape.len(), 4,
        "!!!{name} expects [batch, channels, height, width], got {shape:?}!!!");
    shape
}

/// Keeps the largest value of every window.
///
/// # Formula
///```math
///  y_{n,c,i,j} = \max_{\substack{0 \le u < k_h \\ 0 \le v < k_w}}
///      x_{n,c,\, i s_h + u,\; j s_w + v}
///```
///
/// # Example
/// ```
/// use tensorrs::{linalg::Tensor, activation::Module, nn::MaxPool2d,
///                autodiff::{AutoGrad, Var}};
///
/// let x = Var::leaf(
///     Tensor::new((1..=16).map(|v| v as f32).collect(), vec![1, 1, 4, 4]),
///     false,
/// );
///
/// // 2x2 windows, no overlap
/// let y = MaxPool2d::new((2, 2)).forward(&x);
/// assert_eq!(y.value().get_shape(), vec![1, 1, 2, 2]);
/// assert_eq!(y.value().get_data(), vec![6.0, 8.0, 14.0, 16.0]);
/// ```
///
/// # Notes
/// Only the winner of each window receives a gradient — the others had no effect
/// on the output. That is what makes max pooling keep the strongest response and
/// discard the rest.
///
/// There is no padding: a window that would hang over the edge is simply not
/// taken, so the last rows or columns are dropped when the size does not divide
/// evenly.
pub struct MaxPool2d {
    kernel: (usize, usize),
    stride: (usize, usize),
}

impl MaxPool2d {
    /// Creates a pooling layer whose windows do not overlap.
    ///
    /// # Example
    /// ```
    /// use tensorrs::nn::MaxPool2d;
    ///
    /// // the usual choice: halves both sides
    /// let pool = MaxPool2d::new((2, 2));
    /// assert_eq!(pool.output_size(32, 32), (16, 16));
    /// ```
    ///
    /// # Arguments
    /// * `kernel` — the window size, `(height, width)`; the stride matches it.
    pub fn new(kernel: (usize, usize)) -> Self {
        Self::with_stride(kernel, kernel)
    }

    /// Creates a pooling layer with a stride of your choice.
    ///
    /// # Example
    /// ```
    /// use tensorrs::nn::MaxPool2d;
    ///
    /// // overlapping windows shrink the image more gently
    /// let pool = MaxPool2d::with_stride((3, 3), (2, 2));
    /// assert_eq!(pool.output_size(9, 9), (4, 4));
    /// ```
    ///
    /// # Arguments
    /// * `kernel` — the window size, `(height, width)`.
    /// * `stride` — the step between two windows.
    ///
    /// # Panics
    /// If a stride is zero.
    pub fn with_stride(kernel: (usize, usize), stride: (usize, usize)) -> Self {
        assert!(stride.0 > 0 && stride.1 > 0, "!!!MaxPool2d: stride must be positive!!!");
        Self { kernel, stride }
    }

    /// Returns the spatial size this layer turns `[height, width]` into.
    pub fn output_size(&self, height: usize, width: usize) -> (usize, usize) {
        pooled_size((height, width), self.kernel, self.stride)
    }
}

impl<T: Float> Module<T> for MaxPool2d {
    /// Reduces every window to its largest value.
    ///
    /// # Arguments
    /// * `x` — the input, of shape `[batch, channels, height, width]`.
    ///
    /// # Returns
    /// `[batch, channels, out_height, out_width]`, see [MaxPool2d::output_size].
    ///
    /// # Panics
    /// If the input is not 4-D, or is smaller than one window.
    fn forward(&self, x: &VarRef<T>) -> VarRef<T> {
        let shape = pool_dims(x, "MaxPool2d");
        let (out_h, out_w) = self.output_size(shape[2], shape[3]);

        let cols = windows_per_channel(x, &shape, self.kernel, self.stride);
        let pooled = max_axis_op(&cols, 1);

        reshape_op(&pooled, vec![shape[0], shape[1], out_h, out_w])
    }

    fn parameters(&self) -> Vec<VarRef<T>> {
        vec![]
    }
}

/// Replaces every window with its mean.
///
/// # Formula
///```math
///  y_{n,c,i,j} = \frac{1}{k_h k_w} \sum_{u=0}^{k_h - 1} \sum_{v=0}^{k_w - 1}
///      x_{n,c,\, i s_h + u,\; j s_w + v}
///```
///
/// # Example
/// ```
/// use tensorrs::{linalg::Tensor, activation::Module, nn::AvgPool2d,
///                autodiff::{AutoGrad, Var}};
///
/// let x = Var::leaf(
///     Tensor::new((1..=16).map(|v| v as f32).collect(), vec![1, 1, 4, 4]),
///     false,
/// );
///
/// let y = AvgPool2d::new((2, 2)).forward(&x);
/// assert_eq!(y.value().get_data(), vec![3.5, 5.5, 11.5, 13.5]);
/// ```
///
/// # Notes
/// Every element of a window gets an equal share of the gradient, unlike
/// [MaxPool2d] where all of it goes to the winner. Averaging keeps more of the
/// signal and blurs it; taking the maximum keeps the strongest response and
/// throws the rest away.
pub struct AvgPool2d {
    kernel: (usize, usize),
    stride: (usize, usize),
}

impl AvgPool2d {
    /// Creates a pooling layer whose windows do not overlap.
    ///
    /// # Arguments
    /// * `kernel` — the window size, `(height, width)`; the stride matches it.
    pub fn new(kernel: (usize, usize)) -> Self {
        Self::with_stride(kernel, kernel)
    }

    /// Creates a pooling layer with a stride of your choice.
    ///
    /// # Arguments
    /// * `kernel` — the window size, `(height, width)`.
    /// * `stride` — the step between two windows.
    ///
    /// # Panics
    /// If a stride is zero.
    pub fn with_stride(kernel: (usize, usize), stride: (usize, usize)) -> Self {
        assert!(stride.0 > 0 && stride.1 > 0, "!!!AvgPool2d: stride must be positive!!!");
        Self { kernel, stride }
    }

    /// Returns the spatial size this layer turns `[height, width]` into.
    pub fn output_size(&self, height: usize, width: usize) -> (usize, usize) {
        pooled_size((height, width), self.kernel, self.stride)
    }
}

impl<T: Float> Module<T> for AvgPool2d {
    /// Reduces every window to its mean.
    ///
    /// # Arguments
    /// * `x` — the input, of shape `[batch, channels, height, width]`.
    ///
    /// # Returns
    /// `[batch, channels, out_height, out_width]`, see [AvgPool2d::output_size].
    ///
    /// # Panics
    /// If the input is not 4-D, or is smaller than one window.
    fn forward(&self, x: &VarRef<T>) -> VarRef<T> {
        let shape = pool_dims(x, "AvgPool2d");
        let (out_h, out_w) = self.output_size(shape[2], shape[3]);

        let cols = windows_per_channel(x, &shape, self.kernel, self.stride);

        let summed = sum_axis_op(&cols, 1, false);
        let pooled = &summed / T::from_usize(self.kernel.0 * self.kernel.1);

        reshape_op(&pooled, vec![shape[0], shape[1], out_h, out_w])
    }

    fn parameters(&self) -> Vec<VarRef<T>> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{autodiff::Var, linalg::Tensor};

    fn ramp(shape: Vec<usize>) -> Tensor<f64> {
        let n: usize = shape.iter().product();
        Tensor::new((1..=n).map(|v| v as f64).collect(), shape)
    }

    #[test]
    fn pools_each_channel_on_its_own() {
        // two channels, the second one is the first shifted by 100
        let mut data: Vec<f64> = (1..=16).map(|v| v as f64).collect();
        data.extend((1..=16).map(|v| v as f64 + 100.0));
        let x = Var::leaf(Tensor::new(data, vec![1, 2, 4, 4]), false);

        let y = MaxPool2d::new((2, 2)).forward(&x);

        assert_eq!(y.value().get_shape(), vec![1, 2, 2, 2]);
        assert_eq!(
            y.value().get_data(),
            vec![6.0, 8.0, 14.0, 16.0, 106.0, 108.0, 114.0, 116.0]
        );
    }

    #[test]
    fn batch_and_channels_survive_the_round_trip() {
        let x = Var::leaf(ramp(vec![3, 5, 8, 8]), false);

        let maxed = MaxPool2d::new((2, 2)).forward(&x);
        assert_eq!(maxed.value().get_shape(), vec![3, 5, 4, 4]);

        let avg = AvgPool2d::new((2, 2)).forward(&x);
        assert_eq!(avg.value().get_shape(), vec![3, 5, 4, 4]);

        // overlapping windows
        let overlap = MaxPool2d::with_stride((3, 3), (2, 2)).forward(&x);
        assert_eq!(overlap.value().get_shape(), vec![3, 5, 3, 3]);
    }

    #[test]
    fn max_routes_the_gradient_only_to_the_winner() {
        let x = Var::leaf(ramp(vec![1, 1, 2, 2]), true);

        // one 2x2 window over [[1, 2], [3, 4]]: only the 4 survives
        MaxPool2d::new((2, 2)).forward(&x).sum().backward();

        assert_eq!(x.grad().get_data(), vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn average_spreads_the_gradient_evenly() {
        let x = Var::leaf(ramp(vec![1, 1, 2, 2]), true);

        AvgPool2d::new((2, 2)).forward(&x).sum().backward();

        assert_eq!(x.grad().get_data(), vec![0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn overlapping_windows_accumulate_the_gradient() {
        // a 3-wide input with 2-wide windows at stride 1: the middle column is
        // read by both windows and has to collect from both
        let x = Var::leaf(ramp(vec![1, 1, 1, 3]), true);

        let y = MaxPool2d::with_stride((1, 2), (1, 1)).forward(&x);
        assert_eq!(y.value().get_data(), vec![2.0, 3.0]);

        y.sum().backward();
        // window 0 picks index 1, window 1 picks index 2
        assert_eq!(x.grad().get_data(), vec![0.0, 1.0, 1.0]);
    }
}
