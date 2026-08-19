use rand::{distributions::Standard, prelude::Distribution};

use crate::{
    Float,
    activation::Module,
    autodiff::{AutoGrad, Var, VarRef, permute_op, reshape_op, unfold_op},
    linalg::Tensor,
};

/// A 2-D convolution layer.
///
/// # Formula
///```math
///  y_{n,o,i,j} = b_o + \sum_{c=1}^{C_{in}} \sum_{u=1}^{k_h} \sum_{v=1}^{k_w}
///      x_{n,c,\, i s_h + u - p_h,\; j s_w + v - p_w} \cdot w_{o,c,u,v}
///```
/// Where $`s`$ is the stride, $`p`$ the padding and $`b_o`$ the bias of output
/// channel $`o`$
///
/// # Example
/// ```
/// use tensorrs::{activation::{Module, ReLU}, linalg::Tensor,
///                nn::{Conv2d, Sequential}, autodiff::{AutoGrad, Var}};
///
/// // 3 input channels -> 8 feature maps, keeping the spatial size
/// let model: Sequential<f32> = Sequential::new(vec![
///     Box::new(Conv2d::same(3, 8, (3, 3), true)),
///     Box::new(ReLU::new()),
/// ]);
///
/// // a batch of 2 images, 3 channels, 16x16
/// let x = Var::leaf(Tensor::<f32>::randn(vec![2, 3, 16, 16]), false);
///
/// let y = model.forward(&x);
/// assert_eq!(y.value().get_shape(), vec![2, 8, 16, 16]);
/// ```
///
/// # Notes
/// The convolution is computed as an `im2col` transform followed by one matrix
/// multiplication: [unfold_op](crate::autodiff::unfold_op) lays every window out
/// as a row, and multiplying by the weight matrix convolves all of them with all
/// filters at once. That reuses the parallel [Tensor::matmul], and the gradient
/// comes from the existing matmul and unfold rules rather than a rule of its own.
///
/// The cost is memory: the unfolded input holds `kernel_h * kernel_w` times as
/// many values as the input, so a `3x3` kernel needs about nine times the space
/// for the duration of the forward pass.
///
/// # See Also
/// [Wikipedia: Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network)
pub struct Conv2d<T: Float> {
    /// Filters, of shape `[in_channels * kernel_h * kernel_w, out_channels]` —
    /// laid out so that the unfolded input multiplies straight into them.
    pub weights: VarRef<T>,
    /// Bias, of shape `[1, out_channels]`.
    pub bias: Option<VarRef<T>>,
    in_channels: usize,
    out_channels: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl<T: Float> Conv2d<T>
where
    Standard: Distribution<T>,
{
    /// Creates a convolution with stride `1` and no padding.
    ///
    /// # Example
    /// ```
    /// use tensorrs::{activation::Module, nn::Conv2d};
    ///
    /// let conv = Conv2d::<f32>::new(1, 16, (5, 5), true);
    /// assert_eq!(conv.parameters().len(), 2); // weights and bias
    /// ```
    ///
    /// # Arguments
    /// * `in_channels` — the number of channels the input carries.
    /// * `out_channels` — how many filters to learn, one feature map each.
    /// * `kernel` — the filter size, `(height, width)`.
    /// * `bias` — whether to add a bias per output channel.
    ///
    /// # Notes
    /// Without padding each convolution shrinks the image by `kernel - 1`, so a
    /// stack of them runs out of pixels. [Conv2d::same] keeps the size instead.
    pub fn new(in_channels: usize, out_channels: usize, kernel: (usize, usize), bias: bool) -> Self {
        Self::with_stride_padding(in_channels, out_channels, kernel, (1, 1), (0, 0), bias)
    }

    /// Creates a convolution whose padding keeps the spatial size, at stride `1`.
    ///
    /// # Example
    /// ```
    /// use tensorrs::{activation::Module, linalg::Tensor, nn::Conv2d,
    ///                autodiff::{AutoGrad, Var}};
    ///
    /// let conv = Conv2d::<f32>::same(1, 4, (3, 3), true);
    /// let x = Var::leaf(Tensor::<f32>::randn(vec![1, 1, 8, 8]), false);
    ///
    /// assert_eq!(conv.forward(&x).value().get_shape(), vec![1, 4, 8, 8]);
    /// ```
    ///
    /// # Arguments
    /// * `in_channels`, `out_channels`, `kernel`, `bias` — as in [Conv2d::new].
    ///
    /// # Panics
    /// If either side of the kernel is even — only an odd kernel has a centre to
    /// pad symmetrically around.
    pub fn same(in_channels: usize, out_channels: usize, kernel: (usize, usize), bias: bool) -> Self {
        assert!(kernel.0 % 2 == 1 && kernel.1 % 2 == 1,
            "!!!Conv2d::same() needs an odd kernel, got {kernel:?}!!!");

        Self::with_stride_padding(
            in_channels,
            out_channels,
            kernel,
            (1, 1),
            (kernel.0 / 2, kernel.1 / 2),
            bias,
        )
    }

    /// Creates a convolution with a stride and padding of your choice.
    ///
    /// # Example
    /// ```
    /// use tensorrs::{activation::Module, linalg::Tensor, nn::Conv2d,
    ///                autodiff::{AutoGrad, Var}};
    ///
    /// // stride 2 halves the image
    /// let conv = Conv2d::<f32>::with_stride_padding(3, 6, (3, 3), (2, 2), (1, 1), true);
    /// let x = Var::leaf(Tensor::<f32>::randn(vec![1, 3, 16, 16]), false);
    ///
    /// assert_eq!(conv.forward(&x).value().get_shape(), vec![1, 6, 8, 8]);
    /// ```
    ///
    /// # Arguments
    /// * `in_channels`, `out_channels`, `kernel`, `bias` — as in [Conv2d::new].
    /// * `stride` — the step between two windows, `(height, width)`.
    /// * `padding` — how many zero rows and columns to add on each side.
    ///
    /// # Panics
    /// If a stride is zero.
    ///
    /// # Notes
    /// The weights start from $`U(-k, k)`$ with $`k = 1/\sqrt{\text{fan\_in}}`$,
    /// where fan-in is `in_channels * kernel_h * kernel_w` — the number of values
    /// each output pixel sums over.
    pub fn with_stride_padding(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
    ) -> Self {
        assert!(stride.0 > 0 && stride.1 > 0, "!!!Conv2d: stride must be positive!!!");

        let fan_in = in_channels * kernel.0 * kernel.1;
        let limit = T::one() / T::from_usize(fan_in).sqrt();

        let w_val = (Tensor::rand(vec![fan_in, out_channels]) * T::from_usize(2) - T::one()) * limit;

        Self {
            weights: Var::leaf(w_val, true),
            bias: if bias {
                Some(Var::leaf(
                    Tensor::from_num(T::default(), vec![1, out_channels]),
                    true,
                ))
            } else {
                None
            },
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
        }
    }
}

impl<T: Float> Conv2d<T> {
    /// Returns the spatial size this layer turns `[height, width]` into.
    ///
    /// # Example
    /// ```
    /// use tensorrs::nn::Conv2d;
    ///
    /// let conv = Conv2d::<f32>::with_stride_padding(1, 1, (3, 3), (2, 2), (1, 1), false);
    /// assert_eq!(conv.output_size(16, 16), (8, 8));
    /// ```
    ///
    /// # Returns
    /// `(size + 2 * padding - kernel) / stride + 1` for each axis.
    pub fn output_size(&self, height: usize, width: usize) -> (usize, usize) {
        (
            (height + 2 * self.padding.0 - self.kernel.0) / self.stride.0 + 1,
            (width + 2 * self.padding.1 - self.kernel.1) / self.stride.1 + 1,
        )
    }
}

impl<T: Float> Module<T> for Conv2d<T> {
    /// Convolves the input with every filter.
    ///
    /// # Arguments
    /// * `x` — the input, of shape `[batch, in_channels, height, width]`.
    ///
    /// # Returns
    /// `[batch, out_channels, out_height, out_width]`, see [Conv2d::output_size].
    ///
    /// # Panics
    /// If the input is not 4-D, or carries a different number of channels than
    /// the layer was built for.
    fn forward(&self, x: &VarRef<T>) -> VarRef<T> {
        let shape = x.value().get_shape();
        assert_eq!(shape.len(), 4,
            "!!!Conv2d expects [batch, channels, height, width], got {shape:?}!!!");
        assert_eq!(shape[1], self.in_channels,
            "!!!Conv2d was built for {} input channels, got {}!!!", self.in_channels, shape[1]);

        let batch = shape[0];
        let (out_h, out_w) = self.output_size(shape[2], shape[3]);

        // every window becomes a row: [batch * out_h * out_w, in_channels * kh * kw]
        let cols = unfold_op(x, self.kernel, self.stride, self.padding);

        // one matmul convolves all windows with all filters
        let out = &cols * &self.weights;
        let out = match &self.bias {
            Some(b) => &out + b,
            None => out,
        };

        // rows come back as [batch, out_h, out_w, out_channels]; the rest of the
        // library speaks channels-first, so the channel axis moves up front
        let out = reshape_op(&out, vec![batch, out_h, out_w, self.out_channels]);
        permute_op(&out, &[0, 3, 1, 2])
    }

    fn parameters(&self) -> Vec<VarRef<T>> {
        let mut params = vec![self.weights.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        loss::mse,
        optim::{Adam, Optimizer},
    };

    #[test]
    fn shapes_follow_stride_and_padding() {
        let x = Var::leaf(Tensor::<f32>::randn(vec![2, 3, 16, 16]), false);

        let valid = Conv2d::<f32>::new(3, 8, (3, 3), true);
        assert_eq!(valid.forward(&x).value().get_shape(), vec![2, 8, 14, 14]);

        let same = Conv2d::<f32>::same(3, 8, (3, 3), true);
        assert_eq!(same.forward(&x).value().get_shape(), vec![2, 8, 16, 16]);

        let strided = Conv2d::<f32>::with_stride_padding(3, 8, (3, 3), (2, 2), (1, 1), true);
        assert_eq!(strided.forward(&x).value().get_shape(), vec![2, 8, 8, 8]);
    }

    #[test]
    fn matches_a_convolution_computed_by_hand() {
        // one input channel, one 2x2 filter, no bias: the result has to agree
        // with the sums worked out directly
        let conv = Conv2d::<f64>::new(1, 1, (2, 2), false);

        // pin the weights: unfold lays a window out as [c][i][j], so the filter
        // is [w00, w01, w10, w11]
        conv.weights.0.borrow_mut().value = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);

        let x = Var::leaf(
            Tensor::new((1..=9).map(|v| v as f64).collect(), vec![1, 1, 3, 3]),
            false,
        );

        // windows: [1,2,4,5] [2,3,5,6] [4,5,7,8] [5,6,8,9]
        // dotted with [1,2,3,4]: 37, 47, 67, 77
        let out = conv.forward(&x);
        assert_eq!(out.value().get_shape(), vec![1, 1, 2, 2]);
        assert_eq!(out.value().get_data(), vec![37.0, 47.0, 67.0, 77.0]);
    }

    #[test]
    fn gradients_reach_the_filters_and_the_bias() {
        let conv = Conv2d::<f64>::same(2, 3, (3, 3), true);
        let x = Var::leaf(Tensor::<f64>::randn(vec![2, 2, 5, 5]), false);
        let y = Var::leaf(Tensor::from_num(0.5, vec![2, 3, 5, 5]), false);

        let loss = mse(&conv.forward(&x), &y);
        loss.backward();

        for (i, p) in conv.parameters().iter().enumerate() {
            let g = p.grad().get_data();
            assert_eq!(g.len(), p.value().get_data().len(),
                "parameter {i} got a gradient of the wrong size");
            assert!(g.iter().any(|&v| v != 0.0),
                "parameter {i} never received a gradient");
        }
    }

    #[test]
    fn learns_an_edge_detector() {
        // the target is the input convolved with a fixed vertical edge filter,
        // so a single 3x3 layer can reproduce it exactly
        let x_val = Tensor::<f64>::randn(vec![4, 1, 8, 8]);

        let edge = Tensor::new(
            vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0],
            vec![9, 1],
        );
        let target = Tensor::new(
            x_val
                .unfold_2d((3, 3), (1, 1), (1, 1))
                .matmul(&edge)
                .get_data(),
            vec![4, 1, 8, 8],
        );

        let x = Var::leaf(x_val, false);
        let y = Var::leaf(target, false);

        let conv = Conv2d::<f64>::same(1, 1, (3, 3), false);
        let mut optim = Adam::new(conv.parameters(), 0.05);

        let first = mse(&conv.forward(&x), &y).value().item();

        for _ in 0..300 {
            optim.zero_grad();
            let loss = mse(&conv.forward(&x), &y);
            loss.backward();
            optim.step();
        }

        let last = mse(&conv.forward(&x), &y).value().item();
        assert!(last < first * 1e-4, "the loss barely moved: {first} -> {last}");
    }
}
