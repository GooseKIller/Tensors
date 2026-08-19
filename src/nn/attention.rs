use rand::{distributions::Standard, prelude::Distribution};

use crate::{
    Float,
    activation::{Module, SoftMax},
    autodiff::{AutoGrad, VarRef, permute_op, reshape_op},
    linalg::Tensor,
    nn::Linear,
};

/// Builds an additive mask that hides every position after the current one.
///
/// # Example
/// ```
/// use tensorrs::{linalg::Tensor, nn::causal_mask};
///
/// let mask: Tensor<f32> = causal_mask(3);
/// assert_eq!(mask.get_shape(), vec![1, 1, 3, 3]);
///
/// // the first row may only look at itself
/// assert_eq!(mask.get_data()[..3], [0.0, -1e9, -1e9]);
/// ```
///
/// # Arguments
/// * `len` — the length of the sequence.
///
/// # Returns
/// A tensor of shape `[1, 1, len, len]`, zero where a position may be attended to
/// and a large negative number where it may not. The leading ones broadcast over
/// batch and heads.
///
/// # Notes
/// The mask is *added* to the scores before the softmax, so a hidden position
/// comes out of the exponential as zero. A large negative number is used rather
/// than an actual infinity, which would turn into a `NaN` the moment a whole row
/// is masked.
///
/// This is what makes a decoder autoregressive: without it a token could read the
/// answer straight off the positions it is supposed to predict.
pub fn causal_mask<T: Float>(len: usize) -> Tensor<T> {
    let blocked = T::from_f64(-1e9);
    let mut data = vec![T::default(); len * len];

    for query in 0..len {
        for key in 0..len {
            if key > query {
                data[query * len + key] = blocked;
            }
        }
    }

    Tensor::new(data, vec![1, 1, len, len])
}

/// Multi-head attention.
///
/// # Formula
///```math
///  \text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_h}}\right) V
///```
///```math
///  \text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H)\, W_O,
///  \qquad \text{head}_i = \text{Attention}(X W_Q^i,\, X W_K^i,\, X W_V^i)
///```
/// Where $`d_h`$ is the size of one head and $`H`$ their number
///
/// # Example
/// ```
/// use tensorrs::{activation::Module, linalg::Tensor, nn::MultiHeadAttention,
///                autodiff::{AutoGrad, Var}};
///
/// // 32 features split across 4 heads of 8
/// let attention = MultiHeadAttention::<f32>::new(32, 4, true);
///
/// // a batch of 2 sequences, 6 tokens each
/// let x = Var::leaf(Tensor::<f32>::randn(vec![2, 6, 32]), false);
///
/// // self-attention keeps the shape of its input
/// assert_eq!(attention.forward(&x).value().get_shape(), vec![2, 6, 32]);
/// ```
///
/// # Notes
/// The input is always `[batch, seq, d_model]`. [Module::forward] performs
/// *self*-attention, where the queries, keys and values all come from the same
/// sequence; [MultiHeadAttention::attend] takes the three separately, which is what
/// cross-attention and masking need.
///
/// Splitting into heads costs nothing but a reshape and a permute: the projections
/// stay one matrix each, and the heads become a batch dimension that the batched
/// [Tensor::matmul] already understands.
///
/// Scores grow with $`d_h`$, which is why they are divided by $`\sqrt{d_h}`$ —
/// without it the softmax saturates and the gradient vanishes before training
/// starts.
///
/// # See Also
/// [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
pub struct MultiHeadAttention<T: Float> {
    /// Query projection, `[d_model, d_model]`.
    pub w_q: Linear<T>,
    /// Key projection, `[d_model, d_model]`.
    pub w_k: Linear<T>,
    /// Value projection, `[d_model, d_model]`.
    pub w_v: Linear<T>,
    /// Output projection applied to the concatenated heads.
    pub w_o: Linear<T>,
    heads: usize,
    d_model: usize,
    d_head: usize,
}

impl<T: Float> MultiHeadAttention<T>
where
    Standard: Distribution<T>,
{
    /// Creates a multi-head attention block.
    ///
    /// # Example
    /// ```
    /// use tensorrs::{activation::Module, nn::MultiHeadAttention};
    ///
    /// let attention = MultiHeadAttention::<f32>::new(64, 8, true);
    /// assert_eq!(attention.d_head(), 8);
    /// assert_eq!(attention.parameters().len(), 8); // four projections, weights and bias each
    /// ```
    ///
    /// # Arguments
    /// * `d_model` — the width of the model.
    /// * `heads` — how many attention heads to split it across.
    /// * `bias` — whether the four projections carry a bias.
    ///
    /// # Panics
    /// If `heads` is zero or does not divide `d_model` — every head has to get the
    /// same slice of the features.
    pub fn new(d_model: usize, heads: usize, bias: bool) -> Self {
        assert!(heads > 0, "!!!MultiHeadAttention: needs at least one head!!!");
        assert_eq!(d_model % heads, 0,
            "!!!MultiHeadAttention: {heads} heads do not divide a width of {d_model}!!!");

        Self {
            w_q: Linear::new(d_model, d_model, bias),
            w_k: Linear::new(d_model, d_model, bias),
            w_v: Linear::new(d_model, d_model, bias),
            w_o: Linear::new(d_model, d_model, bias),
            heads,
            d_model,
            d_head: d_model / heads,
        }
    }
}

impl<T: Float> MultiHeadAttention<T> {
    /// Returns the number of heads.
    pub fn heads(&self) -> usize {
        self.heads
    }

    /// Returns the size of one head.
    pub fn d_head(&self) -> usize {
        self.d_head
    }

    /// Returns the width of the model.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Attends from `query` over `key` and `value`.
    ///
    /// # Example
    /// ```
    /// use tensorrs::{linalg::Tensor, nn::{MultiHeadAttention, causal_mask},
    ///                autodiff::{AutoGrad, Var}};
    ///
    /// let attention = MultiHeadAttention::<f32>::new(16, 2, true);
    /// let x = Var::leaf(Tensor::<f32>::randn(vec![1, 5, 16]), false);
    ///
    /// // masked self-attention, as a decoder uses it
    /// let mask = Var::leaf(causal_mask::<f32>(5), false);
    /// let out = attention.attend(&x, &x, &x, Some(&mask));
    ///
    /// assert_eq!(out.value().get_shape(), vec![1, 5, 16]);
    /// ```
    ///
    /// # Arguments
    /// * `query` — `[batch, seq_q, d_model]`, the sequence doing the looking.
    /// * `key`, `value` — `[batch, seq_kv, d_model]`, the sequence being looked at.
    ///   For self-attention all three are the same tensor.
    /// * `mask` — added to the scores before the softmax, broadcast to
    ///   `[batch, heads, seq_q, seq_kv]`. See [causal_mask].
    ///
    /// # Returns
    /// `[batch, seq_q, d_model]`.
    ///
    /// # Panics
    /// If any input is not 3-D, carries a width other than `d_model`, or if the
    /// key and value sequences differ in length.
    pub fn attend(
        &self,
        query: &VarRef<T>,
        key: &VarRef<T>,
        value: &VarRef<T>,
        mask: Option<&VarRef<T>>,
    ) -> VarRef<T> {
        let q_shape = self.check(query, "query");
        let k_shape = self.check(key, "key");
        let v_shape = self.check(value, "value");

        assert_eq!(k_shape[1], v_shape[1],
            "!!!MultiHeadAttention: key has {} positions but value has {}!!!",
            k_shape[1], v_shape[1]);

        let (batch, seq_q, seq_kv) = (q_shape[0], q_shape[1], k_shape[1]);

        // [batch, seq, d_model] -> [batch, heads, seq, d_head]
        let q = self.split_heads(&self.w_q.forward(query), batch, seq_q);
        let k = self.split_heads(&self.w_k.forward(key), batch, seq_kv);
        let v = self.split_heads(&self.w_v.forward(value), batch, seq_kv);

        // scores = Q K^T / sqrt(d_head), of shape [batch, heads, seq_q, seq_kv]
        let k_t = permute_op(&k, &[0, 1, 3, 2]);
        let scores = &(&q * &k_t) / T::from_usize(self.d_head).sqrt();

        let scores = match mask {
            Some(m) => &scores + m,
            None => scores,
        };

        // softmax over the keys, then a weighted sum of the values
        let weights = SoftMax::new().forward(&scores);
        let attended = &weights * &v;

        // [batch, heads, seq_q, d_head] -> [batch, seq_q, d_model]
        let merged = permute_op(&attended, &[0, 2, 1, 3]);
        let merged = reshape_op(&merged, vec![batch, seq_q, self.d_model]);

        self.w_o.forward(&merged)
    }

    /// Splits the feature axis across heads and moves the heads up front, so that
    /// they become a batch dimension of the matrix multiplications.
    fn split_heads(&self, x: &VarRef<T>, batch: usize, seq: usize) -> VarRef<T> {
        let per_head = reshape_op(x, vec![batch, seq, self.heads, self.d_head]);
        permute_op(&per_head, &[0, 2, 1, 3])
    }

    /// Checks one input and returns its shape.
    fn check(&self, x: &VarRef<T>, name: &str) -> Vec<usize> {
        let shape = x.value().get_shape();

        assert_eq!(shape.len(), 3,
            "!!!MultiHeadAttention expects {name} of shape [batch, seq, d_model], got {shape:?}!!!");
        assert_eq!(shape[2], self.d_model,
            "!!!MultiHeadAttention was built for a width of {}, but {name} is {}!!!",
            self.d_model, shape[2]);

        shape
    }
}

impl<T: Float> Module<T> for MultiHeadAttention<T> {
    /// Self-attention: the queries, keys and values all come from `x`.
    ///
    /// # Arguments
    /// * `x` — `[batch, seq, d_model]`.
    ///
    /// # Returns
    /// A tensor of the same shape.
    ///
    /// # Notes
    /// Unmasked, so every position sees every other one. For a decoder use
    /// [MultiHeadAttention::attend] with a [causal_mask].
    fn forward(&self, x: &VarRef<T>) -> VarRef<T> {
        self.attend(x, x, x, None)
    }

    fn parameters(&self) -> Vec<VarRef<T>> {
        let mut params = self.w_q.parameters();
        params.extend(self.w_k.parameters());
        params.extend(self.w_v.parameters());
        params.extend(self.w_o.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        autodiff::Var,
        loss::mse,
        optim::{Adam, Optimizer},
    };

    #[test]
    fn self_attention_keeps_the_shape() {
        let attention = MultiHeadAttention::<f32>::new(32, 4, true);
        let x = Var::leaf(Tensor::<f32>::randn(vec![3, 7, 32]), false);

        assert_eq!(attention.forward(&x).value().get_shape(), vec![3, 7, 32]);
    }

    #[test]
    fn cross_attention_follows_the_query_length() {
        let attention = MultiHeadAttention::<f32>::new(16, 2, true);

        let q = Var::leaf(Tensor::<f32>::randn(vec![2, 3, 16]), false);
        let kv = Var::leaf(Tensor::<f32>::randn(vec![2, 9, 16]), false);

        // the output has one row per query, however long the other sequence is
        assert_eq!(attention.attend(&q, &kv, &kv, None).value().get_shape(), vec![2, 3, 16]);
    }

    #[test]
    fn the_causal_mask_hides_the_future() {
        // one head, identity-ish setup: with a causal mask the first output row
        // can only depend on the first input row
        let attention = MultiHeadAttention::<f64>::new(8, 1, false);

        let mut data: Vec<f64> = (0..4 * 8).map(|i| (i as f64) * 0.1).collect();
        let x = Var::leaf(Tensor::new(data.clone(), vec![1, 4, 8]), false);
        let mask = Var::leaf(causal_mask::<f64>(4), false);

        let first = attention.attend(&x, &x, &x, Some(&mask)).value().get_data()[..8].to_vec();

        // change the last position only
        for v in data.iter_mut().skip(3 * 8) {
            *v += 5.0;
        }
        let x2 = Var::leaf(Tensor::new(data, vec![1, 4, 8]), false);
        let second = attention.attend(&x2, &x2, &x2, Some(&mask)).value().get_data()[..8].to_vec();

        for (a, b) in first.iter().zip(second.iter()) {
            assert!((a - b).abs() < 1e-12,
                "the first position saw a change in the last one: {a} vs {b}");
        }
    }

    #[test]
    fn without_a_mask_the_future_does_leak() {
        // the counterpart of the test above: this is what the mask is preventing
        let attention = MultiHeadAttention::<f64>::new(8, 1, false);

        let mut data: Vec<f64> = (0..4 * 8).map(|i| (i as f64) * 0.1).collect();
        let x = Var::leaf(Tensor::new(data.clone(), vec![1, 4, 8]), false);
        let first = attention.forward(&x).value().get_data()[..8].to_vec();

        for v in data.iter_mut().skip(3 * 8) {
            *v += 5.0;
        }
        let x2 = Var::leaf(Tensor::new(data, vec![1, 4, 8]), false);
        let second = attention.forward(&x2).value().get_data()[..8].to_vec();

        let moved = first.iter().zip(second.iter()).any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(moved, "unmasked attention should let the last position reach the first");
    }

    #[test]
    fn gradients_reach_every_projection() {
        let attention = MultiHeadAttention::<f64>::new(16, 4, true);
        let x = Var::leaf(Tensor::<f64>::randn(vec![2, 5, 16]), false);
        let y = Var::leaf(Tensor::from_num(0.1, vec![2, 5, 16]), false);

        let loss = mse(&attention.forward(&x), &y);
        loss.backward();

        for (i, p) in attention.parameters().iter().enumerate() {
            let g = p.grad().get_data();
            assert_eq!(g.len(), p.value().get_data().len(),
                "parameter {i} got a gradient of the wrong size");
            assert!(g.iter().any(|&v| v != 0.0), "parameter {i} never received a gradient");
        }
    }

    #[test]
    fn learns_to_copy_a_marked_token() {
        // Each sequence carries one "payload" position, flagged by a marker
        // feature. The target is that payload, repeated at every position - so the
        // block only solves it by learning to attend to the flagged token.
        let d_model = 16;
        let attention = MultiHeadAttention::<f64>::new(d_model, 2, true);

        let (batch, seq) = (16usize, 4usize);
        let mut x_data = vec![0.0f64; batch * seq * d_model];
        let mut y_data = vec![0.0f64; batch * seq * d_model];

        for b in 0..batch {
            let marked = b % seq;
            let payload = 0.5 + (b as f64) * 0.03;

            for s in 0..seq {
                let at = (b * seq + s) * d_model;
                x_data[at] = if s == marked { 1.0 } else { 0.0 }; // the flag
                x_data[at + 1] = payload * (s == marked) as u8 as f64;
                x_data[at + 2] = (s as f64) * 0.1; // position, as a distractor

                y_data[at] = payload;
            }
        }

        let x = Var::leaf(Tensor::new(x_data, vec![batch, seq, d_model]), false);
        let y = Var::leaf(Tensor::new(y_data, vec![batch, seq, d_model]), false);

        let mut optim = Adam::new(attention.parameters(), 0.02);
        let first = mse(&attention.forward(&x), &y).value().item();

        for _ in 0..300 {
            optim.zero_grad();
            let loss = mse(&attention.forward(&x), &y);
            loss.backward();
            optim.step();
        }

        let last = mse(&attention.forward(&x), &y).value().item();
        assert!(last < first * 0.2, "the loss barely moved: {first} -> {last}");
    }
}
