use rand::{distributions::Standard, prelude::Distribution};

use crate::{
    Float,
    autodiff::{AutoGrad, Var, VarRef, gather_op, reshape_op},
    linalg::Tensor,
};

/// A lookup table turning token ids into vectors.
///
/// # Example
/// ```
/// use tensorrs::{nn::Embedding, autodiff::AutoGrad};
///
/// // 100 tokens, 16 features each
/// let embedding = Embedding::<f32>::new(100, 16);
///
/// // a batch of two sequences of three tokens
/// let x = embedding.forward(&[vec![7, 3, 0], vec![1, 1, 42]]);
/// assert_eq!(x.value().get_shape(), vec![2, 3, 16]);
/// ```
///
/// # Notes
/// This is the entry point of a language model: ids carry no arithmetic meaning of
/// their own, so the first thing to do with them is look up a vector that does.
///
/// It cannot implement [Module](crate::activation::Module), whose input is a node
/// of the graph — token ids are indices, not values, and nothing differentiates
/// with respect to them. The gradient stops here and lands in the table.
///
/// A token that occurs several times in a batch takes a gradient from every
/// occurrence, see [gather_op](crate::autodiff::gather_op).
///
/// # See Also
/// [PositionalEncoding](crate::nn::PositionalEncoding), which is what usually
/// comes next.
pub struct Embedding<T: Float> {
    /// The table, of shape `[vocab, dim]` — one row per token.
    pub weights: VarRef<T>,
    vocab: usize,
    dim: usize,
}

impl<T: Float> Embedding<T>
where
    Standard: Distribution<T>,
{
    /// Creates a table of `vocab` vectors of `dim` features.
    ///
    /// # Example
    /// ```
    /// use tensorrs::nn::Embedding;
    ///
    /// let embedding = Embedding::<f32>::new(1000, 64);
    /// assert_eq!(embedding.vocab(), 1000);
    /// assert_eq!(embedding.dim(), 64);
    /// ```
    ///
    /// # Arguments
    /// * `vocab` — how many distinct tokens there are.
    /// * `dim` — the width of one token vector.
    ///
    /// # Panics
    /// If either is zero.
    ///
    /// # Notes
    /// The rows start from $`N(0, 1/\sqrt{dim})`$, which puts a token vector at
    /// roughly unit length however wide it is — the same scaling the rest of the
    /// library uses for weights.
    pub fn new(vocab: usize, dim: usize) -> Self {
        assert!(vocab > 0 && dim > 0,
            "!!!Embedding: a table of {vocab}x{dim} has nothing in it!!!");

        let scale = T::one() / T::from_usize(dim).sqrt();
        let table = Tensor::randn(vec![vocab, dim]) * scale;

        Self {
            weights: Var::leaf(table, true),
            vocab,
            dim,
        }
    }
}

impl<T: Float> Embedding<T> {
    /// Returns how many distinct tokens the table holds.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Returns the width of one token vector.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Looks up a flat list of ids.
    ///
    /// # Example
    /// ```
    /// use tensorrs::{nn::Embedding, autodiff::AutoGrad};
    ///
    /// let embedding = Embedding::<f32>::new(10, 4);
    /// assert_eq!(embedding.lookup(&[3, 1, 4]).value().get_shape(), vec![3, 4]);
    /// ```
    ///
    /// # Arguments
    /// * `ids` — the token ids, in order.
    ///
    /// # Returns
    /// `[ids.len(), dim]`.
    ///
    /// # Panics
    /// If an id is not below [Embedding::vocab].
    pub fn lookup(&self, ids: &[usize]) -> VarRef<T> {
        for &id in ids {
            assert!(id < self.vocab,
                "!!!Embedding: token {id} is outside a vocabulary of {}!!!", self.vocab);
        }

        gather_op(&self.weights, ids)
    }

    /// Looks up a batch of sequences.
    ///
    /// # Example
    /// ```
    /// use tensorrs::{nn::Embedding, autodiff::AutoGrad};
    ///
    /// let embedding = Embedding::<f32>::new(50, 8);
    /// let x = embedding.forward(&[vec![1, 2], vec![3, 4], vec![5, 6]]);
    ///
    /// assert_eq!(x.value().get_shape(), vec![3, 2, 8]);
    /// ```
    ///
    /// # Arguments
    /// * `tokens` — one sequence of ids per batch entry; all of them the same length.
    ///
    /// # Returns
    /// `[batch, seq, dim]`, which is the shape every sequence layer here expects.
    ///
    /// # Panics
    /// If `tokens` is empty, the sequences differ in length, or an id is out of range.
    pub fn forward(&self, tokens: &[Vec<usize>]) -> VarRef<T> {
        assert!(!tokens.is_empty(), "!!!Embedding: an empty batch!!!");

        let seq = tokens[0].len();
        assert!(seq > 0, "!!!Embedding: an empty sequence!!!");

        for (i, sequence) in tokens.iter().enumerate() {
            assert_eq!(sequence.len(), seq,
                "!!!Embedding: sequence {i} is {} long, expected {seq}!!!", sequence.len());
        }

        let flat: Vec<usize> = tokens.iter().flatten().copied().collect();
        let looked_up = self.lookup(&flat);

        reshape_op(&looked_up, vec![tokens.len(), seq, self.dim])
    }

    /// Returns the table, which is the only thing there is to train.
    pub fn parameters(&self) -> Vec<VarRef<T>> {
        vec![self.weights.clone()]
    }
}

/// The fixed sinusoidal positions of the original transformer.
///
/// # Formula
///```math
///  PE_{(pos,\, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right) \qquad
///  PE_{(pos,\, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
///```
///
/// # Example
/// ```
/// use tensorrs::{nn::{Embedding, PositionalEncoding}, autodiff::{AutoGrad, Var}};
///
/// let embedding = Embedding::<f32>::new(50, 16);
/// let positions = PositionalEncoding::<f32>::new(64, 16);
///
/// let x = embedding.forward(&[vec![1, 2, 3], vec![4, 5, 6]]);
/// let x = positions.add_to(&x);
///
/// assert_eq!(x.value().get_shape(), vec![2, 3, 16]);
/// ```
///
/// # Notes
/// Attention has no notion of order — permute the tokens and the scores permute
/// with them. Adding a position-dependent pattern to the vectors is what lets the
/// model tell "the cat sat" from "sat the cat".
///
/// These are constants rather than parameters, so they cost nothing to train and
/// extend to sequences longer than any seen during training.
///
/// # See Also
/// [Attention Is All You Need](https://arxiv.org/abs/1706.03762), section 3.5
pub struct PositionalEncoding<T: Float> {
    table: Tensor<T>,
    dim: usize,
}

impl<T: Float> PositionalEncoding<T> {
    /// Builds the encodings for up to `max_len` positions.
    ///
    /// # Example
    /// ```
    /// use tensorrs::nn::PositionalEncoding;
    ///
    /// let positions = PositionalEncoding::<f64>::new(128, 32);
    ///
    /// // position 0 is sin(0), cos(0), sin(0), cos(0), ...
    /// let row = positions.row(0);
    /// assert_eq!(row[0], 0.0);
    /// assert_eq!(row[1], 1.0);
    /// ```
    ///
    /// # Arguments
    /// * `max_len` — the longest sequence to support.
    /// * `dim` — the width of the model; it has to match the embedding.
    ///
    /// # Panics
    /// If either is zero.
    pub fn new(max_len: usize, dim: usize) -> Self {
        assert!(max_len > 0 && dim > 0,
            "!!!PositionalEncoding: {max_len}x{dim} has nothing in it!!!");

        let base = T::from_f64(10000.0);
        let dim_t = T::from_usize(dim);
        let mut data = vec![T::default(); max_len * dim];

        for pos in 0..max_len {
            let position = T::from_usize(pos);

            for i in 0..dim {
                // both members of a pair share a wavelength, one sine, one cosine
                let pair = T::from_usize(i - i % 2);
                let wavelength = base.powf(pair / dim_t);
                let angle = position / wavelength;

                data[pos * dim + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        Self { table: Tensor::new(data, vec![max_len, dim]), dim }
    }

    /// Returns the encoding of one position.
    ///
    /// # Panics
    /// If `pos` is past the length this was built for.
    pub fn row(&self, pos: usize) -> Vec<T> {
        let max_len = self.table.get_shape()[0];
        assert!(pos < max_len,
            "!!!PositionalEncoding: position {pos} is past a table of {max_len}!!!");

        self.table
            .slice(&[pos, 0], &[1, self.dim])
            .expect("positional row out of bounds (bug)")
            .get_data()
    }

    /// Adds the encodings to a batch of token vectors.
    ///
    /// # Arguments
    /// * `x` — `[batch, seq, dim]`, straight out of an [Embedding].
    ///
    /// # Returns
    /// A tensor of the same shape.
    ///
    /// # Panics
    /// If `x` is not 3-D, is wider than this was built for, or is longer than
    /// `max_len`.
    ///
    /// # Notes
    /// The encodings enter the graph as a constant, so no gradient flows into them.
    pub fn add_to(&self, x: &VarRef<T>) -> VarRef<T> {
        let shape = x.value().get_shape();

        assert_eq!(shape.len(), 3,
            "!!!PositionalEncoding expects [batch, seq, dim], got {shape:?}!!!");
        assert_eq!(shape[2], self.dim,
            "!!!PositionalEncoding was built for a width of {}, got {}!!!", self.dim, shape[2]);

        let max_len = self.table.get_shape()[0];
        assert!(shape[1] <= max_len,
            "!!!PositionalEncoding was built for {max_len} positions, got {}!!!", shape[1]);

        // [seq, dim] broadcasts over the batch on its own
        let slice = self
            .table
            .slice(&[0, 0], &[shape[1], self.dim])
            .expect("positional slice out of bounds (bug)");
        let positions = Var::leaf(Tensor::new(slice.get_data(), vec![1, shape[1], self.dim]), false);

        x + &positions
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
    fn a_batch_of_sequences_becomes_a_3d_tensor() {
        let embedding = Embedding::<f64>::new(20, 8);
        let x = embedding.forward(&[vec![1, 2, 3], vec![4, 5, 6]]);

        assert_eq!(x.value().get_shape(), vec![2, 3, 8]);
    }

    #[test]
    fn the_same_token_always_gives_the_same_vector() {
        let embedding = Embedding::<f64>::new(10, 4);

        let x = embedding.forward(&[vec![7, 0, 7]]).value().get_data();
        assert_eq!(x[..4], x[8..12]);

        // and a different one does not
        assert_ne!(x[..4], x[4..8]);
    }

    #[test]
    fn a_repeated_token_collects_from_every_occurrence() {
        let embedding = Embedding::<f64>::new(4, 2);

        // token 1 appears three times, token 0 once, tokens 2 and 3 never
        let out = embedding.forward(&[vec![1, 1], vec![1, 0]]);
        out.sum().backward();

        let grad = embedding.weights.grad().get_data();
        assert_eq!(grad[0..2], [1.0, 1.0]); // token 0
        assert_eq!(grad[2..4], [3.0, 3.0]); // token 1
        assert_eq!(grad[4..8], [0.0, 0.0, 0.0, 0.0]); // untouched tokens
    }

    #[test]
    fn positions_differ_and_repeat_across_the_batch() {
        let embedding = Embedding::<f64>::new(5, 8);
        let positions = PositionalEncoding::<f64>::new(16, 8);

        // the same token at two positions must come out different
        let x = positions.add_to(&embedding.forward(&[vec![2, 2]]));
        let data = x.value().get_data();
        assert_ne!(data[..8], data[8..16]);

        // but the same position in two batch entries must come out the same
        let x = positions.add_to(&embedding.forward(&[vec![2, 3], vec![2, 4]]));
        let data = x.value().get_data();
        assert_eq!(data[..8], data[16..24]);
    }

    #[test]
    fn the_first_position_is_sines_and_cosines_of_zero() {
        let positions = PositionalEncoding::<f64>::new(4, 6);
        let row = positions.row(0);

        for (i, v) in row.iter().enumerate() {
            let expected = if i % 2 == 0 { 0.0 } else { 1.0 };
            assert!((v - expected).abs() < 1e-12, "position 0, feature {i}: {v}");
        }
    }

    #[test]
    fn learns_a_vector_per_token() {
        // every token has to land on its own target vector, which is only
        // possible if the table itself trains
        let (vocab, dim) = (6usize, 4usize);
        let embedding = Embedding::<f64>::new(vocab, dim);

        let tokens: Vec<Vec<usize>> = (0..vocab).map(|t| vec![t]).collect();
        let targets: Vec<f64> = (0..vocab)
            .flat_map(|t| (0..dim).map(move |d| (t as f64) * 0.1 + (d as f64) * 0.01))
            .collect();
        let y = Var::leaf(Tensor::new(targets, vec![vocab, 1, dim]), false);

        let mut optim = Adam::new(embedding.parameters(), 0.1);
        let first = mse(&embedding.forward(&tokens), &y).value().item();

        for _ in 0..300 {
            optim.zero_grad();
            let loss = mse(&embedding.forward(&tokens), &y);
            loss.backward();
            optim.step();
        }

        let last = mse(&embedding.forward(&tokens), &y).value().item();
        assert!(last < first * 1e-4, "the table barely moved: {first} -> {last}");
    }
}
