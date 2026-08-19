//! A transformer encoder block built out of the existing layers.
//!
//! The task is chosen so that only attention solves it: exactly one position in
//! the sequence carries a flag, and the payload sits at that position. The model
//! has to produce that payload at **every** position - that is, to learn to look
//! at the flagged token wherever it happens to stand.
//!
//! The layout is the one from the original paper, in its post-norm variant:
//!
//! ```text
//! x --> MHA ------> + --> LayerNorm --> FFN --> + --> LayerNorm
//!   \____________/  ^              \_________/  ^
//!    residual connection            residual connection
//! ```

use tensorrs::{
    activation::{Module, ReLU},
    autodiff::{AutoGrad, Var, VarRef},
    linalg::Tensor,
    loss::mse,
    nn::{LayerNorm, Linear, MultiHeadAttention},
    optim::{Adam, Optimizer},
};

const BATCH: usize = 32;
const SEQ: usize = 6;
const D_MODEL: usize = 24;
const HEADS: usize = 4;
const D_FF: usize = 48;
const EPOCHS: usize = 400;

struct TransformerBlock {
    attention: MultiHeadAttention<f64>,
    norm1: LayerNorm<f64>,
    ff1: Linear<f64>,
    act: ReLU,
    ff2: Linear<f64>,
    norm2: LayerNorm<f64>,
}

impl TransformerBlock {
    fn new(d_model: usize, heads: usize, d_ff: usize) -> Self {
        Self {
            attention: MultiHeadAttention::new(d_model, heads, true),
            norm1: LayerNorm::new(d_model, 1e-5),
            ff1: Linear::new(d_model, d_ff, true),
            act: ReLU::new(),
            ff2: Linear::new(d_ff, d_model, true),
            norm2: LayerNorm::new(d_model, 1e-5),
        }
    }
}

impl Module<f64> for TransformerBlock {
    fn forward(&self, x: &VarRef<f64>) -> VarRef<f64> {
        // attention, with its residual connection
        let attended = self.attention.forward(x);
        let x = self.norm1.forward(&(&attended + x));

        // the position-wise feed-forward, with a residual connection of its own
        let hidden = self.act.forward(&self.ff1.forward(&x));
        let ff = self.ff2.forward(&hidden);

        self.norm2.forward(&(&ff + &x))
    }

    fn parameters(&self) -> Vec<VarRef<f64>> {
        let mut params = self.attention.parameters();
        params.extend(self.norm1.parameters());
        params.extend(self.ff1.parameters());
        params.extend(self.ff2.parameters());
        params.extend(self.norm2.parameters());
        params
    }
}

/// Returns the input `[BATCH, SEQ, D_MODEL]` and a target of the same shape.
fn make_data() -> (Tensor<f64>, Tensor<f64>) {
    let mut x = vec![0.0; BATCH * SEQ * D_MODEL];
    let mut y = vec![0.0; BATCH * SEQ * D_MODEL];

    for b in 0..BATCH {
        let marked = b % SEQ;
        let payload = 0.4 + (b as f64) * 0.02;

        for s in 0..SEQ {
            let at = (b * SEQ + s) * D_MODEL;

            x[at] = (s == marked) as u8 as f64; // the flag
            x[at + 1] = if s == marked { payload } else { 0.0 }; // the payload
            x[at + 2] = (s as f64) * 0.1; // position, as a distracting feature

            y[at] = payload; // the answer is wanted at every position
        }
    }

    (
        Tensor::new(x, vec![BATCH, SEQ, D_MODEL]),
        Tensor::new(y, vec![BATCH, SEQ, D_MODEL]),
    )
}

fn main() {
    let (x_val, y_val) = make_data();
    let x = Var::leaf(x_val.shallow_copy(), false);
    let y = Var::leaf(y_val.shallow_copy(), false);

    let block = TransformerBlock::new(D_MODEL, HEADS, D_FF);
    let mut optim = Adam::new(block.parameters(), 0.005);

    println!("parameters: {}", block.parameters().len());

    for epoch in 0..=EPOCHS {
        optim.zero_grad();

        let loss = mse(&block.forward(&x), &y);

        loss.backward();
        optim.clip_grad(5.0);
        optim.step();

        if epoch % 50 == 0 {
            println!("Epoch {epoch:3}: loss = {:.6}", loss.value().item());
        }
    }

    // whether the model landed on the value it was asked for
    let pred = block.forward(&x).value().get_data();
    let truth = y_val.get_data();

    println!("\n{:>10} {:>10} {:>10}", "flagged", "predicted", "truth");
    for b in 0..5 {
        let at = (b * SEQ) * D_MODEL; // position zero of each sequence
        println!("{:>10} {:>10.4} {:>10.4}", b % SEQ, pred[at], truth[at]);
    }

    let mae: f64 = (0..BATCH * SEQ)
        .map(|i| (pred[i * D_MODEL] - truth[i * D_MODEL]).abs())
        .sum::<f64>()
        / (BATCH * SEQ) as f64;
    println!("\nMean absolute error: {mae:.4}");
}
