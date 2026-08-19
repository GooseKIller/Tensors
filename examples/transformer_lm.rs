//! A transformer end to end: from token ids to a distribution over the vocabulary.
//!
//! The task is "look at the first token": at every position the model has to name
//! the token standing at the start of the sequence. Only attention solves it,
//! because the token in question sits arbitrarily far away while every position
//! carries a positional feature of its own.
//!
//! ```text
//! ids --> Embedding --> +Positional --> MHA --> +res --> LayerNorm
//!                                        \                  |
//!                                         \                FFN --> +res --> LayerNorm --> Linear --> softmax
//! ```

use tensorrs::{
    activation::{Module, ReLU, SoftMax},
    autodiff::{AutoGrad, Var, VarRef, reshape_op},
    linalg::Tensor,
    loss::cross_entropy,
    nn::{Embedding, LayerNorm, Linear, MultiHeadAttention, PositionalEncoding},
    optim::{Adam, Optimizer},
};

const VOCAB: usize = 12;
const BATCH: usize = 48;
const SEQ: usize = 5;
const D_MODEL: usize = 32;
const HEADS: usize = 4;
const D_FF: usize = 64;
const EPOCHS: usize = 400;

struct Block {
    attention: MultiHeadAttention<f64>,
    norm1: LayerNorm<f64>,
    ff1: Linear<f64>,
    act: ReLU,
    ff2: Linear<f64>,
    norm2: LayerNorm<f64>,
}

impl Block {
    fn new() -> Self {
        Self {
            attention: MultiHeadAttention::new(D_MODEL, HEADS, true),
            norm1: LayerNorm::new(D_MODEL, 1e-5),
            ff1: Linear::new(D_MODEL, D_FF, true),
            act: ReLU::new(),
            ff2: Linear::new(D_FF, D_MODEL, true),
            norm2: LayerNorm::new(D_MODEL, 1e-5),
        }
    }
}

impl Module<f64> for Block {
    fn forward(&self, x: &VarRef<f64>) -> VarRef<f64> {
        let attended = self.attention.forward(x);
        let x = self.norm1.forward(&(&attended + x));

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

/// Pseudorandom sequences, and the target: the first token, repeated throughout.
fn make_data() -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut tokens = Vec::with_capacity(BATCH);
    let mut targets = Vec::with_capacity(BATCH * SEQ);
    let mut state = 12345usize;

    for _ in 0..BATCH {
        let mut sequence = Vec::with_capacity(SEQ);
        for _ in 0..SEQ {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            sequence.push((state >> 16) % VOCAB);
        }

        let answer = sequence[0];
        targets.extend(std::iter::repeat(answer).take(SEQ));
        tokens.push(sequence);
    }

    (tokens, targets)
}

/// One-hot of the expected tokens, shaped `[BATCH * SEQ, VOCAB]`.
fn one_hot(targets: &[usize]) -> Tensor<f64> {
    let mut data = vec![0.0; targets.len() * VOCAB];
    for (row, &token) in targets.iter().enumerate() {
        data[row * VOCAB + token] = 1.0;
    }
    Tensor::new(data, vec![targets.len(), VOCAB])
}

fn accuracy(logits: &[f64], targets: &[usize]) -> f64 {
    let correct = targets
        .iter()
        .enumerate()
        .filter(|(row, &want)| {
            let start = row * VOCAB;
            let best = (0..VOCAB)
                .max_by(|&a, &b| logits[start + a].total_cmp(&logits[start + b]))
                .unwrap();
            best == want
        })
        .count();

    100.0 * correct as f64 / targets.len() as f64
}

fn main() {
    let (tokens, targets) = make_data();
    let y = Var::leaf(one_hot(&targets), false);

    let embedding = Embedding::<f64>::new(VOCAB, D_MODEL);
    let positions = PositionalEncoding::<f64>::new(SEQ, D_MODEL);
    let block = Block::new();
    let head = Linear::<f64>::new(D_MODEL, VOCAB, true);

    let mut params = embedding.parameters();
    params.extend(block.parameters());
    params.extend(head.parameters());
    println!("trainable tensors: {}", params.len());

    let mut optim = Adam::new(params, 0.005);

    let forward = |_: ()| -> VarRef<f64> {
        let x = positions.add_to(&embedding.forward(&tokens));
        let x = block.forward(&x);
        // [BATCH, SEQ, VOCAB] -> [BATCH * SEQ, VOCAB]
        let logits = reshape_op(&head.forward(&x), vec![BATCH * SEQ, VOCAB]);
        SoftMax::new().forward(&logits)
    };

    for epoch in 0..=EPOCHS {
        optim.zero_grad();

        let probs = forward(());
        let loss = cross_entropy(&probs, &y);

        loss.backward();
        optim.clip_grad(5.0);
        optim.step();

        if epoch % 50 == 0 {
            let acc = accuracy(&probs.value().get_data(), &targets);
            println!("Epoch {epoch:3}: loss = {:.5}  accuracy = {acc:.1}%", loss.value().item());
        }
    }

    let probs = forward(());
    println!(
        "\nFinal accuracy: {:.1}%",
        accuracy(&probs.value().get_data(), &targets)
    );

    println!("\n{:>22} {:>8} {:>10}", "sequence", "wanted", "predicted");
    for b in 0..5 {
        let row = b * SEQ + SEQ - 1; // the last position: the furthest from the answer
        let start = row * VOCAB;
        let best = (0..VOCAB)
            .max_by(|&a, &c| probs.value().get_data()[start + a]
                .total_cmp(&probs.value().get_data()[start + c]))
            .unwrap();

        println!("{:>22} {:>8} {:>10}", format!("{:?}", tokens[b]), tokens[b][0], best);
    }
}
