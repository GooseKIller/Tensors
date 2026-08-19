//! Teaching a recurrent network to add up a sequence.
//!
//! The input is a sequence of `SEQ` random numbers and the expected output is
//! their sum. Only memory solves this: the answer cannot be recovered from the
//! last step alone, so the state has to accumulate information.

use tensorrs::{
    activation::Module,
    autodiff::{AutoGrad, Var},
    linalg::Tensor,
    loss::mse,
    nn::{Linear, RNN},
    optim::{Adam, Optimizer},
};

const N_SAMPLES: usize = 256;
const SEQ: usize = 8;
const HIDDEN: usize = 24;
const EPOCHS: usize = 800;

fn main() {
    // [N, SEQ, 1] random numbers drawn from U(0, 1)
    let x_val: Tensor<f64> = Tensor::rand(vec![N_SAMPLES, SEQ, 1]);
    // the target is the sum over time, [N, 1]
    let y_val = x_val.sum_axis(1);

    let x = Var::leaf(x_val.shallow_copy(), false);
    let y = Var::leaf(y_val.shallow_copy(), false);

    let rnn = RNN::<f64>::new(1, HIDDEN, true);
    let head = Linear::<f64>::new(HIDDEN, 1, true);

    let mut params = rnn.parameters();
    params.extend(head.parameters());
    let mut optim = Adam::new(params, 0.005);

    for epoch in 0..=EPOCHS {
        optim.zero_grad();

        let pred = head.forward(&rnn.forward(&x));
        let loss = mse(&pred, &y);

        loss.backward();
        // The classic remedy for recurrent networks. The derivative of tanh is
        // zero in saturation, so an unlucky start can drive the state onto a
        // plateau and stall training; clipping keeps the steps proportionate and
        // makes the run robust to initialisation. The threshold is 5 rather than
        // 1, since 1 clips almost every step and makes learning ragged.
        optim.clip_grad(5.0);
        optim.step();

        if epoch % 100 == 0 {
            println!("Epoch {epoch:3}: loss = {:.6}", loss.value().item());
        }
    }

    // a look at the first five predictions
    let pred = head.forward(&rnn.forward(&x)).value().get_data();
    let truth = y_val.get_data();

    println!("\n{:>10} {:>10} {:>10}", "predicted", "truth", "error");
    for i in 0..5 {
        println!(
            "{:>10.4} {:>10.4} {:>10.4}",
            pred[i],
            truth[i],
            (pred[i] - truth[i]).abs()
        );
    }

    let mae: f64 = pred
        .iter()
        .zip(truth.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>()
        / N_SAMPLES as f64;
    println!("\nMean absolute error over the whole sample: {mae:.4}");
}
