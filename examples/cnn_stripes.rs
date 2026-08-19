//! A convolutional network telling vertical stripes from horizontal ones.
//!
//! The task is picked so that only spatial structure can solve it: both classes
//! share the same distribution of brightness and differ solely in the axis it
//! varies along. A dense layer over the flattened vector manages it too, but a
//! convolution needs no more than a handful of 3x3 filters.

use tensorrs::{
    activation::{Module, ReLU, Sigmoid},
    autodiff::{AutoGrad, Var},
    linalg::Tensor,
    loss::binary_cross_entropy,
    nn::{Conv2d, Flatten, Linear, MaxPool2d, Sequential},
    optim::{Adam, Optimizer},
};

const N_SAMPLES: usize = 128;
const SIZE: usize = 8;
const EPOCHS: usize = 200;

/// Half the sample is vertical stripes, half horizontal, both with noise.
fn make_data() -> (Tensor<f64>, Tensor<f64>) {
    let noise = Tensor::<f64>::randn(vec![N_SAMPLES * SIZE * SIZE]).get_data();

    let mut images = vec![0.0; N_SAMPLES * SIZE * SIZE];
    let mut labels = vec![0.0; N_SAMPLES];

    for n in 0..N_SAMPLES {
        let vertical = n % 2 == 0;
        labels[n] = if vertical { 1.0 } else { 0.0 };

        for y in 0..SIZE {
            for x in 0..SIZE {
                let stripe = if vertical { x } else { y };
                let value = if stripe % 2 == 0 { 1.0 } else { -1.0 };

                let idx = (n * SIZE + y) * SIZE + x;
                images[idx] = value + 0.3 * noise[idx];
            }
        }
    }

    (
        Tensor::new(images, vec![N_SAMPLES, 1, SIZE, SIZE]),
        Tensor::new(labels, vec![N_SAMPLES, 1]),
    )
}

fn main() {
    let (x_val, y_val) = make_data();

    let x = Var::leaf(x_val.shallow_copy(), false);
    let y = Var::leaf(y_val.shallow_copy(), false);

    //  [N,1,8,8] -> [N,4,8,8] -> [N,4,4,4] -> [N,8,4,4] -> [N,8,2,2] -> [N,32] -> [N,1]
    let model: Sequential<f64> = Sequential::new(vec![
        Box::new(Conv2d::same(1, 4, (3, 3), true)),
        Box::new(ReLU::new()),
        Box::new(MaxPool2d::new((2, 2))),
        Box::new(Conv2d::same(4, 8, (3, 3), true)),
        Box::new(ReLU::new()),
        Box::new(MaxPool2d::new((2, 2))),
        Box::new(Flatten::new()),
        Box::new(Linear::new(8 * 2 * 2, 1, true)),
        Box::new(Sigmoid::new()),
    ]);

    let mut optim = Adam::new(model.parameters(), 0.01);

    for epoch in 0..=EPOCHS {
        optim.zero_grad();

        let pred = model.forward(&x);
        let loss = binary_cross_entropy(&pred, &y);

        loss.backward();
        optim.step();

        if epoch % 25 == 0 {
            println!("Epoch {epoch:3}: loss = {:.6}", loss.value().item());
        }
    }

    let pred = model.forward(&x).value().get_data();
    let truth = y_val.get_data();

    let correct = pred
        .iter()
        .zip(truth.iter())
        .filter(|(p, t)| ((**p > 0.5) as u8 as f64) == **t)
        .count();

    println!(
        "\nAccuracy: {:.2}% ({correct}/{N_SAMPLES})",
        100.0 * correct as f64 / N_SAMPLES as f64
    );
}
