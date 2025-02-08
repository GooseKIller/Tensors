# Tensors
Tensors is a lightweight machine learning library in Rust. It provides a simple and efficient way to create and train machine learning models with minimal dependencies.

## Dependencies

The library uses the following dependencies:

- [rayon](https://crates.io/crates/rayon) - for parallel computations on CPU.
- [rand](https://crates.io/crates/rand) - for random number generation.
- [serde](https://crates.io/crates/serde) - for saving models.
- [serde_json](https://crates.io/crates/serde_json) - for loading models.

## Installation

Since the library is not yet published on crates.io, you can add it to your project by specifying the path to your local copy in your `Cargo.toml`:

```toml
[dependencies]
tensors = { path = "../path_to_tensors" }
```

## Example Usage
```rust
use tensors::activation::{Function, Sigmoid};
use tensors::{DataType, matrix};
use tensors::linalg::Matrix;
use tensors::loss::SSE;
use tensors::nn::{Linear, Sequential};
use tensors::optim::Adam;
// simple xor gate realization
fn main() {
    //input data
    let input = matrix![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    //output data
    let output = matrix![[0.0], [1.0], [1.0], [0.0]];

    // architecture of neural network
    let layers: Vec<Box<dyn Function<f32>>> = vec![
        Box::new(Linear::new(2, 2, true)),
        Box::new(Sigmoid::new()),
        Box::new(Linear::new(2, 1, true)),
        Box::new(Sigmoid::new())
    ];

    let mut optim = Adam::new(0.02f32, &layers);
    let mut model = Sequential::new(layers);
    let loss = SSE::new(DataType::f32());
    let mut loss_num = 100f32;

    println!("Initial output: {}", model.forward(input.clone()));

    for i in 0..10000 {
        if loss_num < 0.001 {
            println!("i: {} LOSS: {}", i, loss_num);
            break;
        }
        loss_num = model.train(
            input.clone(),
            output.clone(),
            &mut optim,
            &loss
        );
        if i % 1000 == 0 {
            println!("Loss at iteration {}: {}", i, loss_num);
        }
    }

    println!("Final output: {}", model.forward(input));
}
```


## Contributing

We welcome contributions from the community! If you'd like to contribute to Tensors, please follow these steps:

1. Fork the repository.

2. Create a new branch for your feature or bugfix.

3. Submit a pull request with a detailed description of your changes.

## License

Tensors is licensed under the MIT License. See [LICENSE](LICENSE) for more details.