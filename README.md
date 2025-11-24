# Tensorrs<img alt="LOGO" height="25" src="./assets/tensorsLogo.svg" width="25"/>

**Tensorrs** is a lightweight machine learning library written in Rust.  
It provides a simple and efficient way to build and train neural networks with minimal dependencies.

## Alpha Notice

**Tensorrs is currently in alpha version.**  
The API is unstable — function names, argument types, and behaviors may change at any time.  
Use at your own risk and pin exact versions if needed.

## Dependencies

Tensorrs uses the following crates:

- [`rayon`](https://crates.io/crates/rayon) — for parallel CPU computations
- [`rand`](https://crates.io/crates/rand) — for random number generation
- [`serde`](https://crates.io/crates/serde) — for model serialization
- [`serde_json`](https://crates.io/crates/serde_json) — for model deserialization

## Installation

Add `tensorrs` to your project from [crates.io](https://crates.io/crates/tensorrs):

```toml
[dependencies]
tensorrs = "0.3.2"
```

## Example Usage
```rust
use tensorrs::activation::{Function, Sigmoid};
use tensorrs::{DataType, matrix};
use tensorrs::linalg::Matrix;
use tensorrs::loss::SSE;
use tensorrs::nn::{Linear, Sequential};
use tensorrs::optim::Adam;
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

If you'd like to contribute to Tensors, please follow these steps:

1. Fork the repository.

2. Create a new branch for your feature or bugfix.

3. Submit a pull request with a detailed description of your changes.

See [CONTRIBUTING](assets/CONTRIBUTING.md) for more details

## License

Tensors is licensed under the MIT License. See [LICENSE](assets/LICENSE) for more details.

KOT
```
  |\'/-..--.
 / _ _   ,  ;
`~=`Y'~_<._./
 <`-....__.'  
```
