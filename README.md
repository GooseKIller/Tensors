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
tensorrs = "0.3.3"
```

## Example Usage
```rust
use tensorrs::{
    activation::{Module, Sigmoid}, 
    loss::binary_cross_entropy, 
    nn::{Linear, Sequential}, 
    optim::{Adam, Optimizer}, 
    tensor, 
    utils::{AutoGrad, Var}
};
// simple xor gate realization
fn main() {
    let x_val = tensor![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let y_val = tensor![[0.0], [1.0], [1.0], [0.0]];

    let x = Var::leaf(x_val, true);
    let y = Var::leaf(y_val, true);

    // 2. Define the model architecture using the Sequential container
    // This stacks layers where the output of one flows into the next
    let model = Sequential::new(vec![
        Box::new(Linear::<f32>::new(2, 4, true)),
        Box::new(Sigmoid::new()),
        Box::new(Linear::new(4, 1, true)),
        Box::new(Sigmoid::new()),
    ]);

    let mut optim = Adam::new(model.parameters(), 0.1);
    for i in 0..1000 {
        optim.zero_grad();

        let y_pred = model.forward(&x);
        let loss = binary_cross_entropy(&y_pred, &y);

        loss.backward();

        optim.step();

        if i % 100 == 0{
            println!("{i}: {}", y_pred.value());
            println!("{i}: {}", loss.value());
        }

        if loss.value().item() < 0.01 {
            println!("Early exit");
            println!("{i}: {}", y_pred.value());
            println!("{i}: {}", loss.value());
            break;
        }
    }

    println!("Final output: {}", model.forward(&x));
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
