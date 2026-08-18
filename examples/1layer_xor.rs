use tensorrs::{activation::{Module, PReLU}, loss::mse, nn::{Initializer, Linear, Sequential}, optim::{Adam, Optimizer}, tensor, autodiff::{AutoGrad, Var}};

fn main() {
    let x_val = tensor![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let y_val = tensor![[0.0], [1.0], [1.0], [0.0]];

    let x = Var::leaf(&(x_val * 2.0) - 1.0, false);
    let y = Var::leaf(y_val, false);

    let model = Sequential::new(vec![
            Box::new(Linear::<f32>::with_initializer(2, 1, true, Initializer::He)),
            Box::new(PReLU::new()),
    ]);

    let mut optim = Adam::new(model.parameters(), 0.05);
    for i in 0..300 {
        optim.zero_grad();
        let output = model.forward(&x);
        let loss = mse(&output, &y);

        let value = loss.value().item();
        if value.is_nan() || value < 0.01 {
            println!("Early stop {i}: {value}");
            break;
        }

        loss.backward();
        optim.clip_grad(5.0);
        optim.step();
    }
    for (i, param) in model.parameters().iter().enumerate() {
        println!("Параметр {}: {}", i, param);
    }

    println!("Final results: {}", model.forward(&x));
}