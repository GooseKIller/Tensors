use tensorrs::{
    activation::{Module, Sigmoid}, 
    loss::binary_cross_entropy, 
    nn::{Linear, Sequential}, 
    optim::{Adam, Optimizer}, 
    tensor, 
    autodiff::{AutoGrad, Var}
};

fn main() {
    // 1. Prepare training data for the XOR problem
    // The tensor! macro creates multidimensional matrices with ease
    let x_val = tensor![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let y_val = tensor![[0.0], [1.0], [1.0], [0.0]];

    // Wrap tensors in Var to enable tracking in the computational graph
    // Setting leaf to true allows gradient tracking for these variables
    let x = Var::leaf(x_val, false);
    let y = Var::leaf(y_val, false);

    // 2. Define the model architecture using the Sequential container
    // This stacks layers where the output of one flows into the next
    let model = Sequential::new(vec![
        Box::new(Linear::<f32>::new(2, 4, true)), // Layer 1: 2 inputs -> 4 neurons
        Box::new(Sigmoid::new()),                   // Activation: Scaled Exponential Linear Unit
        Box::new(Linear::new(4, 1, true)),       // Layer 2: 4 neurons -> 1 output
        Box::new(Sigmoid::new()),                 // Final activation to squash output to [0, 1]
    ]);

    // 3. Initialize the Adam optimizer
    // model.parameters() recursively collects all weight and bias references
    let mut optim = Adam::new(model.parameters(), 0.1);
    
    for i in 0..1000 {
        // Reset gradients to prevent accumulation from previous iterations
        optim.zero_grad();

        // Forward pass: compute the predicted output
        let y_pred = model.forward(&x);

        // Calculate loss using Binary Cross Entropy
        let loss = binary_cross_entropy(&y_pred, &y);

        // Backward pass: the "magic" of Autograd computes all gradients
        loss.backward();

        // Update model parameters based on computed gradients
        optim.step();

        if loss.value().item() < 0.01 {
            println!("Early exit");
            println!("{i}: {}", y_pred.value());
            println!("{i}: {}", loss.value());
            break;
        }
    }
}