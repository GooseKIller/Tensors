use tensorrs::{
    activation::{Module, PReLU, Sigmoid},
    autodiff::{AutoGrad, Var, VarRef},
    linalg::Tensor,
    loss::binary_cross_entropy,
    nn::{LayerNorm, Linear, Initializer},
    optim::{Adam, Optimizer}
};

struct ResBlock {
    ly: Linear<f64>,
    bn: LayerNorm<f64>,
    act: PReLU<f64>,
}

impl ResBlock {
    fn new(size: usize) -> Self {
        Self {
            ly: Linear::with_initializer(size, size, true, Initializer::He),
            bn: LayerNorm::new(size, 1e-5),
            act: PReLU::new(),
        }
    }

    fn forward(&self, x: &VarRef<f64>) -> VarRef<f64> {
        let mut out = self.ly.forward(x);
        out = self.bn.forward(&out);
        out = self.act.forward(&out);
        &out + x // Residual connection
    }

    fn parameters(&self) -> Vec<VarRef<f64>> {
        let mut p = self.ly.parameters();
        p.extend(self.bn.parameters());
        p.extend(self.act.parameters());
        p
    }
}

struct ResNetBN {
    input_ly: Linear<f64>,
    blocks: Vec<ResBlock>,
    output_ly: Linear<f64>,
    final_act: Sigmoid,
}

impl ResNetBN {
    fn new(in_features: usize, hidden: usize, num_blocks: usize) -> Self {
        let mut blocks = Vec::new();
        for _ in 0..num_blocks {
            blocks.push(ResBlock::new(hidden));
        }

        Self {
            input_ly: Linear::with_initializer(in_features, hidden, true, Initializer::He),
            blocks,
            output_ly: Linear::with_initializer(hidden, 1, true, Initializer::Xavier),
            final_act: Sigmoid::new(),
        }
    }
}

impl Module<f64> for ResNetBN {
    fn forward(&self, x: &VarRef<f64>) -> VarRef<f64> {
        let mut out = self.input_ly.forward(x);
        
        for block in &self.blocks {
            out = block.forward(&out);
        }

        out = self.output_ly.forward(&out);
        self.final_act.forward(&out)
    }

    fn parameters(&self) -> Vec<VarRef<f64>> {
        let mut p = self.input_ly.parameters();
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p.extend(self.output_ly.parameters());
        p
    }
}

fn main() {
    let n_samples = 1000;
    let features = 20;
    let batch_size = 32;

    // 1. Генерируем данные
    let x_val = Tensor::randn(vec![n_samples, features]);
    let y_val = x_val.sum_axis_keepdim(1).map(|x| if x > 0. { 1. } else { 0. });

    let model = ResNetBN::new(features, 16, 3);
    let mut optim = Adam::new(model.parameters(), 0.001); // Консервативный LR

    for epoch in 0..50 {
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        // Создаем индексы для перемешивания
        //let mut indices: Vec<usize> = (0..n_samples).collect();
        // Тут можно добавить shuffle(indices), если есть рандом

        for i in (0..n_samples).step_by(batch_size) {
            let current_batch_size = (i + batch_size).min(n_samples) - i;
            if current_batch_size == 0 { break; }

            // Используем твой метод slice
            // start_indices: [откуда берем по строкам, откуда по столбцам]
            // shape: [сколько берем строк, сколько столбцов]
            let x_batch_val = x_val.slice(&[i, 0], &[current_batch_size, features]).unwrap();
            let y_batch_val = y_val.slice(&[i, 0], &[current_batch_size, 1]).unwrap();

            // ВАЖНО: оборачиваем в листья без градиентов
            let x_batch = Var::leaf(x_batch_val, false);
            let y_batch = Var::leaf(y_batch_val, false);

            optim.zero_grad();
            
            let y_pred = model.forward(&x_batch);
            let loss = binary_cross_entropy(&y_pred, &y_batch);

            if loss.value().item().is_nan() {
                println!("NaN detected at epoch {}, batch {}", epoch, i);
                return;
            }

            loss.backward();
            optim.clip_grad(1.0);
            optim.step();

            epoch_loss += loss.value().item();
            num_batches += 1;
        }

        if epoch % 5 == 0 {
            println!("Epoch {}: Avg Loss = {:.6}", epoch, epoch_loss / num_batches as f64);
        }
    }
}
