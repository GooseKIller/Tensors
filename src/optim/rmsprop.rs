use crate::{Float, linalg::Tensor, optim::Optimizer, autodiff::{AutoGrad, VarRef}};

pub struct RMSprop<T: Float> {
    params: Vec<VarRef<T>>,
    lr: T,
    v: Vec<Tensor<T>>, // the running average of the squared gradients
    alpha: T,          // the smoothing coefficient (usually 0.99)
    eps: T,            // a small number, for stability
}

impl<T: Float> RMSprop<T> {
    pub fn new(params: Vec<VarRef<T>>, lr: T) -> Self {
        let mut v = Vec::new();
        for p in &params {
            let shape = p.borrow().value.shape.clone();
            v.push(Tensor::from_num(T::default(), shape));
        }

        Self {
            params,
            lr,
            v,
            alpha: T::from_f64(0.99),
            eps: T::from_f64(1e-8),
        }
    }
}

impl<T: Float> Optimizer<T> for RMSprop<T> {
    fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            let mut p = param.borrow_mut();
            let grad = p.grad.borrow().shallow_copy();

            // 1. Update v: v = alpha * v + (1 - alpha) * grad^2
            let grad_sq = &grad & &grad;
            self.v[i] = &(&self.v[i] * self.alpha) + &(&grad_sq * (T::one() - self.alpha));

            // 2. Work out the update: lr * grad / (sqrt(v) + eps)
            let v_sqrt = self.v[i].map(|x| x.sqrt() + self.eps);
            let update = &(&grad * self.lr) / &v_sqrt;

            // 3. Update the weights
            p.value = &p.value - &update;
        }
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
    
    fn params(&self) -> Vec<crate::autodiff::VarRef<T>> {
        self.params.clone()
    }
}