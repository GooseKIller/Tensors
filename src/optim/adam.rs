use crate::{Float, linalg::Tensor, optim::Optimizer, autodiff::{AutoGrad, VarRef}};

/// Adam optimizer — adaptive moment estimation.
///
/// Keeps a running estimate of the first and second moment of every gradient and
/// scales each parameter's step by them, so that rarely updated parameters move
/// further than frequently updated ones.
///
/// # Formula
///```math
///  m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \qquad
///  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
///```
///```math
///  \hat{m}_t = \frac{m_t}{1 - \beta_1^t} \qquad
///  \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
///```
///```math
///  w_t = w_{t-1} - \frac{\alpha \hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
///```
/// Where $`g_t`$ is the gradient at step $`t`$, $`m_t`$ the first moment,
/// $`v_t`$ the second moment and $`\alpha`$ the learning rate
///
/// # Example
/// ```
/// use tensorrs::{tensor, activation::Module, nn::{Initializer, Linear, Sequential},
///                loss::mse, optim::{Adam, Optimizer},
///                autodiff::{AutoGrad, Var}};
///
/// let model: Sequential<f32> = Sequential::new(vec![
///     Box::new(Linear::with_initializer(1, 1, true, Initializer::Zeros)),
/// ]);
/// let mut optim = Adam::new(model.parameters(), 0.1);
///
/// let x = Var::leaf(tensor![[1.0f32], [2.0], [3.0]], false);
/// let y = Var::leaf(tensor![[3.0f32], [5.0], [7.0]], false); // y = 2x + 1
///
/// for _ in 0..300 {
///     optim.zero_grad();
///     let loss = mse(&model.forward(&x), &y);
///     loss.backward();
///     optim.step();
/// }
///
/// // starts at 27.7 and lands within a rounding error of the true line
/// let final_loss = mse(&model.forward(&x), &y).value().item();
/// assert!(final_loss < 0.01);
/// ```
///
/// # Notes
/// The moments start at zero, which biases them towards zero on the first steps —
/// that is what the $`\hat{m}`$ / $`\hat{v}`$ correction undoes.
///
/// The defaults are $`\beta_1 = 0.9`$, $`\beta_2 = 0.999`$ and
/// $`\varepsilon = 10^{-8}`$.
///
/// # See Also
/// [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
pub struct Adam<T:Float> {
    params: Vec<VarRef<T>>,
    lr: T,
    m: Vec<Tensor<T>>, // Первый момент (среднее)
    v: Vec<Tensor<T>>, // Второй момент (нецентрированная дисперсия)
    t: usize,          // Счетчик шагов
    beta1: T,
    beta2: T,
    eps: T,
}

impl<T: Float> Adam<T> {
    /// Creates an Adam optimizer over the given parameters.
    ///
    /// # Example
    /// ```
    /// use tensorrs::{activation::Module, nn::{Linear, Sequential}, optim::Adam};
    ///
    /// let model: Sequential<f32> = Sequential::new(vec![
    ///     Box::new(Linear::new(2, 1, true)),
    /// ]);
    /// let optim = Adam::new(model.parameters(), 0.01);
    /// ```
    ///
    /// # Arguments
    /// * `params` — the parameters to train, usually `model.parameters()`.
    /// * `lr` — the learning rate $`\alpha`$.
    ///
    /// # Notes
    /// A zeroed first and second moment is allocated per parameter, so the
    /// optimizer holds two extra tensors for every trainable tensor.
    pub fn new(params: Vec<VarRef<T>>, lr: T) -> Self {
        let mut m = Vec::new();
        let mut v = Vec::new();

        for p in &params {
            let shape = p.borrow().value.shape.clone();
            m.push(Tensor::from_num(T::default(), shape.clone()));
            v.push(Tensor::from_num(T::default(), shape));
        }

        Self {
            params,
            lr,
            m,
            v,
            t: 0,
            beta1: T::from_f64(0.9),
            beta2: T::from_f64(0.999),
            eps: T::from_f64(1e-8),
        }
    }
}

impl<T: Float> Optimizer<T> for Adam<T> {
    /// Performs one optimization step over every parameter.
    ///
    /// # Notes
    /// Call it after `backward()`, once the gradients are filled in. The internal
    /// step counter $`t`$ is advanced here, which drives the bias correction.
    fn step(&mut self) {
        self.t += 1;
        let t = T::from_usize(self.t);

        for (i, param) in self.params.iter().enumerate() {
            let mut p = param.borrow_mut();
            let grad = p.grad.borrow().shallow_copy();

            // 1. Обновляем моменты: m = beta1 * m + (1 - beta1) * grad
            self.m[i] = &(&self.m[i] * self.beta1) + &(&grad * (T::one() - self.beta1));

            // 2. Обновляем дисперсию: v = beta2 * v + (1 - beta2) * grad^2
            let grad_sq = &grad & &grad;
            self.v[i] = &(&self.v[i] * self.beta2) + &(&grad_sq * (T::one() - self.beta2));

            // 3. Корректировка смещения (Bias correction)
            let m_hat = &self.m[i] / (T::one() - (self.beta1.powf(t.clone())));
            let v_hat = &self.v[i] / (T::one() - (self.beta2.powf(t)));

            // 4. Обновление весов: w = w - lr * m_hat / (sqrt(v_hat) + eps)
            let v_sqrt = v_hat.map(|x| x.sqrt() + self.eps);
            let update = &(&m_hat * self.lr) / &v_sqrt;

            p.value = &p.value - &update;
        }
    }

    /// Resets the gradient of every parameter to zero.
    ///
    /// # Notes
    /// Gradients accumulate, so this has to be called before every backward pass.
    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
    
    /// Returns the parameters this optimizer was built with.
    fn params(&self) -> Vec<crate::autodiff::VarRef<T>> {
        self.params.clone()
    }
}
