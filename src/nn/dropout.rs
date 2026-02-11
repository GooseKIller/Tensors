use crate::{Float, activation::Module, linalg::Tensor, utils::{AutoGrad, Var}};

pub struct Dropout {
    p: f32, // Вероятность отключения (например, 0.5)
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Self { p }
    }
}

impl<T: Float> Module<T> for Dropout {
    fn forward(&self, x: &crate::utils::VarRef<T>) -> crate::utils::VarRef<T> {
        // Во время обучения: генерируем маску из 0 и 1
        // Во время теста (inference): просто возвращаем x * (1-p)
        // Но для твоего движка пока сделаем упрощенную версию для обучения:
        
        let x_val = x.value();
        let mut rng = rand::thread_rng();
        
        // Генерируем маску той же формы, что и x
        let mask_data: Vec<T> = x_val.packed_data().iter().map(|_| {
            if rand::Rng::gen_bool(&mut rng, (1.0 - self.p) as f64) {
                T::one() / (T::from_f64(1.0 - (self.p as f64))) // Масштабируем, чтобы сохранить мат. ожидание
            } else {
                T::default()
            }
        }).collect();
        
        let mask = Var::leaf(Tensor::new(mask_data, x_val.get_shape().clone()), false);
        
        x & &mask
    }

    fn parameters(&self) -> Vec<crate::utils::VarRef<T>> {
        vec![]
    }
}