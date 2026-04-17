use rand::{distributions::Standard, prelude::Distribution};

use crate::{Float, activation::Module, linalg::Tensor, autodiff::{Var, VarRef}};

#[derive(Clone, Debug)]
pub enum Initializer<T: Float> {
    Xavier,
    Zeros,
    He,
    LeCun,
    Custom(fn(usize, usize, bool) -> Tensor<T>),
    FromTensor(Tensor<T>),
}

pub struct Linear<T: Float> {
    pub weights: VarRef<T>,
    pub bias: Option<VarRef<T>>,
}

impl<T:Float> Linear<T>
where
    Standard: Distribution<T> {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        Self::with_initializer(in_features, out_features, bias, Initializer::Xavier)
    }


    pub fn with_initializer(in_features: usize,
        out_features: usize,
        bias: bool,
        initializer: Initializer<T>
    ) -> Self {
        let w_val = match initializer {
            Initializer::He => Self::he_init(in_features, out_features),
            Initializer::Xavier => Self::xavier_init(in_features, out_features),
            Initializer::LeCun => Self::lecun_init(in_features, out_features),
            Initializer::Zeros => Self::zeros_init(in_features, out_features),
            Initializer::Custom(func) => func(in_features, out_features, bias),
            Initializer::FromTensor(mx) => mx,
        };

        let weights = Var::leaf(w_val, true); 
        let bias = if bias {
            let b_val = Tensor::from_num(T::default(), vec![1, out_features]);
            Some(Var::leaf(b_val, true))
        } else {
            None
        };

        Self { weights, bias }
    }


    fn xavier_init(in_features: usize, out_features: usize) -> Tensor<T> {
        let shape = vec![in_features, out_features];
        let fan_in = T::from_usize(in_features);
        let fan_out = T::from_usize(out_features);
        
        // Xavier Uniform limit: sqrt(6 / (fan_in + fan_out))
        let limit = (T::from_usize(6) / (fan_in + fan_out)).sqrt();

        // (rand * 2 - 1) * limit
        (Tensor::rand(shape) * T::from_usize(2) - T::one()) * limit
    }

    fn he_init(in_features: usize, out_features: usize) -> Tensor<T> {
        let shape = vec![in_features, out_features];
        let fan_in = T::from_usize(in_features);
        
        // He Uniform limit: sqrt(6 / fan_in)
        let limit = (T::from_usize(6) / fan_in).sqrt();

        (Tensor::rand(shape) * T::from_usize(2) - T::one()) * limit
    }

    fn lecun_init(in_features: usize, out_features: usize) -> Tensor<T> {
        let shape = vec![in_features, out_features];
        let fan_in = T::from_usize(in_features);

        let limit = (T::from_usize(3)/ fan_in).sqrt();
        (Tensor::rand(shape) * T::from_usize(2) - T::one()) * limit
    }

    fn zeros_init(in_features: usize, out_features: usize) -> Tensor<T> {
        Tensor::from_num(T::default(), vec![in_features, out_features])
    }
}

impl<T: Float> Module<T> for Linear<T> {
    fn forward(&self, x: &VarRef<T>) -> VarRef<T> {
        let x_w = x * &self.weights;

        match &self.bias {
            Some(b) => &x_w + b,
            None => x_w,
        }
    }

    fn parameters(&self) -> Vec<VarRef<T>> {
        let mut params = vec![self.weights.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use crate::{nn::Sequential, tensor, autodiff::AutoGrad};

    use super::*;

    #[test]
    fn test_all_initializers() {
        let x = Var::leaf(tensor![[2.0, 1.0]], true);

        let in_f = 2;
        let mid_f = 3;
        let out_f = 1;

        let inits = vec![
            Initializer::Xavier,
            Initializer::He,
            Initializer::LeCun,
            Initializer::Zeros,
        ];

        for init in inits {
            println!("testing {:?}", init);

            let model = Sequential::new(vec![
                Box::new(Linear::<f32>::with_initializer(in_f,
                     mid_f, true, init.clone())),
                Box::new(Linear::<f32>::with_initializer(mid_f,
                     out_f, true, init.clone())),

            ]);

            let out = model.forward(&x);

            println!("{}", out.value().item());
        }
    }
}