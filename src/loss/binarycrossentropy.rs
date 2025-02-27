use crate::linalg::Matrix;
use crate::loss::Loss;
use crate::Float;

pub struct BCE<T: Float>(T);

impl<T: Float> BCE<T> {
    pub fn new(_data_type: T) -> Self {
        Self(_data_type)
    }
}

impl<T: Float> Loss<T> for BCE<T> {
    fn call(&self, output: &Matrix<T>, target: &Matrix<T>) -> T {
        if output.size() != target.size() {
            panic!("!!!Size of output matrix and target must be equal!!!\nOutput size:{:?} Target size: {:?}", output.size(), target.size())
        }
        let n = output.data.len();
        if n == 0 {
            return T::default();
        }

        let epsilon = T::f32_f64(1e-7, 1e-15); // Small value to avoid log(0)
        let output_clamped = output.max(epsilon).min(T::one() - epsilon);

        let a = target & &output_clamped.ln();
        let b = target.map(|x| T::one() - x) & &output_clamped.map(|z| T::one() - z).ln();
        let loss = -(a + &b);
        loss.sum() / T::from_usize(n)
    }

    fn gradient(&self, output: &Matrix<T>, target: &Matrix<T>) -> Matrix<T> {
        let grads = output.zip_with(target, |y_pred, y_true| {
            y_true / y_pred - (T::one() - y_true) / (T::one() - y_pred)
        });
        let n = grads.data.len();
        grads * (T::one() / T::from_usize(n))
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::Matrix;
    use crate::loss::{Loss, BCE};
    use crate::{matrix, DataType};

    #[test]
    fn call() {
        let a = matrix![[1.0, 0.0]];
        let b = matrix![[0.5, 0.5]];

        let bce = BCE::new(DataType::f64());
        println!("{}", bce.call(&b, &a));
    }

    #[test]
    fn grad() {
        let a = matrix![[1.0, 0.0]];
        let b = matrix![[0.5, 0.5]];

        let bce = BCE::new(DataType::f64());
        println!("{}", bce.gradient(&b, &a));
    }

    #[test]
    fn help() {
        let tar = matrix![[1.0, 0.0, 1.0, 0.0]];
        let out = matrix![[0.9, 0.1, 0.8, 0.2]];
        let bce = BCE::new(DataType::f32());
        println!("{}", bce.call(&out, &tar));

        println!("{}", bce.gradient(&out, &tar));
    }
}
