use rand::random;
use crate::Float;
use crate::nn::Linear;
use crate::linalg::Matrix;
use rand::prelude::Distribution;
use rand::distributions::Standard;

/// Test more safe Implementation for Linear
pub struct LinearBuilder<T: Float> {
    input_size: Option<usize>,
    output_size: Option<usize>,
    bias: bool,
    initializer: Initializer<T>
}

pub enum Initializer<T: Float> {
    Xavier,
    Zeros,
    He,
    Custom(fn(usize, usize, bool) -> Matrix<T>),
}

impl<T: Float> LinearBuilder<T>
where
    Standard: Distribution<T> {

    /// Creates a new linear_builder
    ///
    /// by default bias true and initializer as Xavier
    pub fn new() -> Self {
        Self {
            input_size: None,
            output_size: None,
            bias: true,
            initializer: Initializer::Xavier,
        }
    }

    pub fn input_size(mut self, size: usize) -> Self {
        self.input_size = Some(size);
        self
    }

    pub fn output_size(mut self, size: usize) -> Self {
        self.output_size = Some(size);
        self
    }

    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    pub fn with_initializer(mut self, initializer: Initializer<T>) -> Self {
        self.initializer = initializer;
        self
    }

    pub fn build(self) -> Result<Linear<T>, String> {
        let input_size = self.input_size.ok_or("!!!Input size must be specified!!!")?;
        let output_size = self.output_size.ok_or("!!!Output size must be specified!!!")?;

        let matrix = match self.initializer {
            Initializer::Xavier => self.xavier_init(input_size, output_size),
            Initializer::Zeros => self.zeros_init(input_size, output_size),
            Initializer::He => self.he_init(input_size, output_size),
            Initializer::Custom(func) => func(input_size, output_size, self.bias),
        };

        Ok(Linear{
            matrix,
            bias: self.bias,
        })
    }

    fn xavier_init(&self, input:usize, output: usize) -> Matrix<T> {
        let mut data = Vec::with_capacity(input * output);
        let limit = (T::from_usize(6) / T::from_usize(input + output)).sqrt();

        for _ in 0..(input * output) {
            let value = random::<T>() * T::from(2) * limit - limit;
            data.push(value);
        }

        if self.bias {
            data.extend(vec![T::default(); output]);
        }

        Matrix::new(
            data,
            input + if self.bias { 1 } else { 0 },
            output
        )
    }

    fn zeros_init(&self, input: usize, output: usize) -> Matrix<T> {
        let size = (input + if self.bias { 1 } else { 0 }) * output;
        Matrix::new(vec![T::default(); size],
                    input + if self.bias { 1 } else { 0 },// added bias
                    output)
    }

    fn he_init(&self, input: usize, output: usize) -> Matrix<T> {
        let mut data = Vec::with_capacity(input * output);
        let std = T::sqrt(T::from(2) / T::from_usize(input));

        for _ in 0..(input * output) {
            let value = random::<T>() * std * T::from(2) - std;
            data.push(value);
        }

        if self.bias {
            data.extend(vec![T::default(); output]);
        }

        Matrix::new(
            data,
            input + if self.bias { 1 } else { 0 },
            output
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Matrix;

    #[test]
    fn test_builder_basic() {
        let layer = LinearBuilder::<f32>::new()
            .input_size(2)
            .output_size(3)
            .build()
            .unwrap();

        assert_eq!(layer.matrix.rows(), 3); // 2 input + 1 bias
        assert_eq!(layer.matrix.cols(), 3);
        assert!(layer.bias);
    }

    #[test]
    fn test_builder_without_bias() {
        let layer = LinearBuilder::<f64>::new()
            .input_size(4)
            .output_size(2)
            .with_bias(false)
            .build()
            .unwrap();

        assert_eq!(layer.matrix.rows(), 4); // Только веса
        assert_eq!(layer.matrix.cols(), 2);
        assert!(!layer.bias);
    }

    #[test]
    fn test_custom_initializer_parallel() {
        let layer = LinearBuilder::<f32>::new()
            .input_size(2)
            .output_size(2)
            .with_initializer(Initializer::Custom(|input, output, has_bias| {
                let rows = input + has_bias as usize;
                Matrix::from_fn(rows, output, |i, j| (i * 10 + j) as f32)
            }))
            .build()
            .unwrap();

        // Ожидаемые значения:
        // Веса:  [0,1], [10,11]
        // Смещение: [20,21]
        assert_eq!(layer.matrix.data, vec![0., 1., 10., 11., 20., 21.]);
    }

    #[test]
    fn test_builder_validation() {
        let missing_input = LinearBuilder::<f32>::new()
            .output_size(3)
            .build();
        assert!(missing_input.is_err());

        let missing_output = LinearBuilder::<f32>::new()
            .input_size(2)
            .build();
        assert!(missing_output.is_err());
    }

    #[test]
    fn test_zero_initialization() {
        let layer = LinearBuilder::<f64>::new()
            .input_size(3)
            .output_size(2)
            .with_initializer(Initializer::Zeros)
            .build()
            .unwrap();

        assert!(layer.matrix.data.iter().all(|&x| x == 0.0));
        assert_eq!(layer.matrix.rows(), 4); // 3 input + 1 bias
    }

    #[test]
    fn test_he_initialization_range() {
        let layer = LinearBuilder::<f32>::new()
            .input_size(100)
            .output_size(50)
            .with_initializer(Initializer::He)
            .build()
            .unwrap();

        let expected_std = (2.0f32 / 100.0).sqrt();
        for &value in &layer.matrix.data {
            assert!(value >= -expected_std * 2.0 && value <= expected_std * 2.0,
                    "Value {} out of expected range", value);
        }
    }
}