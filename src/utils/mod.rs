use crate::linalg::Vector;
use crate::Float;

mod autodiff;

pub use autodiff::*;

/// Performs one-hot encoding on a vector of categorical indices.
///
/// Given a vector of indices (`data`), this function converts each index into a one-hot encoded vector
/// of the specified `size`. Each resulting vector has all elements set to zero except for the position
/// corresponding to the index, which is set to one.
///
/// # Parameters
/// - `data`: Input vector containing categorical indices (0-based).
/// - `size`: Dimension of the output one-hot vectors. Must be greater than the maximum index in `data`.
/// - `_`: Phantom parameter to infer the float type `T` (e.g., `f32` or `f64`). Value is ignored.
/// # Examples
/// ```
/// use tensorrs::linalg::Vector;
/// use tensorrs::utils::one_hot_encoding;
/// use tensorrs::vector;
///
/// let data = vec![2, 0, 3];
/// let encoded = one_hot_encoding(data, 4, 0.0f32);
///
/// assert_eq!(
///     encoded,
///     vec![
///         vector![0.0, 0.0, 1.0, 0.0],
///         vector![1.0, 0.0, 0.0, 0.0],
///         vector![0.0, 0.0, 0.0, 1.0]
///     ]
/// );
/// ```
pub fn one_hot_encoding<T: Float>(data: Vec<usize>, size: usize, _: T) -> Vec<Vector<T>> {
    assert!(
        *data.iter().max().unwrap_or(&0usize) < size,
        "!!!Size of the vector must be greater then max number:\
     Vector size: {size}, Max Element: {}!!!",
        data.iter().max().unwrap_or(&0usize)
    );
    let mut vectors = Vec::with_capacity(data.len());
    for i in data {
        let mut vector = vec![T::default(); size];
        vector[i] = T::one();
        vectors.push(Vector::from(vector));
    }
    vectors
}

pub fn one_hot_decoding<T: Float>(data: Vec<Vector<T>>) -> Vec<usize> {
    data.iter()
        .map(|x| {
            x.data
                .iter()
                .position(|&x| x == T::one())
                .unwrap_or_default()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::linalg::Matrix;
    use crate::utils::{one_hot_decoding, one_hot_encoding};
    use crate::DataType;

    #[test]
    fn test_one_hot() {
        let a = vec![1, 2, 3, 4];
        println!("{}", Matrix::from(one_hot_encoding(a, 5, DataType::f32())));
    }

    #[test]
    fn test_decoding() {
        let a = vec![1, 2, 3, 4];
        let one_hot = one_hot_encoding(a, 5, DataType::f32());
        let one_dec = one_hot_decoding(one_hot);

        println!("{:?}", one_dec);
    }
}
