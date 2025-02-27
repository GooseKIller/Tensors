use crate::linalg::Vector;
use crate::Float;

mod autodiff;

pub use autodiff::*;


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
