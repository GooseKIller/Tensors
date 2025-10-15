//!
//! # Tensors
//!
//! Tensors is a lightweight machine learning library in Rust. It provides a simple and efficient way to create and train machine learning models with minimal dependencies.
//! ## Dependencies
//! The library uses the following dependencies:
//! - [rayon](https://crates.io/crates/rayon) - for parallel computations on CPU.
//! - [rand](https://crates.io/crates/rand) - for random number generation.
//! - [serde](https://crates.io/crates/serde) - for saving models.
//! - [serde_json](https://crates.io/crates/serde_json) - for loading models.
//!
//! ## Example
//! ```rust
//! use tensorrs::activation::Function;
//! use tensorrs::DataType;
//! use tensorrs::linalg::{Matrix, Vector};
//! use tensorrs::nn::{Linear, Sequential};
//! use tensorrs::optim::Adam;
//! use tensorrs::loss::MSE;
//! use tensorrs::loss::Loss;
//!
//! let x = Matrix::from(Vector::range(-1.0, 1.0, 0.125).unwrap());
//! let y:Matrix<f32> = 8.0 * &x - 10.0;
//!
//! let layers: Vec<Box< dyn Function<f32>>> = vec![Box::new(Linear::new(1, 1, true))];
//! let mut optim = Adam::new(0.001, &layers);
//! let mut model = Sequential::new(layers);
//! let loss = MSE::new(DataType::f32());
//!
//! for _ in 0..1000 {
//!     model.train(x.transpose(), y.transpose(), &mut optim, &loss);
//! }
//! ```
//! Thanks for using Tensors!!!
use std::cmp::PartialOrd;
use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

pub mod activation;
pub mod linalg;
pub mod loss;
pub mod nn;
pub mod optim;
pub mod utils;
//pub(crate) mod onnx_pb;

/// Numeric type
///
/// Special Trait
///
/// For most of the numbers like (i16, i32, i64, i128, f32, f64)
pub trait Num:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Div
    + AddAssign
    + SubAssign
    + Neg<Output = Self>
    + PartialOrd
    + Copy
    + Clone
    + From<u8>
    + Default
    + Display
    + Debug
    + Sync
    + Send
    + PartialOrd
    + 'static
{
}

macro_rules! impl_num_for_types {
    ($($type:ty),*) => {
        $(
        impl Num for $type {}
        )*
    };
}
impl_num_for_types!(i16, i32, i64, i128, f32, f64);

/// Float type
///
/// Special Numeric Trait for all floating points numbers
///
/// For all float numbers (f32, f64)
pub trait Float: Num {
    fn one() -> Self;
    /// 1 for positive 0 for 0 and -1 for negative
    fn sign(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn abs(self) -> Self;
    fn neg(self) -> Self;

    fn to_f64(self) -> f64;
    fn to_f32(self) -> f32;

    fn to_i32(self) -> i32;

    fn selu_lambda(self) -> Self;

    fn selu_alpha(self) -> Self;

    fn from_f64(value: f64) -> Self;
    fn from_usize(value: usize) -> Self;

    fn from_str(value: &str) -> Self;
    fn cos(self) -> Self;
    fn pi() -> Self;
    fn f32_f64(a: f32, b: f64) -> Self;
    fn if_f32_f64<T>(a: T, b: T) -> T;
}

#[warn(dead_code)]
macro_rules! impl_some_float_for_types {
    ($($type:ty),*) => {
        $(
            fn one() -> Self {1.0}
            fn pi() -> Self {3.14159}
            fn sign(self) -> Self {
                if self > Self::default() {
                    1.0
                } else if self == Self::default() {
                    0.0
                } else {
                    -1.0
                }
            }
            fn sqrt(self) -> Self { self.sqrt() }
            fn cos(self) -> Self {self.cos()}
            fn exp(self) -> Self {self.exp()}
            fn ln(self) -> Self { self.ln() }
            fn abs(self) -> Self { self.abs() }
            fn powf(self, n: $type) -> Self { self.powf(n) }
            fn neg(self) -> Self {Neg::neg(self)}
            fn to_i32(self) -> i32 { self as i32 }
        )*
    };
}

impl Float for f32 {
    impl_some_float_for_types!(f32);

    fn to_f64(self) -> f64 {
        self as f64
    }
    fn to_f32(self) -> f32 { self }

    fn selu_lambda(self) -> Self {
        1.0507f32
    }

    fn selu_alpha(self) -> Self {
        1.67326f32
    }
    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn from_usize(value: usize) -> Self {
        value as f32
    }

    fn from_str(value: &str) -> Self {
        value.parse::<f32>().unwrap()
    }

    fn f32_f64(a: f32, _: f64) -> Self {
        a
    }
    fn if_f32_f64<T>(a: T, _: T) -> T {a}
}

impl Float for f64 {
    impl_some_float_for_types!(f64);
    fn to_f64(self) -> f64 {
        self
    }
    fn to_f32(self) -> f32 { self as f32 }
    fn selu_lambda(self) -> Self {
        1.050700f64
    }

    fn selu_alpha(self) -> Self {
        1.673263f64
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn from_usize(value: usize) -> Self {
        value as f64
    }
    fn from_str(value: &str) -> Self {
        value.parse::<f64>().unwrap()
    }
    fn f32_f64(_: f32, b: f64) -> Self {
        b
    }
    fn if_f32_f64<T>(_: T, b: T) -> T { b }
}

///Structure to improve readability
pub struct DataType;

impl DataType {
    pub fn i16() -> i16 {
        0i16
    }
    pub fn i32() -> i32 {
        0i32
    }
    pub fn i64() -> i64 {
        0i64
    }
    pub fn i128() -> i128 {
        0i128
    }

    pub fn f32() -> f32 {
        0f32
    }
    pub fn f64() -> f64 {
        0f64
    }
}

#[cfg(test)]
mod tests {
    use crate::activation::{Function, ReLU};
    use crate::linalg::Matrix;
    use crate::nn::{Linear, Sequential};
    use std::time::Instant;
    use crate::loss::MSE;
    use crate::matrix;
    use crate::optim::SGD;
    //use prost_build::*;

    #[test]
    fn simple_linear() {
        let fc1: Linear<f64> = Linear::new(16, 64, true);
        let fc2: Linear<f64> = Linear::new(64, 64, true);
        let fc3: Linear<f64> = Linear::new(64, 4, true);
        let act = ReLU::new();

        let data = Matrix::new(vec![1.0; 16], 1, 16);

        let start_time = Instant::now();
        let mut ans = fc1.call(data);
        ans = act.call(ans);
        ans = fc2.call(ans);
        ans = act.call(ans);
        ans = fc3.call(ans);
        let elapsed_time = start_time.elapsed();
        println!("Time: {} micros", elapsed_time.as_micros());
        println!("{}", ans)
    }

    #[test]
    fn some_shit() {
        let x_mx = matrix![
            [3.0,6.0,7.0],
            [2.0,1.0,8.0],
            [1.0, 1.0, 1.0],
            [5.0, 3.0, 3.0]
        ];
        let y_mx = matrix![[135.0, 260.0, 220.0, 360.0]].transpose();


        let layers: Vec<Box<dyn Function<f64>>> = vec![
            Box::new(Linear::new(3, 1, false))
        ];
        let mut nn = Sequential::new(layers);

        let err = MSE::new(0.0);
        let mut optim = SGD::new(0.001);

        for _ in 0..100 {
            let v = nn.train(
                x_mx.clone(),
                y_mx.clone(),
                &mut optim,
                &err
            );
            if v < 0.1 {
                break
            }
            println!("{v}");
        }
        println!("{:?}", nn[0].get_data().unwrap());
    }
}
