//!
//! # Tensors
//!
//! Tensors is a lightweight machine learning library in Rust. It provides a simple and efficient way to create and train machine learning models with minimal dependencies.
//! ## Dependencies
//! The library uses the following dependencies:
//! - [rayon](https://crates.io/crates/rayon) - for parallel computations on CPU.
//! - [rand](https://crates.io/crates/rand) - for random number generation.
//!
//! ## Example
//! ```rust
//! use tensorrs::{activation::Module,
//!     linalg::{Tensor, Vector},
//!     loss::mse, nn::{Linear, Sequential},
//!     optim::{Adam, Optimizer},
//!     autodiff::{AutoGrad, Var}};
//! 
//! let x_val = Tensor::from(Vector::linspace(-1.0, 1.0, 8, true)).reshape(vec![8, 1]);
//! let y_val = &x_val * 8.0 - 10.0;
//! 
//! let x = Var::leaf(x_val, false);
//! let y = Var::leaf(y_val, false);
//! 
//! let model = Sequential::new(vec![
//!     Box::new(Linear::new(1,1, true))
//! ]);
//! 
//! let mut optim = Adam::new(model.parameters(), 0.1);
//! for i in 0..1000 {
//!     optim.zero_grad();
//!     let output = model.forward(&x);
//!     let loss = mse(&output, &y);
//! 
//!     loss.backward();
//!     if i % 100 == 0 {
//!         let val = loss.value().item();
//!         println!("{}", val);
//!         if val < 0.001 {
//!             break;
//!         }
//!     }
//!     optim.step();
//! }
//! ```
//! Thanks for using Tensors!!!

#![doc(html_logo_url = "https://raw.githubusercontent.com/GooseKIller/Tensors/main/assets/tensorsLogo.svg")]
// The whole library is safe Rust. Weight loading used to reach for a raw pointer;
// it goes through Arc::make_mut now, like every other in-place tensor operation.
#![forbid(unsafe_code)]
use std::cmp::PartialOrd;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

pub mod activation;
pub mod linalg;
pub mod loss;
pub mod nn;
pub mod optim;
pub mod autodiff;
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
    + MulAssign
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
    + Sum
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

    fn from_f32(value: f32) -> Self;
    fn from_f64(value: f64) -> Self;
    fn from_usize(value: usize) -> Self;

    fn from_str(value: &str) -> Self;
    fn cos(self) -> Self;
    fn sin(self) -> Self;
    /// Hyperbolic tangent, taken from the platform rather than built out of
    /// exponentials, which overflow long before the function itself saturates.
    fn tanh(self) -> Self;
    fn atan2(self, n:Self) -> Self;
    fn pi() -> Self;
    fn f32_f64(a: f32, b: f64) -> Self;
    fn if_f32_f64<T>(a: T, b: T) -> T;
}

#[warn(dead_code)]
macro_rules! impl_some_float_for_types {
    ($($type:ty),*) => {
        $(
            fn one() -> Self {1.0}
            fn pi() -> Self { Self::f32_f64(core::f32::consts::PI, core::f64::consts::PI) }
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
            fn sin(self) -> Self {self.sin()}
            fn tanh(self) -> Self {self.tanh()}
            fn atan2(self, n:Self) -> Self {self.atan2(n)}
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

    fn from_f32(value: f32) -> Self {
        value
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

    fn from_f32(value: f32) -> Self {
        value as f64
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
mod test {
    use crate::{activation::Module,
        linalg::{Tensor, Vector},
        loss::mse, nn::{Linear, Sequential},
        optim::{Adam, Optimizer},
         autodiff::{AutoGrad, Var}};

    #[test]
    fn test_example() {
        let x_val = Tensor::from(Vector::linspace(-1.0, 1.0, 8, true)).reshape(vec![8, 1]);
        let y_val = &x_val * 8.0 - 10.0;

        let x = Var::leaf(x_val, false);
        let y = Var::leaf(y_val, false);

        let model = Sequential::new(vec![
            Box::new(Linear::new(1,1, true))
        ]);

        let mut optim = Adam::new(model.parameters(), 0.1);
        for i in 0..1000 {
            optim.zero_grad();
            let output = model.forward(&x);
            let loss = mse(&output, &y);

            loss.backward();
            if i % 100 == 0 {
                let val = loss.value().item();
                println!("{}", val);
                if val < 0.001 {
                    break;
                }
            }

            optim.step();
        }
    }
}

#[cfg(test)]
mod leak_audit {
    use crate::activation::*;
    use crate::autodiff::core::build_topo;
    use crate::autodiff::*;
    use crate::linalg::Tensor;
    use crate::nn::*;
    use std::rc::{Rc, Weak};

    /// Builds a graph, notes a weak reference to every one of its nodes, drops
    /// the root and reports how many nodes outlived it.
    fn survivors<F>(build: F) -> usize
    where
        F: FnOnce() -> VarRef<f32>,
    {
        let weaks: Vec<Weak<_>> = {
            let root = build();
            root.zero_grad();
            root.backward();

            build_topo(&root).iter().map(|n| Rc::downgrade(&n.0)).collect()
        };

        weaks.iter().filter(|w| w.upgrade().is_some()).count()
    }

    fn image() -> VarRef<f32> {
        Var::leaf(Tensor::<f32>::randn(vec![2, 3, 8, 8]), false)
    }
    fn sequence() -> VarRef<f32> {
        Var::leaf(Tensor::<f32>::randn(vec![2, 4, 16]), false)
    }
    fn rows() -> VarRef<f32> {
        Var::leaf(Tensor::<f32>::randn(vec![4, 6]), false)
    }

    /// Every intermediate node has to die with the graph. The only nodes allowed
    /// to outlive it are the parameters, which the layer itself owns - they have
    /// to survive, or there would be nothing left to train.
    #[test]
    fn a_graph_leaves_nothing_but_parameters_behind() {
        macro_rules! check {
            ($name:expr, $layer:expr, $input:expr) => {{
                let layer = $layer;
                let expected = Module::<f32>::parameters(&layer).len();
                let alive = survivors(|| Module::<f32>::forward(&layer, $input).sum());
                assert_eq!(alive, expected,
                    "{}: {alive} nodes outlived the graph, expected only its {expected} parameters",
                    $name);
            }};
        }

        check!("Linear", Linear::<f32>::new(6, 3, true), &rows());
        check!("Conv2d", Conv2d::<f32>::same(3, 4, (3, 3), true), &image());
        check!("MaxPool2d", MaxPool2d::new((2, 2)), &image());
        check!("AvgPool2d", AvgPool2d::new((2, 2)), &image());
        check!("Flatten", Flatten::new(), &image());
        check!("LayerNorm", LayerNorm::<f32>::new(16, 1e-5), &sequence());
        check!("Dropout", Dropout::new(0.5), &rows());
        check!("RNN", RNN::<f32>::new(16, 8, true), &sequence());
        check!("MultiHeadAttention", MultiHeadAttention::<f32>::new(16, 2, true), &sequence());
        check!("Sigmoid", Sigmoid::new(), &rows());
        check!("Tanh", Tanh::new(), &rows());
        check!("SoftMax", SoftMax::new(), &rows());
        check!("ReLU", ReLU::new(), &rows());
        check!("PReLU", PReLU::<f32>::new(), &rows());
        check!("ELU", ELU::new(1.0f32), &rows());
        check!("SELU", SELU::new(), &rows());

        // Embedding takes ids rather than a node, so it does not fit the macro
        let embedding = Embedding::<f32>::new(10, 4);
        let alive = survivors(|| embedding.forward(&[vec![1, 2], vec![3, 4]]).sum());
        assert_eq!(alive, embedding.parameters().len(),
            "Embedding: {alive} nodes outlived the graph");
    }

    /// A whole model, not just one layer at a time.
    #[test]
    fn a_stacked_model_leaves_nothing_behind() {
        let model: Sequential<f32> = Sequential::new(vec![
            Box::new(Conv2d::same(3, 4, (3, 3), true)),
            Box::new(ReLU::new()),
            Box::new(MaxPool2d::new((2, 2))),
            Box::new(Flatten::new()),
            Box::new(Linear::new(4 * 4 * 4, 8, true)),
            Box::new(ReLU::new()),
            Box::new(Linear::new(8, 3, true)),
            Box::new(SoftMax::new()),
        ]);

        let alive = survivors(|| model.forward(&image()).sum());
        assert_eq!(alive, model.parameters().len(),
            "{alive} nodes outlived a stacked model, expected only its parameters");
    }
}
