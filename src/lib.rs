use std::fmt::{Debug, Display};
use std::cmp::PartialOrd;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

pub mod linalg;
pub mod activation;
pub mod nn;
pub mod loss;

/// Numeric type
///
/// Special Trait
///
/// For most of the numbers like (i16, i32, i64, i128, f32, f64)
pub trait Num:
Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Div + AddAssign + SubAssign + Neg<Output = Self> + PartialOrd + Copy + Clone + From<u8> + Default + Display + Debug + Sync + Send
{
}

/// Float type
///
/// Special Numeric Trait for all floating points numbers
///
/// For all float numbers (f32, f64)
pub trait Float:
Num{
    fn one() -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;

    fn powf(self, n:Self) -> Self;
    fn neg(self) -> Self;
    
    fn to_f64(self) -> f64;

    fn selu_lambda(self) -> Self;

    fn selu_alpha(self) -> Self;
}

impl Float for f32{
    fn one() -> Self {1f32}
    fn sqrt(self) -> Self { self.sqrt() }
    fn exp(self) -> Self { self.exp() }
    fn powf(self, n: f32) -> Self { self.powf(n) }
    fn neg(self) -> Self { Neg::neg(self) }
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn selu_lambda(self) -> Self {
        1.0507f32
    }

    fn selu_alpha(self) -> Self {
        1.67326f32
    }
}

impl Float for f64{
    fn one() -> Self {1f64}
    fn sqrt(self) -> Self { self.sqrt() }
    fn exp(self) -> Self {self.exp()}
    fn powf(self, n: f64) -> Self { self.powf(n) }
    fn neg(self) -> Self {Neg::neg(self)}
    fn to_f64(self) -> f64 {
        self
    }
    fn selu_lambda(self) -> Self {
        1.0507009873554804934193349852946f64
    }

    fn selu_alpha(self) -> Self {
        1.6732632423543772848170429916717f64
    }
}

pub struct DataType;

impl DataType {
    pub fn i16() -> i16{
        0i16
    }
    pub fn i32() -> i32{
        0i32
    }
    pub fn i64() -> i64{
        0i64
    }
    pub fn i128() -> i128{
        0i128
    }

    pub fn f32() -> f32{
        0f32
    }
    pub fn f64() -> f64{
        0f64
    }
}

#[cfg(test)]
mod tests{
    use std::time::Instant;
    use crate::activation::{Function, ReLU};
    use crate::matrix;
    use crate::linalg::Matrix;
    use crate::nn::Linear;

    #[test]
    fn simple_linear(){
        let fc1:Linear<f64> = Linear::new(16, 64, true);
        let fc2:Linear<f64> = Linear::new(64, 64, true);
        let fc3:Linear<f64> = Linear::new(64, 4, true);
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
    }
}
