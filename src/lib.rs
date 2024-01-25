use std::fmt::{Debug, Display};
use std::cmp::PartialOrd;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

pub mod linalg;
pub mod activation;

/// Numeric type
///
/// Special Trait
///
/// For most of numbers like (i16, i32, i64, i128, f32, f64)
pub trait Num:
Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + AddAssign + SubAssign + PartialOrd + Copy + Clone + From<u8> + Default + Display + Debug
{
}

///Float type
///
/// Special Trait
///
/// For all float numbers (f32, f64)
pub trait Float:
Num{
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;

    fn powf(self, n:Self) -> Self;
    fn neg(self) -> Self;
}

impl Float for f32{
    fn sqrt(self) -> f32{
        self.sqrt()
    }
    fn exp(self) -> Self { self.exp() }

    fn powf(self, n: f32) -> Self { self.powf(n) }

    fn neg(self) -> Self { Neg::neg(self) }
}
impl Float for f64{
    fn sqrt(self) -> f64{
        self.sqrt()
    }
    fn exp(self) -> Self {self.exp()}
    fn neg(self) -> Self {Neg::neg(self)}
    fn powf(self, n: f64) -> Self {self.powf(n)}
}