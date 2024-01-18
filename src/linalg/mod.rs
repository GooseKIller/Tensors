mod vector;
mod matrix;

pub use vector::*;
pub use matrix::*;


use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

/// Numeric type
///
/// Special Trait
///
/// For most of numbers like (i16, i32, i64, i128, u8, u16, u32, u128, f32, f64)
pub trait Num:
Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + AddAssign + SubAssign + Copy + Clone + From<u8> + Default + Display + Debug
{
}
