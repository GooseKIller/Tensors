use std::cmp;
use std::ops::{Add, AddAssign, Sub, Mul, Div, Index};


pub trait Num:
	Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + AddAssign + Copy + Clone + From<u8>
{
}

impl Num for f32 {}
impl Num for f64 {}
impl Num for i32 {}
impl Num for i64 {}
/*
#[macro_export]
macro_rules! vector_element_type_def {
    ($T: ty) => {
        // Implement trait for type
        impl Num for $T {}

        /*// Defining left-hand multiplication in this form
        // prevents errors for uncovered types
        impl Mul<&Vector<$T>> for $T {
            type Output = Vector<$T>;

            fn mul(self, rhs: &Vector<$T>) -> Self::Output {
                rhs * self
            }
        }*/
    };

    ($T: ty, $($Ti: ty),+) => {
        // Decompose type definitions recursively
        vector_element_type_def!($T);
        vector_element_type_def!($($Ti),+);
    };
}

vector_element_type_def!(i16,i32,i64,i128,u8,u16,u32,u128,f32,f64);
*/
#[derive(PartialEq, Eq, Debug)]
pub struct Vector<T: Num>{
	data: Vec<T>,
}

impl<T: Num> Vector<T> where f64: From<T>{
	pub fn new(data: Vec<T>) -> Self{
		Vector{ data }
	}

	pub fn scalar(self, other: &Vector<T>) -> T{
		let mut output = 0.into();
		for i in 0..(cmp::min(self.data.len(), other.data.len())){
			output += self[i].clone() * other[i].clone();
		}
		output
	}

    pub fn len(self) -> f64{
        let squared: f64 = self.data.iter().map(|x|{
            let x: f64 = x.clone().into();
            x * x
        }
        ).sum();
        squared.sqrt()
    }
}

impl<T:Num> Index<usize> for Vector<T>{
	type Output = T;

	fn index(&self, index:usize) -> &Self::Output {
		if index > self.data.len(){
			panic!("Vector index out of bounds");
		}
		&self.data[index]
	}
}


#[cfg(test)]
mod tests{
	use super::*;
	
	#[test]
	fn scalar(){
		let a = vec![1, 2];
		let b = vec![2, 1];

		let a = Vector::new(a);
		let b = Vector::new(b);

		assert_eq!(a.scalar(&b), 4);

	}

	#[test]
	fn scalar_dif_size(){
		let a = vec![1, 2, 1];
		let b = vec![2, 1];

		let a = Vector::new(a);
		let b = Vector::new(b);

		assert_eq!(a.scalar(&b), 4);


	}

    #[test]
    fn len_vector(){
        let a = Vector::new(vec![4,3]);

        assert_eq!(a.len(), 5f64);
    }
}
