use std::cmp::min;
use std::ops::{Add, AddAssign, Sub, Mul, Div, Index, IndexMut, SubAssign};


pub trait Num:
	Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + AddAssign + SubAssign + Copy + Clone + From<u8> + Default
{
}
#[macro_export]
macro_rules! vector_element_type_def {
    ($T: ty) => {
		impl Num for $T {}

		impl Mul<&Vector<$T>> for $T {
			type Output = Vector<$T>;

			fn mul(self, rhs: &Vector<$T>) -> Self::Output {
				self * rhs
			}
		}
	};

	($T:ty, $($Ti: ty),+) => {
		vector_element_type_def!($T);
		vector_element_type_def!($($Ti),+);
	};
}

vector_element_type_def!(i16, i32, i64, i128, u8, u16, u32, u128, f32, f64);

#[derive(PartialEq, Eq, Debug)]
pub struct Vector<T: Num>{
	data: Vec<T>,
	length: usize,
}

impl<T: Num> Vector<T>{
	pub fn new(data: Vec<T>) -> Self{
		Self {
			data: data.clone(),
			length: data.len()
		}
	}

	pub fn from_num(num:T, length:usize) -> Self{
		Self::new(vec![num; length])
	}

	pub fn scalar(self, other: &Vector<T>) -> T{
		let mut output = T::default();
		for i in 0..min(self.data.len(), other.data.len()){
			output += self[i].clone() * other[i].clone();
		}
		output
	}

	pub fn clone(&self) -> Vector<T> {
		Vector{
			data: self.data.clone(),
			length: self.length
		}
	}

	pub fn resize(&self, new_length: usize) -> Vector<T> {
		let mut new_data = Vec::new();
		for i in &self.data{
			new_data.push((*i).clone());
		}

		new_data.resize(new_length, T::default());
		Self{
			data: new_data.to_owned(),
			length: new_length,
		}
	}

	fn len(self) -> T {
		// doesn't work correct unfinished
		/// returns the length of vector
		/// Equation: sqrt(x1^2 + x2^2 + ... + xn^2)
		return T::default();
		/*
        let mut squared = T::default();
		for i in self.data{
			squared += i*i;
		}
		squared.sqrt()
		*/
	}
}

impl<T:Num> Index<usize> for Vector<T>{
	type Output = T;

	fn index(&self, index:usize) -> &Self::Output {
		if index > self.length{
			panic!("!!!Vector index out of bounds!!!");
		}
		&self.data[index]
	}
}

impl<T:Num> IndexMut<usize> for Vector<T> {
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		if index > self.length{
			panic!("!!!Vector index out of bounds!!!");
		}
		&mut self.data[index]
	}
}

impl<T: Num> Add for Vector<T> {
	type Output = Vector<T>;

	fn add(self, rhs: Self) -> Self::Output {
		if self.length != rhs.length {
			panic!("!!!Different size of vectors!!!")
		}
		let mut vec = vec![T::default(); self.length];
		for i in 0..self.length{
			vec[i] = self[i] + rhs[i];
		}
		Vector::new(vec)
	}
}

impl<T:Num> Sub for Vector<T> {
	type Output = Vector<T>;

	fn sub(self, rhs: Self) -> Self::Output {
		if self.length != rhs.length {
			panic!("!!!Different size of vectors!!!")
		}
		let mut vec = vec![T::default(); self.length];
		for i in 0..self.length{
			vec[i] = self[i] - rhs[i];
		}
		Vector::new(vec)
	}
}

impl<T: Num> AddAssign for Vector<T>{
	fn add_assign(&mut self, rhs: Self) {
		if self.length != rhs.length {
			panic!("!!!Different size of vectors!!!")
		}
		for i in 0..self.length{
			self[i] += rhs[i];
		}

	}
}

impl<T:Num> SubAssign for Vector<T>{
	fn sub_assign(&mut self, rhs: Self) {
		if self.length != rhs.length {
			panic!("!!!Different size of vectors!!!")
		}
		for i in 0..self.length{
			self[i] -= rhs[i];
		}
	}
}

impl<T:Num> Mul for Vector<T>{
	type Output = Vector<T>;

	fn mul(self, rhs: Self) -> Self::Output {
		if self.length != rhs.length {
			panic!("!!!Different size of vectors!!!")
		}

		let mut vec = vec![T::default(); self.length];

		for i in 0..self.length{
			vec[i] = self[i] * rhs[i];
		}
		Vector::new(vec)
	}
}


#[cfg(test)]
mod tests{
	use super::*;

	#[test]
	fn from_number(){
		let a = Vector::from_num(1, 2);

		assert_eq!(Vector::new(vec![1,1]), a);
	}

	#[test]
	fn resize(){
		let a = Vector::from_num(1, 3);
		let a = a.resize(2);

		assert_eq!(Vector::new(vec![1,1]), a);
	}
	
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

        assert_eq!(a.len(), 0);
    }

	#[test]
	fn add_vectors(){
		let a = vec![1, 2, 1];
		let b = vec![2, 1, 0];
		let a = Vector::new(a);
		let b = Vector::new(b);

		let c = a+b;

		assert_eq!(Vector::new(vec![3,3,1]), c);
	}

	#[test]
	fn add_to_vector(){
		let a = vec![1, 2, 1];
		let b = vec![2, 1, 0];
		let mut a = Vector::new(a);
		let b = Vector::new(b);

		a += b;

		assert_eq!(Vector::new(vec![3,3,1]), a);
	}

	#[test]
	fn sub_to_vector(){
		let a = vec![1, 2, 1];
		let b = vec![2, 1, 0];
		let mut a = Vector::new(a);
		let b = Vector::new(b);

		a -= b;

		assert_eq!(Vector::new(vec![3,3,1]), a);
	}

	#[test]
	fn mul_to_vector(){
		let a = vec![1, 2, 1];
		let b = vec![2, 1, 0];
		let mut a = Vector::new(a);
		let b = Vector::new(b);

		a = a * b;

		assert_eq!(Vector::new(vec![2,2,0]), a);
	}


}
