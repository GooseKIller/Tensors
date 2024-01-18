use std::cmp::min;
use crate::linalg::matrix::Matrix;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, AddAssign, Sub, Mul, Div, Index, IndexMut, SubAssign};
use crate::linalg::Num;


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

/// Mathematical vector.
///
/// Most of the functions are implemented:
/// Addition, Subtraction, Multiplication, Scalar product
///
/// Addition/subtraction/multiplication, works according to the principle {x_1+y_1, x_2+y_2, x_3+y_3}
#[derive(PartialEq, Eq, Debug)]
pub struct Vector<T: Num>{
	data: Vec<T>,
	length: usize,
}

impl<T: Num> Vector<T>{
	pub fn new() -> Self{
		Self {
			data: vec![],
			length: 0,
		}
	}

	///Creates a Vector from number
	///
	/// # Example
	///```
	/// use tensors::linalg::Vector;
	///
	/// let a = Vector::from_num(1, 2);//{1 1}
	/// ```
	pub fn from_num(num:T, length:usize) -> Self{
		Self::from(vec![num; length])
	}


	/// Finds a scalar value of 2 vectors
	///
	/// # Logic
	///
	/// ```
	/// use tensors::linalg::Vector;
	///
	/// let a = Vector::from(vec![1, 2, 3]);
	/// let b = Vector::from(vec![1, 2, 3]);
	/// let c = a.scalar(&b);//{1*1 2*2 3*3}
	/// ```
	pub fn scalar(self, other: &Vector<T>) -> T{
		let mut output = T::default();
		for i in 0..min(self.data.len(), other.data.len()){
			output += self[i].clone() * other[i].clone();
		}
		output
	}



	pub fn len(self) -> usize{
		self.length.clone()
	}

	pub fn clone(&self) -> Vector<T> {
		Vector{
			data: self.data.clone(),
			length: self.length
		}
	}

	/// Changing size of Vector
	///
	/// # Examples
	///
	/// ```
	/// use tensors::linalg::Vector;
	///
	/// let vector = Vector::from_num(10,1);//{10}
	///	let new_vector = vector.resize(2);//{10 0}
	/// ```
	///
	/// ```
	/// use tensors::linalg::Vector;
	///
	/// let vector = Vector::from_num(10, 2);//{10 10}
	/// let new_vector = vector.resize(1);//{10}
	/// ```
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

impl<T: Num> Add<Vector<T>> for Vector<T> {
	type Output = Vector<T>;

	fn add(self, rhs: Self) -> Self::Output {
		if self.length != rhs.length {
			panic!("!!!Different size of vectors!!!")
		}
		let mut vec = vec![T::default(); self.length];
		for i in 0..self.length{
			vec[i] = self[i] + rhs[i];
		}
		Vector::from(vec)
	}
}

impl<T:Num> Add<T> for Vector<T>{
	type Output = Vector<T>;
	fn add(self, rhs: T) -> Self::Output {
		let mut vec = vec![T::default(); self.length];
		for i in 0..self.length{
			vec[i] = self[i] + rhs;
		}
		Vector::from(vec)
	}
}

impl<T:Num> Sub<Vector<T>> for Vector<T> {
	type Output = Vector<T>;

	fn sub(self, rhs: Self) -> Self::Output {
		if self.length != rhs.length {
			panic!("!!!Different size of vectors!!!")
		}
		let mut vec = vec![T::default(); self.length];
		for i in 0..self.length{
			vec[i] = self[i] - rhs[i];
		}
		Vector::from(vec)
	}
}

impl<T:Num> Sub<T> for Vector<T> {
	type Output = Vector<T>;

	fn sub(self, rhs: T) -> Self::Output {
		let mut vec = vec![T::default(); self.length];
		for i in 0..self.length{
			vec[i] = self[i] - rhs;
		}
		Vector::from(vec)
	}
}

impl<T: Num> AddAssign<Vector<T>> for Vector<T>{
	fn add_assign(&mut self, rhs: Self) {
		if self.length != rhs.length {
			panic!("!!!Different size of vectors!!!")
		}
		for i in 0..self.length{
			self[i] += rhs[i];
		}

	}
}

impl<T:Num> AddAssign<T> for Vector<T>{
	fn add_assign(&mut self, rhs: T) {
		for i in 0..self.length{
			self[i] += rhs;
		}
	}
}

impl<T:Num> SubAssign<Vector<T>> for Vector<T>{
	fn sub_assign(&mut self, rhs: Self) {
		if self.length != rhs.length {
			panic!("!!!Different size of vectors!!!")
		}
		for i in 0..self.length{
			self[i] -= rhs[i];
		}
	}
}

impl<T:Num> SubAssign<T> for Vector<T>{
	fn sub_assign(&mut self, rhs: T) {
		for i in 0..self.length{
			self[i] -= rhs;
		}
	}
}

impl<T:Num> Mul<Vector<T>> for Vector<T>{
	type Output = Vector<T>;

	fn mul(self, rhs: Self) -> Self::Output {
		if self.length != rhs.length {
			panic!("!!!Different size of vectors!!!")
		}

		let mut vec = vec![T::default(); self.length];

		for i in 0..self.length{
			vec[i] = self[i] * rhs[i];
		}
		Vector::from(vec)
	}
}

impl<T:Num> Mul<T> for Vector<T>{
	type Output = Vector<T>;

	fn mul(self, rhs: T) -> Self::Output {
		let mut vec = vec![T::default(); self.length];

		for i in 0..self.length{
			vec[i] = self[i] * rhs;
		}
		Vector::from(vec)
	}
}

impl<T:Num> Into<Vec<T>> for Vector<T>{
	fn into(self) -> Vec<T> {
		self.data.clone().to_vec()
	}
}

impl<T: Num> From<Vec<T>> for Vector<T> {
	fn from(value: Vec<T>) -> Self {
		Self{
			data: value.clone(),
			length: value.len()
		}
	}
}

impl<T:Num> Display for Vector<T> {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		//let string = format!("{{{}}}");
		let string = self.data
			.iter()
			.map(|x| format!("{}", x))
			.collect::<Vec<_>>()
			.join(" ");

		write!(f, "{{{}}}", string)
	}
}

impl<T:Num> Iterator for Vector<T>{
	type Item = T;
	fn next(&mut self) -> Option<Self::Item> {
		self.data.pop()

	}
}

impl<T:Num> From<Matrix<T>> for Vector<T> {
	fn from(value: Matrix<T>) -> Self {
		if value.col() != 1 {
			panic!("!!!Can't translate matrix to vector: columns are not equal 1. Column = {}!!!", value.col())
		}
		Vector::from(value.get_row(0))
	}
}


#[cfg(test)]
mod tests{
	use super::*;

	#[test]
	fn display_test(){
		let a = Vector::from_num(-1, 3);
		let b = Vector::from(vec![-1,-1,-1]);
		assert_eq!(a, b);
	}

	#[test]
	fn from_number(){
		let a = Vector::from_num(1, 2);

		assert_eq!(Vector::from(vec![1,1]), a);
	}

	#[test]
	#[should_panic]
	fn err_from_matrix(){
		let a = Matrix::from_num(10, 2, 2);
		let a = Vector::from(a);
	}

	#[test]
	fn from_matrix(){
		let a = Matrix::from_num(10, 2, 1);
		let a = Vector::from(a);
		let b = Vector::from_num(10, 2);
		assert_eq!(a, b)
	}

	#[test]
	fn resize(){
		let a = Vector::from_num(1, 3);
		let a = a.resize(2);

		assert_eq!(Vector::from(vec![1,1]), a);
	}

	#[test]
	fn scalar(){
		let a = vec![1, 2];
		let b = vec![2, 1];

		let a = Vector::from(a);
		let b = Vector::from(b);

		assert_eq!(a.scalar(&b), 4);

	}

	#[test]
	fn scalar_dif_size(){
		let a = vec![1, 2, 1];
		let b = vec![2, 1];

		let a = Vector::from(a);
		let b = Vector::from(b);

		assert_eq!(a.scalar(&b), 4);


	}

    #[test]
    fn len_vector(){
        let a = Vector::from(vec![4,3]);
		let mut l = 0f64;
		for i in a{
			l += (i*i) as f64
		}

        assert_eq!(l.sqrt(), 5f64);
    }

	#[test]
	fn add_vectors(){
		let a = vec![1, 2, 1];
		let b = vec![2, 1, 0];
		let a = Vector::from(a);
		let b = Vector::from(b);

		let c = a+b;

		assert_eq!(Vector::from(vec![3,3,1]), c);
	}

	#[test]
	fn add_to_vector(){
		let a = vec![1, 2, 1];
		let b = vec![2, 1, 0];
		let mut a = Vector::from(a);
		let b = Vector::from(b);

		a += b;

		assert_eq!(Vector::from(vec![3,3,1]), a);
	}

	#[test]
	fn sub_to_vector(){
		let mut a = Vector::from_num(1, 1);
		let b = Vector::from_num(1, 1);

		a -= b;
		a += 1;

		assert_eq!(Vector::from(vec![1]), a);
	}

	#[test]
	fn mul_to_vector(){
		let a = vec![1, 2, 1];
		let b = vec![2, 1, 0];
		let mut a = Vector::from(a);
		let b = Vector::from(b);

		a = a * b;
		a = a * 0;

		assert_eq!(Vector::from(vec![0,0,0]), a);
	}


}
