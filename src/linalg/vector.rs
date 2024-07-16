use std::cmp::min;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, AddAssign, Sub, Mul, Index, IndexMut, SubAssign, MulAssign};
use crate::{Float, Num};
use crate::linalg::{Matrix, Tensor};


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

vector_element_type_def!(i16, i32, i64, i128, f32, f64);


/// Vector definition
///
///# Example
///```
/// use tensors::vector;
/// use tensors::linalg::Vector;
/// let a = vector![1, 2, 3];
/// ```
#[macro_export]
macro_rules! vector {
    ($($x:expr),*) => {
		Vector::from(
			vec![
				$($x,)*
			]
		)
	};
}

/// Mathematical vector.
///
/// Most of the functions are implemented:
/// Addition, Subtraction, Multiplication, Scalar product
///
/// Addition/subtraction/multiplication, works according to the principle {x_1+y_1, x_2+y_2, x_3+y_3}
#[derive(PartialEq, Eq, Debug)]
pub struct Vector<T: Num>{
	data: Vec<T>,
	pub(crate) length: usize,
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
	pub fn scalar(&self, other: &Vector<T>) -> T{
		let mut output = T::default();
		for i in 0..min(self.data.len(), other.data.len()){
			output += self[i].clone() * other[i].clone();
		}
		output
	}



	/// Returns length value
	///
	/// # Example
	///
	/// ```
	/// use tensors::linalg::Vector;
	///
	/// let a = Vector::from(vec![1, 2, 3]);
	/// a.len();//3
	/// ```
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


	/// Sum of all values of vector
	pub fn sum(self) -> T{
		let mut sum = T::default();
		for i in self.data{
			sum += i.clone();
		}
		sum
	}

	/// Sum of all absolute values of vector
	pub fn abs_sum(self) -> T{
		let mut sum = T::default();
		for i in self.data{
			if i < T::default(){
				sum  -= i
			} else {
				sum += i
			}
		}
		sum
	}

	pub fn try_from(value: Tensor<T>) -> Result<Self, &'static str>{
		if value.shape.len() != 1{
			return Err("Shape size must be 1");
		}
		Ok(Vector::from(value.data))
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

impl<T:Float> Vector<T>{

	/// Finds lengths of vector
    ///
	/// sqrt(x_1^2 + ... + x_n^2)
	/// # Example
	///
	/// ```
	/// use tensors::linalg::Vector;
	/// let a = Vector::from(vec![4.0, 3.0]);
	/// println!("{}", a.length());
	/// //5
	/// ```
	pub fn length(self) -> T{
		let mut ans = T::default();
		for i in self{
			ans += i*i;
		}
		ans.sqrt()
	}

	/// Finds an exp of all elements of vector
	///
	/// Formula: exp(x_i)
	///
	/// # Example
	/// ```
	/// use tensors::vector;
	/// use tensors::linalg::Vector;
	/// let a = vector![1.0, 0.0];
	///
	/// println!("{}", a);
	/// //{e, 1}
	/// ```
	pub fn exp(self) -> Vector<T>{
		let mut ans = Vec::with_capacity(self.length);
		for i in self{
			ans.push(i.exp());
		}
		Vector::from(ans)
	}

}

impl<T:Num> Add<Vector<T>> for Vector<T>{
	type Output = Vector<T>;
	fn add(self, rhs: Vector<T>) -> Self::Output {
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

impl<T: Num> Add<&Vector<T>> for Vector<T> {
	type Output = Vector<T>;

	fn add(self, rhs: &Vector<T>) -> Self::Output {
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

impl<T:Num> AddAssign<Vector<T>> for Vector<T>{
	fn add_assign(&mut self, rhs: Vector<T>) {
		if self.length != rhs.length {
			panic!("!!!Different size of vectors!!!")
		}
		for i in 0..self.length{
			self[i] += rhs[i];
		}
	}
}
impl<T:Num> AddAssign<&Vector<T>> for Vector<T>{
	fn add_assign(&mut self, rhs: &Vector<T>) {
		if self.length != rhs.length {
			panic!("!!!Different size of vectors!!!")
		}
		for i in 0..self.length{
			self[i] += rhs[i];
		}
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

impl<T:Num> AddAssign<T> for Vector<T>{
	fn add_assign(&mut self, rhs: T) {
		for i in 0..self.length{
			self[i] += rhs;
		}
	}
}

impl<T:Num> Sub<Vector<T>> for Vector<T> {
	type Output = Vector<T>;

	fn sub(self, rhs: Vector<T>) -> Self::Output {
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
impl<T:Num> Sub<&Vector<T>> for Vector<T> {
	type Output = Vector<T>;

	fn sub(self, rhs: &Vector<T>) -> Self::Output {
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
impl<T:Num> SubAssign<&Vector<T>> for Vector<T>{
	fn sub_assign(&mut self, rhs: &Vector<T>) {
		if self.length != rhs.length {
			panic!("!!!Different size of vectors!!!")
		}
		for i in 0..self.length{
			self[i] -= rhs[i];
		}
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

impl<T:Num> Mul<&Vector<T>> for Vector<T> {
	type Output = Vector<T>;
	fn mul(self, rhs: &Vector<T>) -> Self::Output {
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

impl<T:Num> MulAssign<Vector<T>> for Vector<T>{
	fn mul_assign(&mut self, rhs: Vector<T>) {
		if self.length != rhs.length {
			panic!("!!!Different size of vectors!!!")
		}

		for i in 0..self.length{
			self[i] = self[i] * rhs[i];
		}
	}
}

impl<T:Num> MulAssign<&Vector<T>> for Vector<T>{
	fn mul_assign(&mut self, rhs: &Vector<T>) {
		if self.length != rhs.length {
			panic!("!!!Different size of vectors!!!")
		}

		for i in 0..self.length{
			self[i] = self[i] * rhs[i];
		}
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

impl<T:Num> MulAssign<T> for Vector<T>  {
	fn mul_assign(&mut self, rhs: T) {

		for i in 0..self.length{
			self[i] = self[i] * rhs;
		}
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

impl<T:Num> From<Tensor<T>> for Vector<T>  {
	fn from(value: Tensor<T>) -> Self{
		if value.shape.len() != 1{
			panic!("Shape size must be 1")
		}
		Vector::from(value.data)
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


#[cfg(test)]
mod tests{
	use super::*;

	#[test]
	fn macros_test(){
		let a = vector![1, 2, 3];
		let b = Vector::from(vec![1, 2, 3]);

		assert_eq!(a, b)
	}

	#[test]
	fn display_test(){
		let a = Vector::from_num(-1, 3);
		let b = Vector::from(vec![-1,-1,-1]);
		assert_eq!(a, b);
	}

	#[test]
	fn simple_vector_task(){
		let vector_a = vector![0.0, -1.0];
		let vector_b = vector![0.0, 1.0];

		let scalar = vector_a.scalar(&vector_b);
		let prod_len = vector_a.length() * vector_b.length();

		let cos_a = (scalar/prod_len) as f32;
		assert_eq!(std::f32::consts::PI, cos_a.acos());
	}

	#[test]
	fn another_task(){
		let a = vector![-3.0+7.0, 3.0-6.0];
		let b = vector![2.0-2.0, 5.0-1.0];
		let c = vector![-4.0-4.0, -2.0+4.0];
		let d= a+b+c;
		assert_eq!(5.0, d.length());
	}

	#[test]
	fn from_number(){
		let a = Vector::from_num(1, 2);

		assert_eq!(Vector::from(vec![1,1]), a);
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
        let a = Vector::from(vec![4.0,3.0]);

        assert_eq!(a.length(), 5f64);
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

	#[test]
	fn exp_test(){
		let a = vector![1.0, 0.0];
		assert_eq!(vector![1.0, std::f64::consts::E], a.exp());
	}

	#[test]
	fn abs_sum(){
		let a = vector![-3.0, 5.0, 7.0];
		assert_eq!(15.0, a.abs_sum());
	}

	#[test]
	fn add_many_times(){
		let mut a = vector![1.0, 1.0, 1.0];
		let mut b = vector![0.0, 0.0, 0.0];
		b += 1.;

		a += &b;
		a = a + &b;
		assert_eq!(vector![3.0,3.0,3.0], a);
	}

	#[test]
	fn sub_many_times(){
		let mut a = vector![1.0, 1.0, 1.0];
		let mut b = vector![2.0, 2.0, 2.0];
		a += 1.;
		b -= 1.;
		a -= &b;
		a = a - b;
		assert_eq!(vector![0.0,0.0,0.0], a);
	}

	#[test]
	fn mul_many_times(){
		let mut a = Vector::from_num(1.0, 3);
		let mut b = Vector::from_num(0.5, 3);
		b *= 4.0;
		a *= &b;
		a = a * &b;
		assert_eq!(Vector::from_num(4.0, 3), a);
	}
}
