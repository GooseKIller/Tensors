use std::ops::{Add, Index, IndexMut, Sub, Mul, AddAssign, SubAssign, MulAssign};
use rayon::prelude::IntoParallelRefMutIterator;
use rayon::prelude::*;
use crate::linalg::{Matrix, Vector};
use crate::Num;

#[derive(PartialEq, Eq, Debug)]
pub struct Tensor<T:Num>{
    pub(crate) data: Vec<T>,
    pub(crate) shape: Vec<usize>
}

impl<T:Num> Tensor<T>{
    pub fn new(data:Vec<T>, shape:Vec<usize>) -> Self{
        let mut mul = 1;
        for i in &shape{
            mul *= i;
        }
        if data.len() != mul{
            panic!("!!!Inconsistent data and dimensions combination for tensor!!!")
        }
        Self{
            data,
            shape
        }
    }

    pub fn from_num(num:T, shape:Vec<usize>) -> Self{
        let mut mul = 1;
        for i in &shape{
            mul *= i;
        }
        let data = vec![num; mul];
        Self{
            data,
            shape
        }
    }
}

impl<T:Num> Index<&[usize]> for Tensor<T>{
    type Output = T;
    fn index(&self, index: &[usize]) -> &Self::Output {
        if self.shape.len() != index.len(){
            panic!("!!!Amount of shape and index does not equals \n\
             Shape size is {}. Index size is {}.!!!", self.shape.len(), index.len())
        }
        let mut linear_index = 0;
        for i in 0..index.len() {
            if index[i] > self.shape[i] {
                panic!("!!!Index out of bounds.\n\
                 Shape: {:?} Index: {:?}!!!",
                       self.shape,
                       index
                )
            }
            linear_index = linear_index * self.shape[i] + index[i];
        }
        &self.data[linear_index]
        /*
        // Mixtral code
        for (i, &dim) in self.shape.iter().enumerate() {
            if index[i] >= dim {
                panic!(
                    "!!!Index out of bounds. Shape: {:?} Index: {:?}!!!",
                    self.shape, index
                )
            }
            linear_index = linear_index * dim + index[i];
        }*/
    }
}

impl<T:Num> IndexMut<&[usize]> for Tensor<T>{
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        if self.shape.len() != index.len(){
            panic!("!!!Amount of shape and index does not equals \n\
             Shape size is {}. Index size is {}.!!!", self.shape.len(), index.len())
        }
        let mut linear_index = 0;
        for i in 0..index.len(){
            if index[i] > self.shape[i] {
                panic!("!!!Index out of bounds.\n\
                 Shape: {:?} Index: {:?}!!!",
                       self.shape,
                       index
                )
            }
            linear_index = linear_index * self.shape[i] + index[i];
        }
        &mut self.data[linear_index]
    }
}

impl<T:Num> From<Matrix<T>> for Tensor<T> {
    fn from(value: Matrix<T>) -> Self {
        let data = value.data;
        let shape = vec![value.rows, value.cols];
        Self {
            data,
            shape 
        }
    }
}

impl<T:Num> From<Vector<T>> for Tensor<T>  {
    fn from(value: Vector<T>) -> Self {
        let data:Vec<T> = value.into();
        let shape = vec![data.len()];
        Self {
            data,
            shape
        }

    }
}

impl<T:Num> Add<T> for Tensor<T>{
    type Output = Tensor<T>;
    fn add(self, rhs: T) -> Self::Output {
        let mut data = self.data.clone();
        let shape = self.shape.clone();
        data.par_iter_mut().for_each(|x| {
            *x += rhs;
        });
        Self{
            data,
            shape
        }
    }
}

impl<T:Num> AddAssign<T> for Tensor<T>  {
    fn add_assign(&mut self, rhs: T) {
        self.data.par_iter_mut().for_each(|x| {
            *x += rhs;
        });
    }
}

impl<T:Num> Add<&Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;
    fn add(self, rhs: &Tensor<T>) -> Self::Output {
        assert_eq!(self.shape, rhs.shape, "{}", format!("!!!Can not add shapes must be equal!!!\
        \nMatrix A:{:?} Matrix B:{:?}", self.shape, rhs.shape));
        let mut data = self.data.clone();
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x += rhs.data[i];
        });
        Tensor::new(
            data,
            self.shape.clone()
        )
    }
}

impl<T:Num> AddAssign<&Tensor<T>> for Tensor<T>{
    fn add_assign(&mut self, rhs: &Tensor<T>) {
        assert_eq!(self.shape, rhs.shape, "{}", format!("!!!Can not add shapes must be equal!!!\
        \nMatrix A:{:?} Matrix B:{:?}", self.shape, rhs.shape));
        self.data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x += rhs.data[i];
        });
    }
}

impl<T:Num> Sub<T> for Tensor<T>  {
    type Output = Tensor<T>;
    fn sub(self, rhs: T) -> Self::Output {
        let mut data = self.data.clone();
        let shape = self.shape.clone();
        data.par_iter_mut().for_each(|x| {
            *x -= rhs;
        });
        Self{
            data,
            shape
        }
    }
}

impl<T:Num> SubAssign<T> for Tensor<T>  {
    fn sub_assign(&mut self, rhs: T) {
        self.data.par_iter_mut().for_each(|x| {
            *x -= rhs;
        });
    }
}

impl<T:Num> Sub<&Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;
    fn sub(self, rhs: &Tensor<T>) -> Self::Output {
        assert_eq!(self.shape, rhs.shape, "{}", format!("!!!Can not add shapes must be equal!!!\
        \nMatrix A:{:?} Matrix B:{:?}", self.shape, rhs.shape));
        let mut data = self.data.clone();
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x -= rhs.data[i];
        });
        Tensor::new(
            data,
            self.shape.clone()
        )
    }
}

impl<T:Num> SubAssign<&Tensor<T>> for Tensor<T>{
    fn sub_assign(&mut self, rhs: &Tensor<T>) {
        assert_eq!(self.shape, rhs.shape, "{}", format!("!!!Can not add shapes must be equal!!!\
        \nMatrix A:{:?} Matrix B:{:?}", self.shape, rhs.shape));
        self.data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x -= rhs.data[i];
        });
    }
}

impl<T:Num> Mul<T> for Tensor<T> {
    type Output = Tensor<T>;
    fn mul(self, rhs: T) -> Self::Output {
        let mut data = self.data.clone();
        let shape = self.shape.clone();
        data.par_iter_mut().for_each(|x| {
            *x = *x * rhs;
        });
        Self{
            data,
            shape
        }
    }
}

impl<T:Num> MulAssign<T> for Tensor<T>  {
    fn mul_assign(&mut self, rhs: T) {
        self.data.par_iter_mut().for_each(|x| {
            *x = *x * rhs;
        });
    }
}

impl<T:Num> Clone for Tensor<T>{
    fn clone(&self) -> Self {
        Self{
            data: self.data.clone(),
            shape: self.shape.clone(),
        }
    }
}

#[cfg(test)]
mod test{
    use crate::linalg::tensor::Tensor;
    use crate::linalg::{Matrix, Vector};

    #[test]
    fn new_tensor(){
        let data = vec![1];
        let shape = vec![1usize, 1usize, 1usize];
        let _tensor = Tensor::new(data, shape.clone());
        let _tensor = Tensor::from_num(1, shape);
    }

    #[test]
    fn index(){
        let tr = Tensor::new(vec![1,2,3,4,5,6,7,8], vec![2,2,2]);
        for i in 0..2{
            for j in 0..2{
                for k in 0..2{
                    println!("I:{}, J:{} Value:{}", i,j,tr[&[i,j, k]])
                }

            }
        }
    }

    #[test]
    fn into_matrix(){
        let tr = Tensor::new(vec![1,2,3,4], vec![2,2]);
        let mx = Matrix::from(tr);
        println!("{}", mx)
    }

    #[test]
    fn into_vector(){
        let tr = Tensor::new(vec![1,2,3,4], vec![4]);
        let mx = Vector::from(tr);
        println!("{}", mx)
    }

    #[test]
    fn add(){
        let mut tr = Tensor::new(vec![1,2,3,4,5,6,7,8], vec![2,2,2]);
        let ans = Tensor::new(vec![2,3,4,5,6,7,8,9], vec![2,2,2]);
        assert_eq!(ans, tr.clone()+1);
        tr += 1;
        assert_eq!(ans, tr);
    }
    #[test]
    fn sub(){
        let mut tr = Tensor::new(vec![2,3,4,5,6,7,8,9], vec![2,2,2]);
        let ans = Tensor::new(vec![1,2,3,4,5,6,7,8], vec![2,2,2]);
        assert_eq!(ans, tr.clone()-1);
        tr -= 1;
        assert_eq!(ans, tr);
    }

    #[test]
    fn mul(){
        let mut tr = Tensor::new(vec![1,2,3,4,5,6,7,8], vec![2,2,2]);
        let ans = Tensor::new(vec![2,4,6,8,10,12,14,16], vec![2,2,2]);
        assert_eq!(ans, tr.clone()*2);
        tr *= 2;
        assert_eq!(ans, tr)
    }
}