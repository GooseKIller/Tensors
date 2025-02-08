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
        let mul = shape.iter().product::<usize>();
        assert_eq!(data.len(), mul, "!!!Inconsistent data and dimensions combination for tensor!!!");
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

    /// !!NOT FINISHED YET!!
    pub fn kronecker(&self, other:Tensor<T>) -> Tensor<T>{
        let new_shape: Vec<usize> = self.shape.iter()
            .zip(other.shape.clone()).map(|(&s, o)| s*o).collect();

        let new_size = new_shape.iter().product::<usize>();

        let mut ans = vec![T::default(); new_size];

        ans.par_iter_mut().enumerate().for_each(|(index, x)| {
            let mut a_indices = vec![0; self.shape.len()];
            let mut b_indices = vec![0; other.shape.len()];

            let mut temp_index = index;
            for dim in (0..self.shape.len()).rev() {
                a_indices[dim] = temp_index % self.shape[dim];
                temp_index /= self.shape[dim];

                b_indices[dim] = temp_index % other.shape[dim];
                temp_index /= other.shape[dim];
            }
            let a_value = self[&a_indices];
            let b_value = other[&b_indices];

            *x = a_value * b_value;
        });

        Tensor {
            data: ans,
            shape: new_shape,
        }
    }

    pub fn convolve(&self, kernel: &Tensor<T>) -> Tensor<T> {
        assert_eq!(kernel.shape, kernel.shape,
            "!!!Kernel dimensions must be less than or equal to input tensor dimensions.!!!");

        let output_shape: Vec<usize> = self.shape.iter()
            .zip(kernel.shape.iter())
            .map(|(input, kernel)| input - kernel + 1)
            .collect();

        let output_size = output_shape.iter().product::<usize>();
        let mut output_data = vec![T::default(); output_size];

        output_data.par_iter_mut().enumerate().for_each(|(output_index, output_value)| {
            let mut sum = T::default();
            let output_coords = self.index_to_coords(output_index, &output_shape);

            for kernel_index in 0..kernel.data.len() {
                let kernel_coords = kernel.index_to_coords(kernel_index, &kernel.shape);
                let input_coords: Vec<usize> = output_coords.iter()
                    .zip(kernel_coords.iter())
                    .map(|(&o, &k)| o + k)
                    .collect();

                let input_index = self.coords_to_index(&input_coords);
                sum += self.data[input_index] * kernel.data[kernel_index];
            }

            *output_value = sum;
        });

        Tensor {
            data: output_data,
            shape: output_shape,
        }
    }

    fn index_to_coords(&self, index: usize, shape: &[usize]) -> Vec<usize> {
        let mut coords = Vec::new();
        let mut idx = index;
        for &dim in shape.iter().rev() {
            coords.push(idx % dim);
            idx /= dim;
        }
        coords.reverse();
        coords
    }

    fn coords_to_index(&self, coords: &[usize]) -> usize {
        coords.iter().enumerate().map(|(dim, &coord)| coord * self.shape[dim]).sum()
    }
}

impl<T:Num> Index<&[usize]> for Tensor<T>{
    type Output = T;
    fn index(&self, index: &[usize]) -> &Self::Output {
        assert_eq!(self.shape.len(), index.len(),
                   "!!!Amount of shape and index does not equal \nShape size is {}. Index size is {}.!!!",
                   self.shape.len(), index.len());

        let mut linear_index = 0;
        for i in 0..index.len() {
            assert!(index[i] <= self.shape[i],
                    "!!!Index out of bounds.\nShape: {:?} Index: {:?}!!!",
                    self.shape,
                    index);
            linear_index = linear_index * self.shape[i] + index[i];
        }
        &self.data[linear_index]
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

    #[test]
    fn kron(){
        let a = Tensor::new(vec![1,2,3,4], vec![2,2]);
        let b = Tensor::new(vec![0,5,6,7], vec![2,2]);

        println!("{:?}", a.kronecker(b));
    }

    #[test]
    fn conv() {
        /*
        let input_data = vec![
            1, 2, 3, 0,
            0, 1, 2, 3,
            1, 0, 1, 2,
            2, 3, 0, 1,
        ];
        let input_shape = vec![4, 4];
        let input_tensor = Tensor::new(input_data, input_shape);

        // Создаем ядро свертки 2x2
        let kernel_data = vec![
            1, 0,
            0, 1,
        ];
        let kernel_shape = vec![2, 2];
        let kernel_tensor = Tensor::new(kernel_data, kernel_shape);

        // Применяем свертку
        let output_tensor = input_tensor.convolve(&kernel_tensor);

        println!("Output Tensor: {:?}", output_tensor);
        */
    }
}