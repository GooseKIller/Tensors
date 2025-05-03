use std::fmt::{Display, Formatter};
use crate::linalg::{Matrix, Vector};
use crate::{Float, Num};
use rayon::prelude::IntoParallelRefMutIterator;
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};
use rand::distributions::{Distribution, Standard};
use rand::random;


struct IndexIterator {
    shape: Vec<usize>,
    current: Vec<usize>,
    done: bool,
}

impl IndexIterator {
    fn new(shape: &Vec<usize>) -> Self {
        IndexIterator {
            shape: shape.clone(),
            current: vec![0; shape.len()],
            done: false,
        }
    }
}

impl Iterator for IndexIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current.clone();

        // Increment the current index
        for i in (0..self.shape.len()).rev() {
            self.current[i] += 1;
            if self.current[i] < self.shape[i] {
                break;
            }
            self.current[i] = 0;
            if i == 0 {
                self.done = true;
            }
        }

        Some(result)
    }
}

pub fn broadcast_shape(curr_shape: &Vec<usize>, other_shape: &Vec<usize>) -> Vec<usize> {
    let mut result = Vec::new();
    let mut self_iter = curr_shape.iter().rev();
    let mut other_iter = other_shape.iter().rev();

    loop {
        match (self_iter.next(), other_iter.next()) {
            (Some(&a), Some(&b)) => {
                if a == b {
                    result.push(a);
                } else if a == 1 {
                    result.push(b);
                } else if b == 1 {
                    result.push(a);
                } else {
                    panic!(
                        "Incompatible shapes for broadcasting, got {:?} and {:?}",
                        curr_shape, other_shape
                    );
                }
            }
            (Some(&a), None) => result.push(a),
            (None, Some(&b)) => result.push(b),
            (None, None) => break,
        }
    }
    result.reverse();
    result
}

#[macro_export]
macro_rules! tensor {
    // ([$([$([$x:expr),*]),*]),*])
    // 3D Tensor
    /*([$(
        [$(
            $($x:expr),* $(,)*
        ),* $(,)*]
    ),* $(,)*])

     */
    ([$($inner:tt),* $(,)?]) => {{
        Tensor::from(vec![$(
            tensor!(@inner $inner)
        ),*])
    }};
    // Внутренние уровни: рекурсивно преобразуем каждый уровень
    (@inner [$($inner:tt),* $(,)?]) => {{
        vec![$(
            tensor!(@inner $inner)
        ),*]
    }};
    // Базовый случай: элемент-выражение
    (@inner $x:expr) => { $x };

    // 2D Tensor
    ($([$($x:expr),* $(,)*]),* $(,)*) => {
        Tensor::from(vec![
            $(vec![
                $($x,)*
            ],)*
        ])
    };
    // 1D Tensor
    ($($x:expr),*) => {
        Tensor::from(
            vec![
                $($x,)*
            ]
        )
    };


    /*
    // 1D tensor
    ([$($x:expr),* $(,)?]) => {{
        let data = vec![$($x),*];
        let len = data.len();
        Tensor::new(data, vec![len])
    }};

    // 2D tensor
    ([ $( [ $($x:expr),* $(,)? ] ),* $(,)? ]) => {{
        let data = vec![$( vec![$($x),*] ),*];
        let rows = data.len();
        let cols = data[0].len();
        let flat: Vec<_> = data.into_iter().flatten().collect();
        Tensor::new(flat, vec![rows, cols])
    }};

    // 3D tensor
    ([ $( [ $( [ $($x:expr),* $(,)? ] ),* $(,)? ] ),* $(,)? ]) => {{
        let data = vec![
            $(
                vec![
                    $( vec![$($x),*] ),*
                ]
            ),*
        ];
        let d1 = data.len();
        let d2 = data[0].len();
        let d3 = data[0][0].len();
        let flat: Vec<_> = data.into_iter().flatten().flatten().collect();
        Tensor::new(flat, vec![d1, d2, d3])
    }};
     */
}



/// Reference: [nreHieW](https://github.com/nreHieW/r-nn/blob/main/src/core/tensor/mod.rs)
#[derive(PartialEq, Eq, Debug)]
pub struct Tensor<T: Num> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: Vec<usize>,
}

fn product(shape: &Vec<usize>) -> usize {
    shape.iter().product()
}

impl<T: Num> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            product(&shape),
            "!!!Inconsistent data and dimensions combination for tensor!!!"
        );
        Self { data, shape }
    }

    pub fn from_num(num: T, shape: Vec<usize>) -> Self {
        let mut mul = 1;
        for i in &shape {
            mul *= i;
        }
        let data = vec![num; mul];
        Self { data, shape }
    }

    /// Return flatten vector
    pub fn get_data(&self) -> Vec<T> {
        self.data.clone()
    }

    pub fn reshape(&self, shape:Vec<usize>) -> Self {
        assert_eq!(product(&shape), product(&self.shape), "!!!Reshape size mismatch!!!");
        Self {
            data: self.data.clone(),
            shape
        }
    }

    /// Removes dimensions from the tensor.
    ///
    /// This function allows for the removal of dimensions from tensor.
    ///
    /// # Notes
    /// If the specified index `dim` is out of bounds,
    /// the function will return the same tensor without any modifications.
    pub fn squeeze(&self, dim: i32) -> Self {
        if dim == -1 {
            Self {
                data: self.data.clone(),
                shape: self.shape.iter().filter(|&x| *x != 1).cloned().collect(),
            }
        } else {
            Self {
                data: self.data.clone(),
                shape: self
                    .shape
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != dim as usize)
                    .map(|(_, v)| *v)
                    .collect(),
            }
        }
    }

    pub fn unsqueeze(&self, dim: usize) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);
        Self {
            data: self.data.clone(),
            shape: new_shape,
        }
    }

    /// !!NOT FINISHED YET!!
    pub fn kronecker(&self, other: Tensor<T>) -> Tensor<T> {
        let new_shape: Vec<usize> = self
            .shape
            .iter()
            .zip(other.shape.clone())
            .map(|(&s, o)| s * o)
            .collect();

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

    pub fn cat(&self, other: &Tensor<T>, dim:usize) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape[dim] = self.shape[dim] + other.shape[dim];
        for i in 0..self.shape.len() {
            if i != dim {
                assert_eq!(self.shape[i], other.shape[i]);
            }
        }

        let index_iter = IndexIterator::new(&new_shape);
        let result_data: Vec<_> = index_iter
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|idx| {
                if idx[dim] < self.shape[dim] {
                    self[&idx.as_slice()].clone()
                } else {
                    let mut other_idx = idx.clone();
                    other_idx[dim] -= self.shape[dim];
                    other[&other_idx.as_slice()].clone()
                }
            })
            .collect();


        Self {
            data: result_data,
            shape: new_shape,
        }
    }
/*
    fn _get_slice(&self, idxs: Vec<Range<usize>>, broadcasted_shape: Vec<usize>) -> Self {
        let mut result_shape = Vec::with_capacity(idxs.len());
        let mut result_data = Vec::new();

        for range in &idxs {
            result_shape.push(range.end - range.start);
        }

        let index_iter = IndexIterator::new(&result_shape);
        for idx in index_iter {
            let original_idx: Vec<usize> = idx
                .iter()
                .zip(idxs.iter())
                .map(|(&i, range)| range.start + i)
                .collect();

            let item = self
                ._get_item(original_idx, broadcasted_shape.clone())
                .clone();
            result_data.push(item);
        }

        Self {
            data: result_data,
            shape: result_shape,
        }
    }
    ®
 */

    pub fn convolve(&self, kernel: &Tensor<T>) -> Tensor<T> {
        assert_eq!(
            kernel.shape, kernel.shape,
            "!!!Kernel dimensions must be less than or equal to input tensor dimensions.!!!"
        );

        let output_shape: Vec<usize> = self
            .shape
            .iter()
            .zip(kernel.shape.iter())
            .map(|(input, kernel)| input - kernel + 1)
            .collect();

        let output_size = output_shape.iter().product::<usize>();
        let mut output_data = vec![T::default(); output_size];

        output_data
            .par_iter_mut()
            .enumerate()
            .for_each(|(output_index, output_value)| {
                let mut sum = T::default();
                let output_coords = self.index_to_coords(output_index, &output_shape);

                for kernel_index in 0..kernel.data.len() {
                    let kernel_coords = kernel.index_to_coords(kernel_index, &kernel.shape);
                    let input_coords: Vec<usize> = output_coords
                        .iter()
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
        coords
            .iter()
            .enumerate()
            .map(|(dim, &coord)| coord * self.shape[dim])
            .sum()
    }
}

impl<T: Float> Tensor<T> {
    /// Creates a matrix with random numbers(between 0 and 1)
    /// This is achieved using the [Box-Muller transform](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform), which generates normally distributed random numbers
    /// from uniformly distributed random numbers.
    pub fn randn(&self, shape: Vec<usize>) -> Self
    where
        Standard: Distribution<T> {
        let n = product(&shape);
        Self {
            data: vec![T::default(); n]
                .iter()
                .map(|_| {
                    (-T::from(2) * random::<T>().ln()).sqrt() // Bpx - Muller Method
                        * (T::from(2) * T::pi() * random::<T>()).cos()
                })
                .collect(),
            shape,
        }
    }

}

impl<T: Num> Index<&[usize]> for Tensor<T> {
    type Output = T;
    fn index(&self, index: &[usize]) -> &Self::Output {
        assert_eq!(
            self.shape.len(),
            index.len(),
            "!!!Amount of shape and index does not equal \nShape size is {}. Index size is {}.!!!",
            self.shape.len(),
            index.len()
        );

        let mut linear_index = 0;
        for i in 0..index.len() {
            assert!(
                index[i] <= self.shape[i],
                "!!!Index out of bounds.\nShape: {:?} Index: {:?}!!!",
                self.shape,
                index
            );
            linear_index = linear_index * self.shape[i] + index[i];
        }
        &self.data[linear_index]
    }
}
/*
impl<T: Num> Index<&Vec<usize>> for Tensor<T> {
    type Output = T;
    fn index(&self, index: &Vec<usize>) -> &Self::Output {
        assert_eq!(
            self.shape.len(),
            index.len(),
            "!!!Amount of shape and index does not equal \nShape size is {}. Index size is {}.!!!",
            self.shape.len(),
            index.len()
        );

        let mut linear_index = 0;
        for i in 0..index.len() {
            assert!(
                index[i] <= self.shape[i],
                "!!!Index out of bounds.\nShape: {:?} Index: {:?}!!!",
                self.shape,
                index
            );
            linear_index = linear_index * self.shape[i] + index[i];
        }
        &self.data[linear_index]
    }
}*/

impl<T: Num> IndexMut<&[usize]> for Tensor<T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        if self.shape.len() != index.len() {
            panic!(
                "!!!Amount of shape and index does not equals \n\
             Shape size is {}. Index size is {}.!!!",
                self.shape.len(),
                index.len()
            )
        }
        let mut linear_index = 0;
        for i in 0..index.len() {
            if index[i] > self.shape[i] {
                panic!(
                    "!!!Index out of bounds.\n\
                 Shape: {:?} Index: {:?}!!!",
                    self.shape, index
                )
            }
            linear_index = linear_index * self.shape[i] + index[i];
        }
        &mut self.data[linear_index]
    }
}
/*
impl<T: Num> IndexMut<&Vec<usize>> for Tensor<T> {
    fn index_mut(&mut self, index: &Vec<usize>) -> &mut Self::Output {
        if self.shape.len() != index.len() {
            panic!(
                "!!!Amount of shape and index does not equals \n\
             Shape size is {}. Index size is {}.!!!",
                self.shape.len(),
                index.len()
            )
        }
        let mut linear_index = 0;
        for i in 0..index.len() {
            if index[i] > self.shape[i] {
                panic!(
                    "!!!Index out of bounds.\n\
                 Shape: {:?} Index: {:?}!!!",
                    self.shape, index
                )
            }
            linear_index = linear_index * self.shape[i] + index[i];
        }
        &mut self.data[linear_index]
    }
}*/

impl<T: Num> From<Matrix<T>> for Tensor<T> {
    fn from(value: Matrix<T>) -> Self {
        let data = value.data;
        let shape = vec![value.rows, value.cols];
        Self { data, shape }
    }
}

impl<T: Num> From<Vector<T>> for Tensor<T> {
    fn from(value: Vector<T>) -> Self {
        let data: Vec<T> = value.into();
        let shape = vec![data.len()];
        Self { data, shape }
    }
}

impl<T:Num> From<Vec<T>> for Tensor<T> {
    fn from(value: Vec<T>) -> Self {
        let shape = vec![value.len()];
        Self {data: value, shape}
    }
}

impl<T:Num> From<Vec<Vec<T>>> for Tensor<T> {
    fn from(value: Vec<Vec<T>>) -> Self {
        let rows = value.len();
        let cols  = value.first().map_or(0, |row| row.len());

        for row in value.iter().skip(1) {
            assert_eq!(row.len(), cols, "!!!All columns must be equal!!!");
        }
        assert!(
            !(rows != 0 && cols == 0),
            "!!!Invalid matrix dimensions. Multiple empty rows!!!"
        );
        let data = value.into_iter().flatten().collect();
        Self {data, shape: vec![rows, cols]}
    }
}

impl<T: Num> From<Vec<Vec<Vec<T>>>> for Tensor<T> {
    fn from(value: Vec<Vec<Vec<T>>>) -> Self {
        let d1 = value.len();
        let d2 = value
            .first()
            .map_or(0, |layer| layer.len());
        let d3 = value
            .first()
            .and_then(|layer| layer.first().map(|row| row.len()))
            .expect("!!!Expected 3D Tensor but got malformed structure!!!");

        for (i, layer) in value.iter().enumerate() {
            assert_eq!(
                layer.len(),
                d2,
                "!!!Inconsistent d2 (rows) in layer {}: expected {d2}, got {}!!!",
                i,
                layer.len()
            );
            for (j, row) in layer.iter().enumerate() {
                assert_eq!(
                    row.len(),
                    d3,
                    "!!!Inconsistent d3 (columns) at layer {}, row {}: expected {}, got {}!!!",
                    i,
                    j,
                    d3
                    row.len()
                );
            }
        }

        let flat = value.into_iter().flatten().flatten().collect();
        Tensor {
            data: flat,
            shape: vec![d1, d2, d3],
        }
    }
}


impl<T: Num> Add<T> for Tensor<T> {
    type Output = Tensor<T>;
    fn add(self, rhs: T) -> Self::Output {
        let mut data = self.data.clone();
        let shape = self.shape.clone();
        data.par_iter_mut().for_each(|x| {
            *x += rhs;
        });
        Self { data, shape }
    }
}

impl<T: Num> AddAssign<T> for Tensor<T> {
    fn add_assign(&mut self, rhs: T) {
        self.data.par_iter_mut().for_each(|x| {
            *x += rhs;
        });
    }
}

impl<T: Num> Add<&Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;
    fn add(self, rhs: &Tensor<T>) -> Self::Output {
        assert_eq!(
            self.shape,
            rhs.shape,
            "{}",
            format!(
                "!!!Can not add shapes must be equal!!!\
        \nMatrix A:{:?} Matrix B:{:?}",
                self.shape, rhs.shape
            )
        );
        let mut data = self.data.clone();
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x += rhs.data[i];
        });
        Tensor::new(data, self.shape.clone())
    }
}

impl<T: Num> AddAssign<&Tensor<T>> for Tensor<T> {
    fn add_assign(&mut self, rhs: &Tensor<T>) {
        assert_eq!(
            self.shape,
            rhs.shape,
            "{}",
            format!(
                "!!!Can not add shapes must be equal!!!\
        \nMatrix A:{:?} Matrix B:{:?}",
                self.shape, rhs.shape
            )
        );
        self.data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x += rhs.data[i];
        });
    }
}

impl<T: Num> Sub<T> for Tensor<T> {
    type Output = Tensor<T>;
    fn sub(self, rhs: T) -> Self::Output {
        let mut data = self.data.clone();
        let shape = self.shape.clone();
        data.par_iter_mut().for_each(|x| {
            *x -= rhs;
        });
        Self { data, shape }
    }
}

impl<T: Num> SubAssign<T> for Tensor<T> {
    fn sub_assign(&mut self, rhs: T) {
        self.data.par_iter_mut().for_each(|x| {
            *x -= rhs;
        });
    }
}

impl<T: Num> Sub<&Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;
    fn sub(self, rhs: &Tensor<T>) -> Self::Output {
        assert_eq!(
            self.shape,
            rhs.shape,
            "{}",
            format!(
                "!!!Can not add shapes must be equal!!!\
        \nMatrix A:{:?} Matrix B:{:?}",
                self.shape, rhs.shape
            )
        );
        let mut data = self.data.clone();
        data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x -= rhs.data[i];
        });
        Tensor::new(data, self.shape.clone())
    }
}

impl<T: Num> SubAssign<&Tensor<T>> for Tensor<T> {
    fn sub_assign(&mut self, rhs: &Tensor<T>) {
        assert_eq!(
            self.shape,
            rhs.shape,
            "{}",
            format!(
                "!!!Can not add shapes must be equal!!!\
        \nMatrix A:{:?} Matrix B:{:?}",
                self.shape, rhs.shape
            )
        );
        self.data.par_iter_mut().enumerate().for_each(|(i, x)| {
            *x -= rhs.data[i];
        });
    }
}

impl<T: Num> Mul<T> for Tensor<T> {
    type Output = Tensor<T>;
    fn mul(self, rhs: T) -> Self::Output {
        let mut data = self.data.clone();
        let shape = self.shape.clone();
        data.par_iter_mut().for_each(|x| {
            *x = *x * rhs;
        });
        Self { data, shape }
    }
}

impl<T: Num> MulAssign<T> for Tensor<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.data.par_iter_mut().for_each(|x| {
            *x = *x * rhs;
        });
    }
}

impl<T:Num> Display for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let raw: Vec<String> = self.data.iter().map(|x| format!("{x}")).collect();
        let width = raw.iter().map(|s| s.len()).max().unwrap_or(0);

        let padded: Vec<String> = raw
            .into_iter()
            .map(|s| format!("{:>width$}", s, width= width))
            .collect();

        fn rec(
            f: &mut Formatter<'_>,
            shape: &[usize],
            data: &[String],
            idx: &mut usize,
            indent: usize,
        ) -> Result<(), std::fmt::Error> {
            if shape.len() == 1 {
                // Базовый случай: одномерный вектор
                f.write_str("[")?;
                for i in 0..shape[0] {
                    if i > 0 {
                        f.write_str(" ")?;
                    }
                    f.write_str(&data[*idx])?;
                    *idx += 1;
                }
                f.write_str("]")?;
            } else {
                // Рекурсивный случай: выводим shape[0] блоков
                f.write_str("[")?;
                let n = shape[0];
                let sub = &shape[1..];
                for i in 0..n {
                    if i > 0 {
                        // между блоками более высокой размерности вставляем пустую строку
                        if shape.len() >= 3 {
                            f.write_str("\n\n")?;
                        } else {
                            f.write_str("\n")?;
                        }
                        // отступ перед новым блоком
                        for _ in 0..indent + 1 {
                            f.write_str(" ")?;
                        }
                    }
                    rec(f, sub, data, idx, indent + 1)?;
                }
                f.write_str("]")?;
            }
            Ok(())
        }

        let mut idx = 0;
        rec(f, &self.shape, &padded, &mut idx, 0)
    }
}

impl<T: Num> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::linalg::tensor::Tensor;
    use crate::linalg::{Matrix, Vector};

    #[test]
    fn macro_test() {
        let a = tensor![1,2, 3,4];
        println!("{a}");
    }

    #[test]
    fn new_tensor() {
        let data = vec![1];
        let shape = vec![1usize, 1usize, 1usize];
        let _tensor = Tensor::new(data, shape.clone());
        let _tensor = Tensor::from_num(1, shape);
    }

    #[test]
    fn index() {
        let tr = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]);
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    println!("I:{}, J:{} Value:{}", i, j, tr[&[i, j, k]])
                }
            }
        }
    }

    #[test]
    fn into_matrix() {
        let tr = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let mx = Matrix::from(tr);
        println!("{}", mx)
    }

    #[test]
    fn into_vector() {
        let tr = Tensor::new(vec![1, 2, 3, 4], vec![4]);
        let mx = Vector::from(tr);
        println!("{}", mx)
    }

    #[test]
    fn add() {
        let mut tr = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]);
        let ans = Tensor::new(vec![2, 3, 4, 5, 6, 7, 8, 9], vec![2, 2, 2]);
        assert_eq!(ans, tr.clone() + 1);
        tr += 1;
        assert_eq!(ans, tr);
    }
    #[test]
    fn sub() {
        let mut tr = Tensor::new(vec![2, 3, 4, 5, 6, 7, 8, 9], vec![2, 2, 2]);
        let ans = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]);
        assert_eq!(ans, tr.clone() - 1);
        tr -= 1;
        assert_eq!(ans, tr);
    }

    #[test]
    fn mul() {
        let mut tr = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]);
        let ans = Tensor::new(vec![2, 4, 6, 8, 10, 12, 14, 16], vec![2, 2, 2]);
        assert_eq!(ans, tr.clone() * 2);
        tr *= 2;
        assert_eq!(ans, tr)
    }

    #[test]
    fn kron() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let b = Tensor::new(vec![0, 5, 6, 7], vec![2, 2]);

        println!("{:?}", a.kronecker(b));
    }

    #[test]
    fn cat() {
        let a = Tensor::new(vec![1,2,3,4,5,6], vec![2,3]);
        let b = Tensor::new(vec![7,8,9,10,11,12], vec![2,3]);
        println!("{}", Matrix::from(a.cat(&b, 1)))
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
