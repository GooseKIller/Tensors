use rand::random;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::{sync::Arc, usize};
//use rayon::prelude::*;
use crate::{Float, Num, linalg::Vector};
//use std::ops::{Index, IndexMut};
use std::fmt::{Debug, Display, Formatter};
//use crate::linalg::{Matrix, Vector};
//use rayon::prelude::IntoParallelRefMutIterator;
use rand::distributions::{Distribution, Standard};

pub(crate) fn product(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

fn compute_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![1; shape.len()];
    let mut step = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = step;
        step *= shape[i] as isize;
    }
    strides
}

#[macro_export]
macro_rules! tensor {
    // 3D
    ( [ $( [ $( [ $( $x:expr ),* $(,)? ] ),* $(,)? ] ),* $(,)? ] ) => {
        $crate::linalg::Tensor::from(vec![
            $(
                vec![
                    $(
                        vec![ $( $x ),* ],
                    )*
                ],
            )*
        ];)
    };
    // 2D
    ($([$($x:expr),* $(,)*]),* $(,)*) => {
        $crate::linalg::Tensor::from(vec![
            $(vec![
                $($x,)*
            ],)*
        ])
    };
    // 1D
    ( $( $x:expr ),* $(,)? ) => {
        $crate::linalg::Tensor::from(vec![ $( $x ),* ])
    };
}


#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) struct Storage<T> {
    pub(crate) data: Vec<T>
}

#[derive(Debug, PartialEq, Eq)]
pub struct Tensor<T> {
    pub(crate) storage: Arc<Storage<T>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<isize>,
    pub(crate) offset: isize
}

impl<T: Clone> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        if data.len() == 1 {
            return Self::scalar(data[0].clone());
        }
        assert_eq!(data.len(), product(&shape[..]),
         "!!!Inconsistent data and dimensions combination for tensor!!!"
        );
        let strides = compute_strides(&shape[..]);
        Self { storage: Arc::new(Storage { data }),
            shape,
            strides,
            offset: 0,
        }
    }

    pub fn scalar(num: T) -> Self {
        Self {
            storage: Arc::new(Storage { data: vec![num] }),
            shape: vec![1],
            strides: vec![0],
            offset: 0, 
        }
    }

    pub fn item(&self) -> T {
        assert_eq!(self.numel(), 1, "!!!item() requires numel == 1!!!");
        self.storage.data[self.offset as usize].clone()
    }

    pub fn get_data(&self) -> Vec<T> {
        self.packed_data()
    }
    
    pub fn get_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        if indices.len() != self.shape.len() {
            return None;
        }
        
        let mut idx: isize = self.offset;
        for (i, &ind) in indices.iter().enumerate() {
            if ind >= self.shape[i] {
                return None;
            }
            idx += self.strides[i] * ind as isize;
        }

        self.storage.data.get(idx as usize)
    }

    pub fn slice(&self, start_indices: &[usize], shape: &[usize]) -> Option<Self> {
        if start_indices.len() != self.shape.len()
            || shape.len() != self.shape.len() {
            return None;
        }
        let mut offset = self.offset;
        for (i, &ind) in start_indices.iter().enumerate() {
            if ind + shape[i] > self.shape[i] {
                return None;
            }
            offset += self.strides[i] * ind as isize;
        }

        Some(Self {
            storage: self.storage.clone(),
            shape: shape.to_vec(),
            strides: self.strides.clone(),
            offset,
        })
    }

    pub fn step_slice(&self, start_indices: &[isize], shape: &[usize], steps: &[isize]) -> Option<Self> {
        if start_indices.len() != self.shape.len() ||
           shape.len() != self.shape.len() ||
           steps.len() != self.shape.len() {
            return None;
        }

        let mut offset = self.offset;
        let mut new_strides = Vec::with_capacity(self.shape.len());

        for i in 0..self.shape.len() {
            let end = start_indices[i] + steps[i] * (shape[i] as isize - 1);

            if start_indices[i] < 0 || end < 0 ||
               start_indices[i] >= self.shape[i] as isize ||
               end >= self.shape[i] as isize {
                return None;
            }

            offset += self.strides[i] * start_indices[i];
            new_strides.push(self.strides[i] * steps[i]);
        }

        Some(Self {
            storage: self.storage.clone(),
            shape: shape.to_vec(),
            strides: new_strides,
            offset,
        })
    }

    fn numel(&self) -> usize {
        product(&self.shape[..])
    }

    pub(crate) fn storage_indices(&self) -> Vec<usize> {
        let ndim = self.shape.len();
        if ndim == 0 {
            return vec![];
        }
        
        let n = self.numel();
        let mut suffix = vec![1; ndim];
        for i in (0..ndim - 1).rev() {
            suffix[i] = suffix[i + 1] * self.shape[i + 1];
        }

        let mut indices = Vec::with_capacity(n);
        for linear in 0..n {
            let mut rem = linear;
            let mut idx = self.offset;
            for i in 0..ndim {
                let coord = rem / suffix[i];
                rem %= suffix[i];
                idx += self.strides[i] * coord as isize;
            }
            indices.push(idx as usize);
        }
        indices
    }

    pub(crate) fn packed_data(&self) -> Vec<T> {
        let indices = self.storage_indices();
        let data = &self.storage.data;
        indices.into_iter().map(|i| data[i].clone()).collect()
    }

    pub fn shallow_copy(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }

    pub fn reshape(&self, new_shape:Vec<usize>) -> Self {
        assert_eq!(product(&new_shape), product(&self.shape),
         "!!!Reshape size mismatch!!!");
        assert!(self.is_contiguous(),
         "!!!Non-contiguous reshape not supported!!!");
        Self {
            storage: self.storage.clone(),
            strides: compute_strides(&new_shape),
            shape: new_shape,
            offset: self.offset,
        }
    }

    pub fn permute(&self, axes: &[usize]) -> Option<Self> {
        if axes.len() != self.shape.len() {
            return None;
        }
        
        let mut seen = vec![false; axes.len()];
        for &a in axes {
            if a > axes.len() {
                return None;
            } else if seen[a] {
                return None;
            }
            seen[a] = true;
        }

        let new_shape = axes.iter().map(|&i| self.shape[i]).collect();
        let new_strides = axes.iter().map(|&i| self.strides[i]).collect();

        Some(
            Self {
                storage: self.storage.clone(),
                shape: new_shape,
                strides: new_strides,
                offset: self.offset
            }
        )
    }

    pub fn transpose(&self) -> Self {
        assert_eq!(self.shape.len(), 2, "!!!transpose(): only 2D tensors supported!!!
        \nTry to use permute");
        self.permute(&[1, 0]).unwrap()
    }

    pub(crate) fn broadcast_to(&self, target_shape: &[usize]) -> Option<Self> {
        let src_ndim = self.shape.len();
        let dst_ndim = target_shape.len();
        
        if src_ndim > dst_ndim {
            return None;
        }

        let mut new_strides = vec![0isize; dst_ndim];

        for i in 0..dst_ndim {
            let dst_dim = target_shape[dst_ndim - 1 - i];
            let src_dim = self
                .shape
                .get(src_ndim.wrapping_sub(1 + i))
                .copied()
                .unwrap_or(1);

            let src_stride = self
                .strides
                .get(src_ndim.wrapping_sub(1 + i))
                .copied()
                .unwrap_or(0);

            if src_dim == dst_dim {
                new_strides[dst_ndim - 1 - i] = src_stride;
            } else if src_dim == 1 {
                new_strides[dst_ndim - 1 - i] = 0;
            } else {
                return None;
            }
        }

        Some(Self {
            storage: self.storage.clone(),
            shape: target_shape.to_vec(),
            strides: new_strides,
            offset: self.offset
        })
    }

    pub fn is_contiguous(&self) -> bool {
        if self.shape.len() == 1 {
            return true;
        }

        let mut excepted_stride = 1;
        for i in (0..self.shape.len()).rev() {
            if self.strides[i] != excepted_stride {
                return false;
            }
            excepted_stride *= self.shape[i] as isize;
        }
        true
    }

    pub fn is_scalar(&self) -> bool {
        self.packed_data().len() == 1
    }

    pub(crate) fn can_inplace(&self) -> bool {
        self.is_contiguous()
            && Arc::strong_count(&self.storage) == 1
            && !self.strides.iter().any(|&s| s == 0)
    }
}

impl<T: Num> Tensor<T> {
    pub fn from_num(num: T, shape: Vec<usize>) -> Self {
        Self::new(vec![num; product(&shape[..])], shape)
    }

    pub fn full_(&mut self, num: T) {
        let data = Arc::make_mut(&mut self.storage);
        data.data.fill(num);
    }

    pub fn matmul(&self, other: &Tensor<T>) -> Tensor<T> {
        let a_ndim = self.shape.len();
        let b_ndim = other.shape.len();

        assert!(a_ndim >= 2 && b_ndim >= 2,
            "matmul: both tensors must be at least 2D");
        
        // A: (..., M, K), B: (..., K, N)
        let a_m = self.shape[a_ndim - 2];
        let a_k = self.shape[a_ndim - 1];
        let b_k = other.shape[b_ndim - 2];
        let b_n = other.shape[b_ndim - 1];
        assert!(a_k == b_k,
             "matmul: inner dimensions must match (A.shape[-1] == B.shape[-2])");
            
        
        let a_batch = &self.shape[..a_ndim - 2];
        let b_batch = &other.shape[..b_ndim - 2];

        fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
            let al = a.len();
            let bl = b.len();
            let out_len = std::cmp::max(al, bl);
            let mut out = vec![1usize; out_len];
            for i in 0..out_len {
                let ai = if i >= out_len - al { a[i - (out_len - al)] } else { 1 };
                let bi = if i >= out_len - bl { b[i - (out_len - bl)] } else { 1 };
                if ai == bi || ai == 1 || bi == 1 {
                    out[i] = std::cmp::max(ai, bi);
                } else {
                    return None;
                }
            }
            Some(out)
        }
        
        let batch_shape = broadcast_shapes(a_batch, b_batch)
            .expect("matmul: batch shapes not broadcastable");

        let mut out_shape = batch_shape.clone();
        out_shape.push(a_m);
        out_shape.push(b_n);
        let out_len = product(&out_shape);

        // precompute suffix for mapping linear idx -> coords
        let mut suffix = vec![1usize; out_shape.len()];
        for i in (0..out_shape.len() - 1).rev() {
            suffix[i] = suffix[i + 1] * out_shape[i + 1];
        }

        let compute_offset = |t: &Tensor<T>, out_coords: &Vec<usize>, tail1: usize, tail2: usize| -> usize {
            let t_ndim = t.shape.len();
            // map t axes to out axes: out_ndim = out_shape.len()
            let out_ndim = out_shape.len();
            let mut off = t.offset;
            // for axes before last two of t:
            let prefix = t_ndim - 2;
            for axis in 0..prefix {
                // which axis in out corresponds to this axis in t?
                let out_axis = out_ndim - t_ndim + axis;
                let coord = if t.shape[axis] == 1 {
                    0usize
                } else {
                    out_coords[out_axis]
                };
                off += t.strides[axis] * (coord as isize);
            }
            // second-last axis (row / k depending on tensor)
            let axis_row = t_ndim - 2;
            let coord_row = tail1;
            off += t.strides[axis_row] * (coord_row as isize);
            // last axis
            let axis_col = t_ndim - 1;
            let coord_col = tail2;
            off += t.strides[axis_col] * (coord_col as isize);

            off as usize
        };

        // prepare output buffer
        let mut out_data = vec![T::default(); out_len];

        out_data.par_iter_mut().enumerate().for_each(|(out_idx, slot)| {
            // compute full coordinates for out_idx
            let mut rem = out_idx;
            let mut coords = vec![0usize; out_shape.len()];
            for i in 0..out_shape.len() {
                let c = rem / suffix[i];
                rem %= suffix[i];
                coords[i] = c;
            }

            // indices i (M index) and j (N index)
            let i = coords[out_shape.len() - 2];
            let j = coords[out_shape.len() - 1];

            // compute dot over k
            let mut acc = T::default();
            for k in 0..a_k {
                // A element at (...batch..., i, k)
                let a_off = compute_offset(self, &coords, i, k);
                // B element at (...batch..., k, j)
                let b_off = compute_offset(other, &coords, k, j);
                // multiply and accumulate (cloning underlying data)
                let a_val = self.storage.data[a_off].clone();
                let b_val = other.storage.data[b_off].clone();
                acc = acc + (a_val * b_val);
            }
            *slot = acc;
        });

        Tensor::new(out_data, out_shape)
    }

    pub fn sum(&self) -> Tensor<T> {
        let mut ans = T::default();
        for i in self.packed_data().iter() {
            ans += *i
        }
        Tensor::scalar(ans)
    }

    pub fn sum_axis(&self, axis: usize) -> Tensor<T> {
        assert!(axis < self.shape.len(), "!!!Axis index out of range!!!");

        let mut out_shape = self.shape.clone();
        let reduce_dim = out_shape.remove(axis);

        let out_len = product(&out_shape[..]);
        let mut out_data = vec![T::default(); out_len];

        out_data.par_iter_mut().enumerate().for_each(|(out_idx, x)|{
            let mut tmp = out_idx;
            let mut base_offset = self.offset as usize;
            
            for (i, &dim) in out_shape.iter().enumerate().rev() {
                let idx = tmp % dim;
                tmp /= dim;

                let in_axis = if i >= axis { i + 1 } else { i };
                base_offset += idx * (self.strides[in_axis] as usize);
            }

            let mut acc = T::default();
            let stride = self.strides[axis] as usize;

            for k in 0..reduce_dim {
                acc = acc + self.storage.data[base_offset + k * stride].clone();
            }

            *x = acc;
        });

        Tensor::new(out_data, out_shape)
    }

    pub fn sum_axis_keepdim(&self, axis: usize) -> Tensor<T> {
        assert!(axis < self.shape.len(), "!!!Axis index out of range!!!");

        let mut out_shape = self.shape.clone();
        let reduce_dim = out_shape[axis];
        out_shape[axis] = 1;

        let out_len = product(&out_shape[..]);
        let mut out_data = vec![T::default(); out_len];

        out_data.par_iter_mut().enumerate().for_each(|(out_idx, x)| {
            let mut tmp = out_idx;
            let mut base_offset = self.offset as usize;

            for (i, &dim) in out_shape.iter().enumerate().rev() {
                let idx = tmp % dim;
                tmp /= dim;

                base_offset += idx * (self.strides[i] as usize);
            }

            let mut acc = T::default();
            let stride = self.strides[axis] as usize;

            for k in 0..reduce_dim {
                acc = acc + self.storage.data[base_offset + k * stride].clone();
            }

            *x = acc;
        });

        Tensor::new(out_data, out_shape)
    }

    pub fn reduce_to_shape(&self, target: &[usize]) -> Option<Tensor<T>> {
        // 1. You cannot reduce a tensor to a higher-rank shape
        if self.shape.len() < target.len() {
            return None;
        }

        let mut out = self.shallow_copy();
        
        // 2. Reduce leading dimensions until ranks match
        // Example: self [4, 2, 3] -> target [2, 1]
        // First, sum axis 0: [4, 2, 3] -> [2, 3]
        while out.get_shape().len() > target.len() {
            out = out.sum_axis(0);
        }

        // 3. Now ranks match. Iterate and reduce dimensions where target is 1.
        // We iterate through the current shape. 
        // Important: if we remove an axis, the indices of the remaining axes shift!
        let mut i = 0;
        while i < out.get_shape().len() {
            let current_shape = out.get_shape();
            let s = current_shape[i];
            let t = target[i];

            if t == 1 && s > 1 {
                out = out.sum_axis(i);
                // After sum_axis, the rank decreases. 
                // The "next" dimension is now at the SAME index 'i'.
                // So we do NOT increment 'i' here.
            } else if s == t {
                i += 1;
            } else {
                // Shapes are incompatible (e.g., trying to reduce 3 to 2)
                return None;
            }
        }

        Some(out)
    }

    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T + Sync + Send,
    {
        let mut new_data = self.packed_data();
        new_data.par_iter_mut().for_each(|x| *x = f(*x));

        Self::new(new_data, self.shape.clone())
    }
}

impl<T: Float> Tensor<T> {
    /// Creates a matrix with random numbers(between 0 and 1)
    /// This is achieved using the
    ///  [Box-Muller transform](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform), which generates normally distributed random numbers
    /// from uniformly distributed random numbers.
    pub fn randn(shape: Vec<usize>) -> Self
    where
        Standard: Distribution<T> {
        let n = product(&shape);
        Self::new(
            vec![T::default(); n]
                .iter()
                .map(|_| {
                    (-T::from(2) * random::<T>().ln()).sqrt() // Bpx - Muller Method
                        * (T::from(2) * T::pi() * random::<T>()).cos()
                })
                .collect(),
            shape
        )
    }
    pub fn rand(shape: Vec<usize>) -> Self
    where
        Standard: Distribution<T> {
        Self::new(
            vec![T::default(); product(&shape)]
                .iter()
                .map(|_| random::<T>())
                .collect(),
            shape
        )
    }
}


impl<T: Copy> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        let data = self.packed_data(); // логически упакованный Vec<T>

        Tensor {
            storage: Arc::new(Storage { data }),
            shape: self.shape.clone(),
            strides: compute_strides(&self.shape),
            offset: 0,
        }
    }
}

impl<T: Num> From<Vector<T>> for Tensor<T> {
    fn from(value: Vector<T>) -> Self {
        let data = value.data;
        Self::new(data, vec![value.length])
    }
}

impl<T: Num> From<Vec<T>> for Tensor<T> {
    fn from(value: Vec<T>) -> Self {
        let n = value.len();
        Tensor::new(value, vec![n])
    }
}

impl<T: Num> From<Vec<Vec<T>>> for Tensor<T> {
    fn from(rows: Vec<Vec<T>>) -> Self {
        let r = rows.len();
        let c = if r == 0 { 0 } else { rows[0].len() };

        for (i, row) in rows.iter().enumerate() {
            if row.len() != c {
                panic!(
                    "Tensor::from(Vec<Vec<T>>): inconsistent row lengths: row 0 has {}, row {} has {}",
                    c,
                    i,
                    row.len()
                );
            }
        }

        // Flatten (перемещая элементы)
        let mut flat = Vec::with_capacity(r * c);
        for row in rows {
            flat.extend(row);
        }

        Tensor::new(flat, vec![r, c])
    }
}

impl<T: Num> From<Vec<Vec<Vec<T>>>> for Tensor<T> {
    fn from(blocks: Vec<Vec<Vec<T>>>) -> Self {
        let d0 = blocks.len();
        let d1 = if d0 == 0 { 0 } else { blocks[0].len() };
        let d2 = if d1 == 0 { 0 } else { blocks[0][0].len() };

        // Проверяем согласованность размеров
        for (i, block) in blocks.iter().enumerate() {
            if block.len() != d1 {
                panic!(
                    "Tensor::from(Vec<Vec<Vec<T>>>): inconsistent block sizes: block 0 has {}, block {} has {}",
                    d1, i, block.len()
                );
            }
            for (j, row) in block.iter().enumerate() {
                if row.len() != d2 {
                    panic!(
                        "Tensor::from(Vec<Vec<Vec<T>>>): inconsistent row sizes: block 0 row 0 has {}, block {} row {} has {}",
                        d2, i, j, row.len()
                    );
                }
            }
        }

        // Flatten in row-major: for b in blocks { for r in b { extend row } }
        let mut flat = Vec::with_capacity(d0 * d1 * d2);
        for block in blocks {
            for row in block {
                flat.extend(row);
            }
        }

        Tensor::new(flat, vec![d0, d1, d2])
    }
}

impl<T: Display + Clone + Debug> Display for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let values = self.packed_data();

        // scalar (0-D) case
        if self.is_scalar() {
            return write!(f, "{}", self.item());
        } 

        // prepare padded strings
        let raw: Vec<String> = values.into_iter().map(|x| format!("{}", x)).collect();
        let width = raw.iter().map(|s| s.len()).max().unwrap_or(0);
        let padded: Vec<String> = raw.into_iter()
            .map(|s| format!("{:>width$}", s, width = width))
            .collect();

        // recursive printer: writes blocks according to shape
        fn rec(
            f: &mut Formatter<'_>,
            shape: &[usize],
            data: &[String],
            idx: &mut usize,
            indent: usize,
        ) -> std::fmt::Result {
            if shape.len() == 1 {
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
                f.write_str("[")?;
                let n = shape[0];
                let sub = &shape[1..];
                for i in 0..n {
                    if i > 0 {
                        // разделение между крупными блоками
                        if shape.len() >= 3 {
                            f.write_str("\n\n")?;
                        } else {
                            f.write_str("\n")?;
                        }
                        // отступ для нового блока
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

        let mut idx = 0usize;
        rec(f, &self.shape, &padded, &mut idx, 0)
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::{Tensor, tensor::base::compute_strides};

    #[test]
    fn transpose_view() {
        let a = tensor![
            [1, 2, 3],
            [4, 5, 6]
        ];
        let b= a.slice(&[0, 1],
             &[2,2]).unwrap();

        println!("{b}");
        println!("BAA: {:?}", a.storage_indices());
        println!("BAA: {:?}", b.storage_indices());

        let t = a.transpose();

        assert_eq!(t.get_shape(), &[3, 2]);
        assert!(!t.is_contiguous());

        assert_eq!(
            t.get_data(),
            vec![1, 4, 2, 5, 3, 6]
        );
    }

    #[test]
    fn matmul_test() {
        let a = tensor![[1,2], [3,4]];
        let b = tensor![[5,6], [7,8]];

        println!("{}", a.matmul(&b));
    }

    #[test]
    fn matmul_transpose_view() {
        let a = tensor![[1., 2.], [3., 4.]];
        let b = tensor![[5., 6.], [7., 8.]].transpose();

        let c = a.matmul(&b);
        assert_eq!(c, tensor![[17., 23.], [39., 53.]]);
    }

    #[test]
    fn matmul_batched() {
        let a = Tensor::from(vec![
            vec![vec![1., 2.], vec![3., 4.]],
            vec![vec![5., 6.], vec![7., 8.]],
        ]);
        let b = Tensor::from(vec![
            vec![vec![1., 0.], vec![0., 1.]],
            vec![vec![2., 0.], vec![0., 2.]],
        ]);

        let c = a.matmul(&b);

        assert_eq!(c, Tensor::from(vec![
            vec![vec![1., 2.], vec![3., 4.]],
            vec![vec![10., 12.], vec![14., 16.]],
        ]));
    }

    #[test]
    fn matmul_broadcast_batch() {
        let a = Tensor::from(vec![
            vec![vec![1., 2.], vec![3., 4.]],
            vec![vec![5., 6.], vec![7., 8.]],
        ]); // (2,2,2)

        let b = tensor![[1., 0.], [0., 1.]]; // (2,2)

        let c = a.matmul(&b);

        assert_eq!(c, a);
    }



    #[test]
    fn scalar_tensor_print() {
        let a = Tensor::scalar(1.0);
        println!("{}", a);
    }

    #[test]
    fn comp_strid() {
        println!("{:?}", compute_strides(&[3, 2, 2]));
    }

    #[test]
    fn sum_axis() {
        let a = Tensor::new((0..4).map(|x| x as f32).collect(), vec![2, 2]);
        
        let b = a.slice(&[1, 0], &[1, 2]).unwrap();

        let what = b.sum_axis_keepdim(0);
        assert_eq!(what, tensor![[2.0, 3.0]]);

        let x = tensor![[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]];

        assert_eq!(x.sum_axis(0), tensor![5.0, 7.0, 9.0]);
        assert_eq!(x.sum_axis(1), tensor![6.0, 15.0]);
    }

}

/*
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
}
///A `Tensor` represents a multi-dimensional mathematical structure used for
/// numerical computations and machine learning operations.
///
/// Reference: [nreHieW](https://github.com/nreHieW/r-nn/blob/main/src/core/tensor/mod.rs)
///
/// The `Tensor` struct uses flat vector storage with shape information to represent
/// multi-dimensional data. All mathematical operations are implemented without borrowing.
///
/// # Example
/// ```rust
/// use tensorrs::linalg::Tensor;
///
/// // Create a 2x3 tensor filled with zeros
/// let tensor = Tensor::from_num(0.0, vec![2, 3]);
/// println!("Shape: {:?}", tensor.get_shape());
/// println!("Data: {:?}", tensor.get_data());
/// ```
/// Reference: [nreHieW](https://github.com/nreHieW/r-nn/blob/main/src/core/tensor/mod.rs)
#[derive(PartialEq, Eq, Debug)]
pub struct Tensor<T: Num> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    //pub(crate) offset: isize
}
fn product(shape: &Vec<usize>) -> usize {
    shape.iter().product()
}
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut acc = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }
    strides
}
impl<T: Num> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            product(&shape),
            "!!!Inconsistent data and dimensions combination for tensor!!!"
        );
        let strides = compute_strides(&shape[..]);
        Self { data, shape, strides }
    }
    pub fn from_num(num: T, shape: Vec<usize>) -> Self {
        let mut mul = 1;
        for i in &shape {
            mul *= i;
        }
        let data = vec![num; mul];
        Self::new(data, shape)
    }

    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }

        let mut excepted_stride = 1;
        for i in (0..self.shape.len()).rev() {
            if self.strides[i] != excepted_stride {
                return false;
            }
            excepted_stride *= self.shape[i];
        }
        true
    }
    /// Return flatten vector
    pub fn get_data(&self) -> Vec<T> {
        self.data.clone()
    }
    /// Return shape of a tensor
    pub fn get_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    pub fn reshape(&self, new_shape:Vec<usize>) -> Self {
        assert_eq!(product(&new_shape), product(&self.shape),
         "!!!Reshape size mismatch!!!");
        assert!(self.is_contiguous(),
         "!!!Non-contiguous reshape not supported!!!");
        Self {
            data: self.data.clone(),
            strides: compute_strides(&new_shape),
            shape: new_shape,
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
            Self::new(
                self.data.clone(),
                self.shape.iter().filter(|&x| *x != 1).cloned().collect(),
            )
        } else {
            Self::new(
                self.data.clone(),
                self
                    .shape
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != dim as usize)
                    .map(|(_, v)| *v)
                    .collect()
            )
        }
    }
    pub fn unsqueeze(&self, dim: usize) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);
        Self::new(
            self.data.clone(),
            new_shape
        )
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
        Tensor::new(ans, new_shape)
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
        Self::new(result_data, new_shape)
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
    /// Performs convolution with the kernel (valid padding).
    ///
    /// # Panics
    /// If kernel dimensions exceed input dimensions.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let input = Tensor::new(vec![1,2,3,4,5,6,7,8,9], vec![3,3]);
    /// let kernel = Tensor::new(vec![1,0,0,1], vec![2,2]);
    /// let conv = input.convolve(&kernel);
    /// ```
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
        Tensor::new(output_data, output_shape)
    }

    /// Performs convolution with zero padding (same size output).
    ///
    /// # Panics
    /// If tensor dimensions don't match.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let input = Tensor::new(vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0], vec![3,3]);
    /// let kernel = Tensor::from_num(1.0/9.0, vec![3,3]);
    /// let conv = input.conv_zero(&kernel);
    /// ```
    pub fn conv_zero(&self, kernel: &Tensor<T>) -> Tensor<T> {
        assert_eq!(self.shape.len(), kernel.shape.len(), "!!!Dimensions must match for convolution!!!");
        let ndim = self.shape.len();
        let pads: Vec<usize> = kernel.shape.iter().map(|&k| k / 2).collect();
        let output_shape = self.shape.clone();
        let output_size = product(&output_shape);
        let mut output_data = vec![T::default(); output_size];
        output_data
            .par_iter_mut()
            .enumerate()
            .for_each(|(output_index, output_value)| {
                let output_coords = self.index_to_coords(output_index, &output_shape);
                let mut sum = T::default();
                for kernel_index in 0..kernel.data.len() {
                    let kernel_coords = kernel.index_to_coords(kernel_index, &kernel.shape);
                    let mut input_coords_i32: Vec<i32> = Vec::with_capacity(ndim);
                    for d in 0..ndim {
                        let ic = output_coords[d] as i32 + kernel_coords[d] as i32 - pads[d] as i32;
                        input_coords_i32.push(ic);
                    }
                    let in_bounds = input_coords_i32.iter().enumerate().all(|(d, &ic)| ic >= 0 && ic < self.shape[d] as i32);
                    if in_bounds {
                        let input_coords: Vec<usize> = input_coords_i32.iter().map(|&ic| ic as usize).collect();
                        let input_index = self.coords_to_index(&input_coords);
                        sum += self.data[input_index] * kernel.data[kernel_index];
                    }
                }
                *output_value = sum;
            });
        Tensor::new(
            output_data,
            output_shape
        )
    }

    /// Performs convolution with mirror (reflect) padding.
    ///
    /// # Panics
    /// If tensor dimensions don't match.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let input = Tensor::new(vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0], vec![3,3]);
    /// let kernel = Tensor::from_num(1.0/9.0, vec![3,3]);
    /// let conv = input.conv_with_mirror_padding(&kernel);
    /// ```
    pub fn conv_with_mirror_padding(&self, kernel: &Tensor<T>) -> Tensor<T> {
        assert_eq!(self.shape.len(), kernel.shape.len(), "!!!Dimensions must match for convolution!!!");
        let ndim = self.shape.len();
        let pads: Vec<usize> = kernel.shape.iter().map(|&k| k / 2).collect();
        let output_shape = self.shape.clone();
        let output_size = product(&output_shape);
        let mut output_data = vec![T::default(); output_size];
        output_data
            .par_iter_mut()
            .enumerate()
            .for_each(|(output_index, output_value)| {
                let output_coords = self.index_to_coords(output_index, &output_shape);
                let mut sum = T::default();
                for kernel_index in 0..kernel.data.len() {
                    let kernel_coords = self.index_to_coords(kernel_index, &kernel.shape);
                    let mut input_coords_i32: Vec<i32> = Vec::with_capacity(ndim);
                    for d in 0..ndim {
                        let ic = output_coords[d] as i32 + kernel_coords[d] as i32 - pads[d] as i32;
                        input_coords_i32.push(ic);
                    }
                    let mut actual_input_coords: Vec<usize> = Vec::with_capacity(ndim);
                    for d in 0..ndim {
                        actual_input_coords.push(self.mirror_index(input_coords_i32[d], self.shape[d]));
                    }
                    let input_index = self.coords_to_index(&actual_input_coords);
                    sum += self.data[input_index] * kernel.data[kernel_index];
                }
                *output_value = sum;
            });
        Tensor::new(
            output_data,
            output_shape
        )
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
        let mut index = 0;
        let mut multiplier = 1;
        for dim in (0..self.shape.len()).rev() {
            index += coords[dim] * multiplier;
            multiplier *= self.shape[dim];
        }
        index
    }
    fn mirror_index(&self, idx: i32, size: usize) -> usize {
        let size_i32 = size as i32;
        if idx < 0 {
            ((-idx - 1) as usize) % size
        } else if idx >= size_i32 {
            ((2 * size_i32 - idx - 1) as usize) % size
        } else {
            idx as usize
        }
    }

    pub fn map<F>(&self, f: F) -> Self
    where 
        F: Fn(T) -> T + Sync + Send,
    {
        let new_data = self.data
            .clone()
            .par_iter_mut()
            .map(|x| f(*x))
            .collect();
        Tensor::new(new_data, self.shape.clone())
    }

    pub fn zip_with<F>(&self, other: &Tensor<T>, f: F) -> Self
    where 
        F: Fn(T, T) -> T + Sync + Send,
    {
        assert_eq!(self.shape, other.shape, "!!!Shape must be equal!!!");
        let new_data = self
            .data
            .clone()
            .par_iter_mut()
            .enumerate()
            .map(|(i, x)| f(*x, other.data[i]))
            .collect();
        
        Tensor::new(new_data, self.shape.clone())
    }

    pub fn hadamard(&self, other: &Self) -> Self {
        self.zip_with(other, |x, y| x*y)
    }

    pub fn full_(&mut self, num: T) {
        self.data.fill(num);
    }
}

impl<T: Float> Tensor<T> {
    /// Creates a matrix with random numbers(between 0 and 1)
    /// This is achieved using the [Box-Muller transform](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform), which generates normally distributed random numbers
    /// from uniformly distributed random numbers.
    pub fn randn(shape: Vec<usize>) -> Self
    where
        Standard: Distribution<T> {
        let n = product(&shape);
        Self::new(
            vec![T::default(); n]
                .iter()
                .map(|_| {
                    (-T::from(2) * random::<T>().ln()).sqrt() // Bpx - Muller Method
                        * (T::from(2) * T::pi() * random::<T>()).cos()
                })
                .collect(),
            shape
        )
    }
    pub fn rand(shape: Vec<usize>) -> Self
    where
        Standard: Distribution<T> {
        Self::new(
            vec![T::default(); product(&shape)]
                .iter()
                .map(|_| random::<T>())
                .collect(),
            shape
        )
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
        let mut linear = 0;
        for i in 0..index.len() {
            assert!(index[i] < self.shape[i],
                "!!!Index out of bounds, shape: {} index: {}!!!",
                self.shape[i], index[i]);
            linear += index[i] * self.strides[i];
        }

        &self.data[linear]
    }
}
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
        let mut linear = 0;
        for i in 0..index.len() {
            assert!(index[i] < self.shape[i],
                "!!!Index out of bounds, shape: {} index: {}!!!",
                self.shape[i], index[i]);
            linear += index[i] * self.strides[i];
        }

        &mut self.data[linear]
    }
}

impl<T: Num> From<Matrix<T>> for Tensor<T> {
    fn from(value: Matrix<T>) -> Self {
        let data = value.data;
        let shape = vec![value.rows, value.cols];
        Self::new(data, shape)
    }
}
impl<T: Num> From<Vector<T>> for Tensor<T> {
    fn from(value: Vector<T>) -> Self {
        let data: Vec<T> = value.into();
        let shape = vec![data.len()];
        Self::new(data, shape)
    }
}
impl<T:Num> From<Vec<T>> for Tensor<T> {
    fn from(value: Vec<T>) -> Self {
        let shape = vec![value.len()];
        Self::new(value, shape)
    }
}
impl<T:Num> From<Vec<Vec<T>>> for Tensor<T> {
    fn from(value: Vec<Vec<T>>) -> Self {
        let rows = value.len();
        let cols = value.first().map_or(0, |row| row.len());
        for row in value.iter().skip(1) {
            assert_eq!(row.len(), cols, "!!!All columns must be equal!!!");
        }
        assert!(
            !(rows != 0 && cols == 0),
            "!!!Invalid matrix dimensions. Multiple empty rows!!!"
        );
        let data = value.into_iter().flatten().collect();
        Self::new(data, vec![rows, cols])
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
                layer.len(), d2,
                "!!!Inconsistent d2 (rows) in layer {}: expected {d2}, got {}!!!",
                i, layer.len()
            );
            for (j, row) in layer.iter().enumerate() {
                assert_eq!(
                    row.len(),d3,
                    "!!!Inconsistent d3 (columns) at layer {}, row {}: expected {}, got {}!!!",
                    i, j,d3,row.len()
                );
            }
        }
        let flat = value.into_iter().flatten().flatten().collect();
        Self::new(flat, vec![d1, d2, d3])
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
            strides: self.strides.clone()
        }
    }
}
#[cfg(test)]
mod test {
    use crate::linalg::tensor::Tensor;
    use crate::linalg::{Matrix, Vector};
    use crate::matrix;
    #[test]
    fn macro_test() {
        let a = tensor![[1,2], [1,2], [1,2], [1,2]];
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
    #[test]
    fn conv_zeros() {
        let a = tensor![[1.0,2.0,3.0],
                [4.0,5.0,6.0],
                [7.9, 8.0, 9.0]];
        let krnl = Tensor::from_num(1.0/9.0, vec![3,3]);
        println!("{}", a.conv_zero(&krnl));

        let a = matrix![[1.0,2.0,3.0],
                [4.0,5.0,6.0],
                [7.9, 8.0, 9.0]];
        
        let krnl = Matrix::from_num(1.0/9.0, 3, 3);
        println!("{}", a.conv_zero(&krnl));
    }
}*/