use rand::random;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::{sync::Arc, usize};
//use rayon::prelude::*;
use crate::{Float, Num, linalg::{Vector, broadcast_shape}};
use std::ops::{Index, IndexMut};
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

/// Builds a [`Tensor`] from an array literal.
///
/// # Example
/// ```
/// use tensorrs::tensor;
///
/// let a = tensor![1.0, 2.0, 3.0];          // shape [3]
/// let b = tensor![[1.0, 2.0], [3.0, 4.0]]; // shape [2, 2]
/// let c = tensor![[[1.0, 2.0]], [[3.0, 4.0]]]; // shape [2, 1, 2]
///
/// assert_eq!(a.get_shape(), vec![3]);
/// assert_eq!(b.get_shape(), vec![2, 2]);
/// assert_eq!(c.get_shape(), vec![2, 1, 2]);
/// ```
///
/// # Notes
/// Literals of rank 1, 2 and 3 are supported. For higher ranks build the tensor
/// from a flat vector with [`Tensor::new`] and an explicit shape.
#[macro_export]
macro_rules! tensor {
    // 3D
    ( $( [ $( [ $( $x:expr ),* $(,)? ] ),* $(,)? ] ),* $(,)? ) => {
        $crate::linalg::Tensor::from(vec![
            $(
                vec![
                    $(
                        vec![ $( $x ),* ],
                    )*
                ],
            )*
        ])
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

/// A `Tensor` represents a multi-dimensional mathematical structure used for
/// numerical computations and machine learning operations.
///
/// # Example
/// ```
/// use tensorrs::linalg::Tensor;
///
/// // A 2x3 tensor filled with zeros
/// let a = Tensor::from_num(0.0, vec![2, 3]);
/// assert_eq!(a.get_shape(), vec![2, 3]);
///
/// // Reshaping and transposing are free: they only rearrange metadata
/// let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
/// assert_eq!(b.transpose().get_data(), vec![1.0, 3.0, 2.0, 4.0]);
/// ```
///
/// # Layout
/// The elements live in a single flat buffer shared through an [`Arc`], and are
/// interpreted through three fields:
/// * `shape` — the size of every dimension;
/// * `strides` — how many elements to step in the buffer per dimension;
/// * `offset` — the buffer index of the first element.
///
/// Because of that, [`Tensor::slice`], [`Tensor::permute`], [`Tensor::reshape`] and
/// friends return *views*: they share the elements with the original tensor and
/// copy nothing. Methods that must produce packed data — such as
/// [`Tensor::get_data`] — repack it on the fly.
///
/// Reference: [nreHieW](https://github.com/nreHieW/r-nn/blob/main/src/core/tensor/mod.rs)
#[derive(Debug, PartialEq, Eq)]
pub struct Tensor<T> {
    pub(crate) storage: Arc<Storage<T>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<isize>,
    pub(crate) offset: isize
}

impl<T: Clone> Tensor<T> {
    /// Creates a tensor from a flat data vector and a shape.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    /// assert_eq!(a.get_shape(), vec![2, 3]);
    /// ```
    ///
    /// # Arguments
    /// * `data` — the elements in row-major order.
    /// * `shape` — the size of every dimension.
    ///
    /// # Panics
    /// If `data.len()` differs from the product of `shape`.
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        /*if data.len() == 1 {
            return Self::scalar(data[0].clone());
        }*/
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

    /// Creates a single-element tensor of shape `[1]`.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::scalar(5.0);
    /// assert!(a.is_scalar());
    /// assert_eq!(a.item(), 5.0);
    /// ```
    ///
    /// # Notes
    /// Loss values and other reduced results are represented this way, so they
    /// stay usable in tensor arithmetic.
    pub fn scalar(num: T) -> Self {
        Self {
            storage: Arc::new(Storage { data: vec![num] }),
            shape: vec![1],
            strides: vec![0],
            offset: 0, 
        }
    }

    /// Returns the single value held by the tensor.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let loss = Tensor::scalar(0.25);
    /// assert_eq!(loss.item(), 0.25);
    /// ```
    ///
    /// # Panics
    /// If the tensor holds more than one element.
    pub fn item(&self) -> T {
        assert_eq!(self.numel(), 1, "!!!item() requires numel == 1!!!");
        self.storage.data[self.offset as usize].clone()
    }

    /// Returns the elements as a flat vector in logical (row-major) order.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
    /// assert_eq!(a.get_data(), vec![1, 2, 3, 4]);
    /// assert_eq!(a.transpose().get_data(), vec![1, 3, 2, 4]);
    /// ```
    ///
    /// # Notes
    /// The data is always copied and repacked, so the result is contiguous even
    /// when the tensor itself is a strided view.
    pub fn get_data(&self) -> Vec<T> {
        self.packed_data()
    }
    
    /// Returns the shape of the tensor.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::from_num(0.0, vec![2, 3, 4]);
    /// assert_eq!(a.get_shape(), vec![2, 3, 4]);
    /// ```
    pub fn get_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Returns a reference to the element at `indices`.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
    /// assert_eq!(a.get(&[1, 0]), Some(&3));
    /// assert_eq!(a.get(&[2, 0]), None); // out of range
    /// assert_eq!(a.get(&[0]), None);    // wrong number of indices
    /// ```
    ///
    /// # Returns
    /// [`None`] if `indices` does not have one entry per dimension, or if any
    /// index is out of range.
    ///
    /// # Notes
    /// Unlike indexing with `[..]`, this never panics.
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

    /// Returns a view starting at `start_indices` and covering `shape`.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    /// let row = a.slice(&[1, 0], &[1, 3]).unwrap();
    /// assert_eq!(row.get_data(), vec![4, 5, 6]);
    /// ```
    ///
    /// # Arguments
    /// * `start_indices` — the corner the slice starts at, one index per dimension.
    /// * `shape` — the size of the slice, one entry per dimension.
    ///
    /// # Returns
    /// [`None`] if the arguments do not match the rank of the tensor, or if the
    /// slice leaves its bounds.
    ///
    /// # Notes
    /// Nothing is copied — the result shares storage with `self`.
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

    /// Returns a strided view: like [`Tensor::slice`], but taking every
    /// `steps[i]`-th element.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![6]);
    ///
    /// let every_second = a.step_slice(&[0], &[3], &[2]).unwrap();
    /// assert_eq!(every_second.get_data(), vec![1, 3, 5]);
    ///
    /// let reversed = a.step_slice(&[5], &[6], &[-1]).unwrap();
    /// assert_eq!(reversed.get_data(), vec![6, 5, 4, 3, 2, 1]);
    /// ```
    ///
    /// # Arguments
    /// * `start_indices` — the first index taken in every dimension.
    /// * `shape` — how many elements to take in every dimension.
    /// * `steps` — the step between two taken elements; negative values walk backwards.
    ///
    /// # Returns
    /// [`None`] if the arguments do not match the rank of the tensor, or if the
    /// first or last element of any dimension falls outside it.
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

    /// Returns another view of the same data.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
    /// let b = a.shallow_copy();
    /// assert_eq!(b.get_data(), a.get_data());
    /// ```
    ///
    /// # Notes
    /// Only the shape/strides/offset metadata is duplicated; the elements are
    /// shared with `self`.
    pub fn shallow_copy(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }

    /// Returns a view with the same elements arranged in `new_shape`.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![6]);
    /// let b = a.reshape(vec![2, 3]);
    /// assert_eq!(b.get_shape(), vec![2, 3]);
    /// assert_eq!(b.get_data(), vec![1, 2, 3, 4, 5, 6]);
    /// ```
    ///
    /// # Panics
    /// * If `new_shape` holds a different number of elements than the current shape.
    /// * If the tensor is not contiguous. Build a new tensor from
    ///   [`Tensor::get_data`] in that case.
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

    /// Reorders the dimensions of the tensor.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    /// let b = a.permute(&[1, 0]).unwrap(); // the same as transpose()
    /// assert_eq!(b.get_shape(), vec![3, 2]);
    /// assert_eq!(b.get_data(), vec![1, 4, 2, 5, 3, 6]);
    /// ```
    ///
    /// # Arguments
    /// * `axes` — a permutation of `0..ndim`, where `axes[i]` is the old position
    ///   of the new `i`-th dimension.
    ///
    /// # Returns
    /// [`None`] if `axes` has the wrong length or repeats an axis.
    ///
    /// # Notes
    /// Only the strides are rearranged, so the result is a view and usually not
    /// contiguous.
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

    /// Swaps the two dimensions of a 2D tensor.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    /// assert_eq!(a.transpose().get_shape(), vec![3, 2]);
    /// ```
    ///
    /// # Panics
    /// If the tensor is not 2D — use [`Tensor::permute`] for higher ranks.
    pub fn transpose(&self) -> Self {
        assert_eq!(self.shape.len(), 2, "!!!transpose(): only 2D tensors supported!!!
        \nTry to use permute");
        self.permute(&[1, 0]).unwrap()
    }

    pub(crate) fn broadcast_to(&self, target_shape: &[usize]) -> Option<Self> {
        let src_ndim = self.shape.len();
        let dst_ndim = target_shape.len();
        
        if src_ndim > dst_ndim {
            eprintln!("SRC ndim > DST NDIM");
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
                eprintln!("src_dim != 1 || src_dim !- dst_dim");
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

    /// Reports whether the elements lie in the buffer in row-major order without gaps.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
    /// assert!(a.is_contiguous());
    /// assert!(!a.transpose().is_contiguous());
    /// ```
    ///
    /// # Notes
    /// A non-contiguous tensor cannot be reshaped, since [`Tensor::reshape`] only
    /// recomputes strides.
    ///
    /// A dimension of size `1` is visited exactly once, so its stride is ignored
    /// by the check — scalars keep a stride of `0` and still count as contiguous.
    pub fn is_contiguous(&self) -> bool {
        let mut excepted_stride = 1;
        for i in (0..self.shape.len()).rev() {
            // A dimension of size 1 is visited exactly once, so its stride is
            // never used and may hold anything (scalars keep a stride of 0).
            if self.shape[i] == 1 {
                continue;
            }
            if self.strides[i] != excepted_stride {
                return false;
            }
            excepted_stride *= self.shape[i] as isize;
        }
        true
    }

    /// Reports whether the tensor holds exactly one element.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// assert!(Tensor::scalar(1.0).is_scalar());
    /// assert!(!Tensor::from_num(1.0, vec![2, 2]).is_scalar());
    /// ```
    pub fn is_scalar(&self) -> bool {
        self.packed_data().len() == 1
    }

    pub(crate) fn can_inplace(&self) -> bool {
        self.is_contiguous()
            && Arc::strong_count(&self.storage) == 1
            && !self.strides.iter().any(|&s| s == 0)
    }
}

impl<T: Clone> Index<&[usize]> for Tensor<T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        if index.len() != self.shape.len() {
            panic!("!!!Incompatible shapes!!!");
        }

        let mut idx: isize = self.offset;

        for (i, &ind) in index.iter().enumerate() {
            if ind >= self.shape[i] {
                panic!("Index out of bounds");
            }
            idx += self.strides[i] * ind as isize;
        }

        &self.storage.data[idx as usize]
    }
}

impl<T: Clone> IndexMut<&[usize]> for Tensor<T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        if index.len() != self.shape.len() {
            panic!("!!!Incompatible shapes!!!");
        }

        let mut idx: isize = self.offset;

        for (i, &ind) in index.iter().enumerate() {
            if ind >= self.shape[i] {
                panic!("Index out of bounds");
            }
            idx += self.strides[i] * ind as isize;
        }

        // copy-on-write через Arc::make_mut
        let storage = Arc::make_mut(&mut self.storage); // клонирует Storage при наличии других Arc
        &mut storage.data[idx as usize]
    }
}

impl<T: Num> Tensor<T> {
    /// Creates a tensor of the given shape filled with one value.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let zeros = Tensor::from_num(0.0, vec![2, 3]);
    /// assert_eq!(zeros.get_data(), vec![0.0; 6]);
    /// ```
    pub fn from_num(num: T, shape: Vec<usize>) -> Self {
        Self::new(vec![num; product(&shape[..])], shape)
    }

    /// Overwrites every element with `num`, in place.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let mut a = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
    /// a.full_(0);
    /// assert_eq!(a.get_data(), vec![0, 0, 0, 0]);
    /// ```
    ///
    /// # Notes
    /// The whole underlying buffer is filled, not only the part visible through
    /// this view. If the storage is shared it is cloned first, so other tensors
    /// keep their values.
    pub fn full_(&mut self, num: T) {
        let data = Arc::make_mut(&mut self.storage);
        data.data.fill(num);
    }

    /// Batched matrix multiplication.
    ///
    /// # Formula
    /// ```math
    /// C_{ij} = \sum_{k=1}^{K} A_{ik} B_{kj}
    /// ```
    /// Where $`A`$ is of shape $`M \times K`$ and $`B`$ is of shape $`K \times N`$
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]); // 2x3
    /// let b = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![3, 2]); // 3x2
    ///
    /// let c = a.matmul(&b);
    /// assert_eq!(c.get_shape(), vec![2, 2]);
    /// assert_eq!(c.get_data(), vec![22, 28, 49, 64]);
    /// ```
    ///
    /// # Arguments
    /// * `other` — the right-hand tensor of shape `(..., K, N)`, where `self` is
    ///   of shape `(..., M, K)`.
    ///
    /// # Returns
    /// A tensor of shape `(..., M, N)`. The leading batch dimensions of both
    /// operands are broadcast against each other.
    ///
    /// # Panics
    /// * If either tensor has fewer than two dimensions.
    /// * If the inner dimensions disagree (`self.shape[-1] != other.shape[-2]`).
    /// * If the batch dimensions are not broadcastable.
    ///
    /// # Notes
    /// The output is filled in parallel with
    /// [rayon](https://crates.io/crates/rayon), and both operands are read
    /// through their strides, so transposed views need no repacking.
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

    /// Sums every element into a scalar tensor.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
    /// assert_eq!(a.sum().item(), 10);
    /// ```
    pub fn sum(&self) -> Tensor<T> {
        let mut ans = T::default();
        for i in self.packed_data().iter() {
            ans += *i
        }
        Tensor::scalar(ans)
    }

    /// Sums along `axis` and drops that dimension.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    /// assert_eq!(a.sum_axis(0).get_data(), vec![5, 7, 9]); // shape [3]
    /// assert_eq!(a.sum_axis(1).get_data(), vec![6, 15]);   // shape [2]
    /// ```
    ///
    /// # Arguments
    /// * `axis` — the dimension to sum over.
    ///
    /// # Panics
    /// If `axis` is not smaller than the rank of the tensor.
    ///
    /// # Notes
    /// Use [`Tensor::sum_axis_keepdim`] to keep the reduced dimension as `1`.
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

    /// Sums along `axis` and keeps that dimension with size `1`.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    ///
    /// let s = a.sum_axis_keepdim(1);
    /// assert_eq!(s.get_shape(), vec![2, 1]);
    /// assert_eq!(s.get_data(), vec![6, 15]);
    /// ```
    ///
    /// # Arguments
    /// * `axis` — the dimension to sum over.
    ///
    /// # Panics
    /// If `axis` is not smaller than the rank of the tensor.
    ///
    /// # Notes
    /// Keeping the dimension makes the result broadcastable against the original
    /// tensor, which is what normalisation layers need.
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

    /// Sums the tensor down to `target`, undoing a broadcast.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// // A bias of shape [3] was broadcast over a batch of 2 rows;
    /// // its gradient has to be summed back over the batch.
    /// let grad = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    /// let bias_grad = grad.reduce_to_shape(&[3]).unwrap();
    /// assert_eq!(bias_grad.get_data(), vec![5, 7, 9]);
    /// ```
    ///
    /// # Arguments
    /// * `target` — the shape to reduce to.
    ///
    /// # Returns
    /// [`None`] if the tensor cannot be reduced to `target`.
    ///
    /// # Notes
    /// Kept for backward compatibility; forwards to
    /// [`Tensor::reduce_broadcast_grad`].
    pub fn reduce_to_shape(&self, target: &[usize]) -> Option<Tensor<T>> {
        // For backward compatibility, try reduce_broadcast first
        self.reduce_broadcast_grad(target)
    }
    
    /// Reduces a gradient to `target` by summing over the dimensions that were
    /// broadcast in the forward pass.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// // forward: [3] + [2, 3] -> [2, 3], so the backward pass has to
    /// // turn a [2, 3] gradient back into a [3] one
    /// let grad = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    /// assert_eq!(grad.reduce_broadcast_grad(&[3]).unwrap().get_data(), vec![5.0, 7.0, 9.0]);
    ///
    /// // an axis the target keeps as 1 is summed but not dropped
    /// let g = grad.reduce_broadcast_grad(&[1, 3]).unwrap();
    /// assert_eq!(g.get_shape(), vec![1, 3]);
    /// ```
    ///
    /// # Arguments
    /// * `target` — the shape of the operand as it was before broadcasting.
    ///
    /// # Returns
    /// [`None`] if the tensor cannot be reduced to `target`.
    ///
    /// # Notes
    /// Leading dimensions are summed away until the ranks match, then every
    /// dimension the target holds as `1` is summed with
    /// [`Tensor::sum_axis_keepdim`], so that the result has exactly the shape of
    /// `target`. This is what the autodiff engine calls to route gradients back
    /// through broadcasting.
    pub fn reduce_broadcast_grad(&self, target: &[usize]) -> Option<Tensor<T>> {
        // Lower rank than the target: prepend axes of size 1 so both ranks match,
        // then reduce as usual.
        if self.shape.len() < target.len() {
            if self.numel() == 1 {
                // A scalar broadcasts to any shape
                return Some(Tensor::from_num(self.item(), target.to_vec()));
            }

            let mut expanded_shape = vec![1; target.len() - self.shape.len()];
            expanded_shape.extend(self.shape.iter().copied());

            return Tensor::new(self.packed_data(), expanded_shape)
                .reduce_broadcast_grad(target);
        }

        let mut out = self.shallow_copy();

        // Sum away the leading axes that broadcasting prepended, until ranks match
        while out.shape.len() > target.len() {
            out = out.sum_axis(0);
        }

        // Ranks match now. Every axis the target holds as 1 was broadcast and has
        // to be summed, but the axis itself must stay so the ranks keep matching.
        for i in 0..target.len() {
            let s = out.shape[i];
            let t = target[i];

            if s == t {
                continue;
            } else if t == 1 {
                out = out.sum_axis_keepdim(i);
            } else {
                return None;
            }
        }

        Some(out)
    }

    /// Applies `f` to every element and returns a new tensor.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
    /// assert_eq!(a.map(|x| x * 2).get_data(), vec![2, 4, 6, 8]);
    /// ```
    ///
    /// # Arguments
    /// * `f` — the function applied to every element.
    ///
    /// # Notes
    /// The elements are repacked first, so the result is always contiguous.
    /// `f` runs in parallel and therefore has to be `Sync + Send`.
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T + Sync + Send,
    {
        let mut new_data = self.packed_data();
        new_data.par_iter_mut().for_each(|x| *x = f(*x));

        Self::new(new_data, self.shape.clone())
    }


    /// Multiplies two tensors element-wise and sums the products.
    ///
    /// # Formula
    /// ```math
    /// \langle a, b \rangle = \sum_{i=1}^{n} a_i b_i
    /// ```
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3], vec![3]);
    /// let b = Tensor::new(vec![4, 5, 6], vec![3]);
    /// assert_eq!(a.mul_sum(&b), 32); // 1*4 + 2*5 + 3*6
    /// ```
    ///
    /// # Arguments
    /// * `rhs` — a tensor whose shape is broadcastable against `self`.
    ///
    /// # Returns
    /// The sum as a plain `T`, not as a tensor.
    ///
    /// # Panics
    /// If the two shapes are not broadcastable.
    ///
    /// # Notes
    /// Runs sequentially below 1024 elements and in parallel above that. This is
    /// the inner loop of the convolutions.
    pub fn mul_sum(&self, rhs: &Self) -> T {
        if self.is_contiguous() && rhs.is_contiguous()  && self.shape == rhs.shape {
            return self.mul_sum_contiguous(rhs);
        }
        let shape = broadcast_shape(&self.shape[..], &rhs.shape[..])
        .expect("!!!Uncopatable shape!!!");

        let a_view = self.broadcast_to(&shape).expect("broadcast_to failed (bug)");
        let b_view = rhs.broadcast_to(&shape).expect("broadcast_to failed (bug)");

        let a_data = &a_view.storage.data;
        let b_data = &b_view.storage.data;
        let a_idx = a_view.storage_indices();
        let b_idx = b_view.storage_indices();

        let n = product(&shape[..]);

        if n < 1024 {
            (0..n)
                .map(|i| a_data[a_idx[i]] * b_data[b_idx[i]])
                .sum()
        } else {
            (0..n)
                .into_par_iter()
                .map(|i| a_data[a_idx[i]] * b_data[b_idx[i]])
                .sum()
        }
    }

    fn mul_sum_contiguous(&self, rhs: &Self) -> T {
        // Both tensors are contiguous and share a shape, so each of them occupies
        // exactly `n` elements starting at its own offset.
        let n = self.numel();
        let a = &self.storage.data[self.offset as usize..self.offset as usize + n];
        let b = &rhs.storage.data[rhs.offset as usize..rhs.offset as usize + n];

        a.iter()
        .zip(b.iter())
        .map(|(x, y)| *x * *y)
        .sum()
    }

    /// Iterates over every sliding window of the tensor.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![3, 3]);
    /// let windows: Vec<_> = a.window(&[2, 2], &[1, 1]).collect();
    ///
    /// assert_eq!(windows.len(), 4);
    /// assert_eq!(windows[0].get_data(), vec![1, 2, 4, 5]);
    /// ```
    ///
    /// # Arguments
    /// * `kernel` — the window size per dimension.
    /// * `stride` — the step between two windows per dimension.
    ///
    /// # Returns
    /// An iterator of views, `(dim - kernel) / stride + 1` of them per dimension.
    ///
    /// # Panics
    /// If `kernel` or `stride` does not hold one entry per dimension, or if a
    /// window is larger than the tensor.
    ///
    /// # Notes
    /// The windows are views, so nothing is copied. Combined with
    /// [`Tensor::mul_sum`] this is how the convolutions are built.
    pub fn window<'a>(
        &'a self,
        kernel: &[usize],
        stride: &[usize],
    ) -> impl Iterator<Item = Tensor<T>> + 'a {
        assert_eq!(kernel.len(), self.shape.len());
        assert_eq!(stride.len(), self.shape.len());

        let kernel = kernel.to_vec();   // ← копия
        let stride = stride.to_vec();   // ← копия

        let out_shape: Vec<usize> = self.shape.iter()
            .zip(kernel.iter())
            .zip(stride.iter())
            .map(|((&dim, &k), &s)| {
                assert!(dim >= k);
                (dim - k) / s + 1
            })
            .collect();

        let total = product(&out_shape);

        let base_strides = self.strides.clone();
        let base_offset = self.offset;
        let storage = self.storage.clone();

        (0..total).map(move |idx| {
            let mut tmp = idx;
            let mut coord = vec![0; out_shape.len()];

            for i in (0..coord.len()).rev() {
                coord[i] = tmp % out_shape[i];
                tmp /= out_shape[i];
            }

            let mut offset = base_offset;
            for i in 0..coord.len() {
                offset += coord[i] as isize * stride[i] as isize * base_strides[i];
            }

            Tensor {
                storage: storage.clone(),
                shape: kernel.clone(),
                strides: base_strides.clone(),
                offset,
            }
        })
    }
}

impl<T: Float> Tensor<T> {
    /// Creates a tensor filled with samples of the standard normal distribution
    /// `N(0, 1)`.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let w = Tensor::<f32>::randn(vec![2, 3]);
    /// assert_eq!(w.get_shape(), vec![2, 3]);
    /// ```
    ///
    /// # Arguments
    /// * `shape` — the size of every dimension.
    ///
    /// # Notes
    /// Uses the [Box-Muller transform](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform),
    /// which turns two uniform numbers $`u_1, u_2 \sim U(0, 1)`$ into a normally
    /// distributed one:
    /// ```math
    /// z = \sqrt{-2 \ln u_1} \cdot \cos(2 \pi u_2)
    /// ```
    /// This is the usual choice for initialising weights.
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
    /// Creates a tensor filled with uniform random numbers between `0` and `1`.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::<f32>::rand(vec![2, 2]);
    /// assert_eq!(a.get_shape(), vec![2, 2]);
    /// assert!(a.get_data().iter().all(|&x| (0.0..=1.0).contains(&x)));
    /// ```
    ///
    /// # Arguments
    /// * `shape` — the size of every dimension.
    ///
    /// # Notes
    /// See [`Tensor::randn`] for normally distributed values.
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

    #[test]
    fn strided_1d_view_is_not_contiguous() {
        let a = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![6]);
        let strided = a.step_slice(&[0], &[3], &[2]).unwrap();

        assert_eq!(strided.get_data(), vec![1, 3, 5]);
        assert!(!strided.is_contiguous());

        // a scalar keeps a stride of 0 and still counts as contiguous
        assert!(Tensor::scalar(1).is_contiguous());
        // so does any axis of size 1
        assert!(a.reshape(vec![1, 6, 1]).is_contiguous());
    }

    #[test]
    fn mul_sum_respects_view_offset() {
        // a contiguous view that does not start at the beginning of the buffer
        let a = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![3, 2]);
        let row = a.slice(&[1, 0], &[1, 2]).unwrap();
        assert_eq!(row.get_data(), vec![3, 4]);
        assert!(row.is_contiguous());

        let ones = Tensor::new(vec![1, 1], vec![1, 2]);
        assert_eq!(row.mul_sum(&ones), 7);

        // the strided path has to agree with the contiguous one
        let strided = a.step_slice(&[0, 0], &[3, 1], &[1, 2]).unwrap();
        assert_eq!(strided.get_data(), vec![1, 3, 5]);
        assert_eq!(strided.mul_sum(&Tensor::from_num(1, vec![3, 1])), 9);
    }

    #[test]
    fn reduce_broadcast_grad_keeps_target_rank() {
        let grad = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);

        // rank reduction: [3] was broadcast over a batch of 2
        let g = grad.reduce_broadcast_grad(&[3]).unwrap();
        assert_eq!(g.get_shape(), vec![3]);
        assert_eq!(g.get_data(), vec![5, 7, 9]);

        // keepdim target: the axis must stay instead of being dropped
        let g = grad.reduce_broadcast_grad(&[1, 3]).unwrap();
        assert_eq!(g.get_shape(), vec![1, 3]);
        assert_eq!(g.get_data(), vec![5, 7, 9]);

        let g = grad.reduce_broadcast_grad(&[2, 1]).unwrap();
        assert_eq!(g.get_shape(), vec![2, 1]);
        assert_eq!(g.get_data(), vec![6, 15]);

        let g = grad.reduce_broadcast_grad(&[1, 1]).unwrap();
        assert_eq!(g.get_shape(), vec![1, 1]);
        assert_eq!(g.get_data(), vec![21]);

        // a leading axis added by broadcasting is summed away
        let batched = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]);
        let g = batched.reduce_broadcast_grad(&[2, 2]).unwrap();
        assert_eq!(g.get_shape(), vec![2, 2]);
        assert_eq!(g.get_data(), vec![6, 8, 10, 12]);

        // not a broadcast: honestly report failure instead of returning garbage
        assert!(grad.reduce_broadcast_grad(&[2, 2]).is_none());
    }

    #[test]
    fn tensor_macro_builds_3d() {
        let a = tensor![
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ];
        assert_eq!(a.get_shape(), vec![2, 2, 2]);
        assert_eq!(a.get_data(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // 1D and 2D still resolve to their own arms
        assert_eq!(tensor![1.0, 2.0, 3.0].get_shape(), vec![3]);
        assert_eq!(tensor![[1.0, 2.0], [3.0, 4.0]].get_shape(), vec![2, 2]);
    }
}
