use rand::random;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::{sync::Arc, usize};
//use rayon::prelude::*;
use crate::{Float, Num, linalg::{Vector, broadcast_shape, tensor::ops::PARALLEL_THRESHOLD}};
use std::ops::{Index, IndexMut};
use std::fmt::{Debug, Display, Formatter};
//use crate::linalg::{Matrix, Vector};
//use rayon::prelude::IntoParallelRefMutIterator;
use rand::distributions::{Distribution, Standard};

/// Folds an out-of-range index back inside by reflecting it off the edges.
///
/// The reflection repeats, so an index arbitrarily far outside still lands in
/// `0..n`: for `n = 3` the sequence `-2 -1 0 1 2 3 4` maps to `1 0 0 1 2 2 1`.
fn reflect(mut index: isize, n: usize) -> usize {
    let n = n as isize;
    if n == 0 {
        return 0;
    }

    loop {
        if index < 0 {
            index = -index - 1;
        } else if index >= n {
            index = 2 * n - index - 1;
        } else {
            return index as usize;
        }
    }
}

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
/// let a = tensor![1.0, 2.0, 3.0];              // shape [3]
/// let b = tensor![[1.0, 2.0], [3.0, 4.0]];     // shape [2, 2]
/// let c = tensor![[[1.0, 2.0]], [[3.0, 4.0]]]; // shape [2, 1, 2]
///
/// // one image, two channels, 1x2 - the layout Conv2d expects
/// let d = tensor![[[[1.0, 2.0]], [[3.0, 4.0]]]]; // shape [1, 2, 1, 2]
///
/// assert_eq!(a.get_shape(), vec![3]);
/// assert_eq!(b.get_shape(), vec![2, 2]);
/// assert_eq!(c.get_shape(), vec![2, 1, 2]);
/// assert_eq!(d.get_shape(), vec![1, 2, 1, 2]);
/// ```
///
/// # Panics
/// At compile time if the literal is ragged — every group on a level has to hold
/// the same number of elements.
///
/// # Notes
/// Literals of rank 1 to 4 are supported, four being what
/// [Conv2d](crate::nn::Conv2d) and the pooling layers work on:
/// `[batch, channels, height, width]`. Data is read in row-major order, the
/// outermost axis varying slowest.
///
/// Beyond rank 4 build the tensor from a flat vector with [`Tensor::new`] and an
/// explicit shape — a rank-5 literal fails to compile rather than silently
/// misreading the nesting.
#[macro_export]
macro_rules! tensor {
    // 4D - must come before 3D, being the more deeply nested pattern
    ( $( [ $( [ $( [ $( $x:expr ),* $(,)? ] ),* $(,)? ] ),* $(,)? ] ),* $(,)? ) => {
        $crate::linalg::Tensor::from(vec![
            $(
                vec![
                    $(
                        vec![
                            $(
                                vec![ $( $x ),* ],
                            )*
                        ],
                    )*
                ],
            )*
        ])
    };
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

    /// Takes the slice at `index` along `axis` and drops that dimension.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    ///
    /// assert_eq!(a.select(0, 1).get_shape(), vec![3]);
    /// assert_eq!(a.select(0, 1).get_data(), vec![4, 5, 6]); // second row
    /// assert_eq!(a.select(1, 2).get_data(), vec![3, 6]);    // third column
    /// ```
    ///
    /// # Arguments
    /// * `axis` — the dimension to index into.
    /// * `index` — the position along that dimension.
    ///
    /// # Panics
    /// If `axis` is out of range for the tensor, or `index` out of range for the axis.
    ///
    /// # Notes
    /// Unlike [Tensor::slice] the data is repacked, so the result is contiguous
    /// and owns its layout. Selecting from a 1-D tensor gives a tensor of shape
    /// `[1]` rather than a rank-0 one.
    ///
    /// [Tensor::stack] is the inverse: it puts the parts back together along a
    /// new axis.
    pub fn select(&self, axis: usize, index: usize) -> Self {
        assert!(axis < self.shape.len(),
            "!!!select(): axis {axis} is out of range for a tensor of rank {}!!!", self.shape.len());
        assert!(index < self.shape[axis],
            "!!!select(): index {index} is out of range for axis {axis} of size {}!!!", self.shape[axis]);

        let mut start = vec![0usize; self.shape.len()];
        start[axis] = index;

        let mut view_shape = self.shape.clone();
        view_shape[axis] = 1;

        let view = self
            .slice(&start, &view_shape)
            .expect("select(): slice went out of bounds (bug)");

        let mut out_shape = self.shape.clone();
        out_shape.remove(axis);
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        Self::new(view.packed_data(), out_shape)
    }

    /// Joins tensors of an equal shape along a new axis.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2], vec![2]);
    /// let b = Tensor::new(vec![3, 4], vec![2]);
    ///
    /// let s = Tensor::stack(&[a, b], 0);
    /// assert_eq!(s.get_shape(), vec![2, 2]);
    /// assert_eq!(s.get_data(), vec![1, 2, 3, 4]);
    /// ```
    ///
    /// # Arguments
    /// * `parts` — the tensors to join; all of them must share one shape.
    /// * `axis` — where to insert the new dimension, from `0` to the rank of the parts.
    ///
    /// # Returns
    /// A tensor whose shape is the shape of the parts with `parts.len()` inserted
    /// at `axis`.
    ///
    /// # Panics
    /// If `parts` is empty, the shapes disagree, or `axis` is past the rank of the parts.
    ///
    /// # Notes
    /// The inverse of [Tensor::select]: `Tensor::stack(&parts, axis).select(axis, i)`
    /// gives back `parts[i]`. Stacking a sequence of per-step results is what turns
    /// an [RNN](crate::nn::RNN) loop back into one tensor.
    pub fn stack(parts: &[Self], axis: usize) -> Self {
        assert!(!parts.is_empty(), "!!!stack(): needs at least one tensor!!!");

        let part_shape = parts[0].shape.clone();
        for (i, p) in parts.iter().enumerate() {
            assert_eq!(p.shape, part_shape,
                "!!!stack(): tensor {i} has shape {:?}, expected {:?}!!!", p.shape, part_shape);
        }
        assert!(axis <= part_shape.len(),
            "!!!stack(): axis {axis} is past the rank {} of the parts!!!", part_shape.len());

        let mut out_shape = part_shape.clone();
        out_shape.insert(axis, parts.len());

        // everything before `axis` is walked in the outer loop, everything from
        // `axis` on is copied in one contiguous run per part
        let outer: usize = part_shape[..axis].iter().product();
        let inner: usize = part_shape[axis..].iter().product();

        let packed: Vec<Vec<T>> = parts.iter().map(|p| p.packed_data()).collect();
        let mut data = Vec::with_capacity(outer * parts.len() * inner);

        for o in 0..outer {
            for part in &packed {
                data.extend_from_slice(&part[o * inner..(o + 1) * inner]);
            }
        }

        Self::new(data, out_shape)
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

        // copy-on-write by way of Arc::make_mut
        let storage = Arc::make_mut(&mut self.storage); // clones the Storage if any other Arc holds it
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
    /// Scatters `self` back into a zero tensor of `orig_shape` at `index` along
    /// `axis` — the backward pass of [Tensor::select].
    pub(crate) fn scatter_select(&self, orig_shape: &[usize], axis: usize, index: usize) -> Tensor<T> {
        let dim = orig_shape[axis];
        let outer: usize = orig_shape[..axis].iter().product();
        let inner: usize = orig_shape[axis + 1..].iter().product();

        let src = self.packed_data();
        let mut data = vec![T::default(); product(orig_shape)];

        for o in 0..outer {
            let dst_at = (o * dim + index) * inner;
            let src_at = o * inner;
            data[dst_at..dst_at + inner].copy_from_slice(&src[src_at..src_at + inner]);
        }

        Tensor::new(data, orig_shape.to_vec())
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

        // The offset of the batch part: it depends neither on k nor on the
        // position within the last two axes, so it is worked out once per
        // output element.
        let batch_offset = |t: &Tensor<T>, out_coords: &[usize]| -> isize {
            let t_ndim = t.shape.len();
            let out_ndim = out_shape.len();
            let mut off = t.offset;

            for axis in 0..t_ndim - 2 {
                let out_axis = out_ndim - t_ndim + axis;
                let coord = if t.shape[axis] == 1 { 0 } else { out_coords[out_axis] };
                off += t.strides[axis] * (coord as isize);
            }

            off
        };

        // prepare output buffer
        let mut out_data = vec![T::default(); out_len];

        let a_row_axis = a_ndim - 2;
        let a_k_axis = a_ndim - 1;
        let b_k_axis = b_ndim - 2;
        let b_col_axis = b_ndim - 1;

        let compute = |coords: &mut Vec<usize>, out_idx: usize, slot: &mut T| {
            let mut rem = out_idx;
            for axis in 0..out_shape.len() {
                coords[axis] = rem / suffix[axis];
                rem %= suffix[axis];
            }

            let i = coords[out_shape.len() - 2];
            let j = coords[out_shape.len() - 1];

            // everything independent of k is hoisted out of the inner loop
            let a_base = batch_offset(self, coords) + self.strides[a_row_axis] * (i as isize);
            let b_base = batch_offset(other, coords) + other.strides[b_col_axis] * (j as isize);
            let a_step = self.strides[a_k_axis];
            let b_step = other.strides[b_k_axis];

            let mut acc = T::default();
            for k in 0..a_k {
                let a_off = (a_base + a_step * (k as isize)) as usize;
                let b_off = (b_base + b_step * (k as isize)) as usize;
                acc = acc + self.storage.data[a_off] * other.storage.data[b_off];
            }

            *slot = acc;
        };

        if out_len < PARALLEL_THRESHOLD {
            // a small output: one coordinate buffer for the whole pass, no threads involved
            let mut coords = vec![0usize; out_shape.len()];
            for (out_idx, slot) in out_data.iter_mut().enumerate() {
                compute(&mut coords, out_idx, slot);
            }
        } else {
            // for_each_init rather than for_each: the buffer is built once per
            // worker thread instead of once per output element
            out_data.par_iter_mut().enumerate().for_each_init(
                || vec![0usize; out_shape.len()],
                |coords, (out_idx, slot)| compute(coords, out_idx, slot),
            );
        }

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

        if out_data.len() < PARALLEL_THRESHOLD {

            out_data.iter_mut().enumerate().for_each(|(out_idx, x)|{
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

        } else {

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

        }

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
    /// Picks rows out of a table by index.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// // a table of 4 rows, 2 values each
    /// let table = Tensor::new(vec![10, 11, 20, 21, 30, 31, 40, 41], vec![4, 2]);
    ///
    /// let picked = table.gather_rows(&[2, 0, 2]);
    /// assert_eq!(picked.get_shape(), vec![3, 2]);
    /// assert_eq!(picked.get_data(), vec![30, 31, 10, 11, 30, 31]);
    /// ```
    ///
    /// # Arguments
    /// * `indices` — which rows to take, in order; repeats are allowed.
    ///
    /// # Returns
    /// A tensor of shape `[indices.len(), columns]`.
    ///
    /// # Panics
    /// If the tensor is not 2-D, or an index is past the end of the table.
    ///
    /// # Notes
    /// This is the lookup behind [Embedding](crate::nn::Embedding): a table of one
    /// row per token, and a sequence of ids picking rows out of it.
    /// [Tensor::scatter_add_rows] is the inverse used by its backward pass.
    pub fn gather_rows(&self, indices: &[usize]) -> Tensor<T> {
        assert_eq!(self.shape.len(), 2,
            "!!!gather_rows() expects a 2-D table, got {:?}!!!", self.shape);

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let src = self.packed_data();

        let mut data = vec![T::default(); indices.len() * cols];

        for (slot, &row) in indices.iter().enumerate() {
            assert!(row < rows,
                "!!!gather_rows(): index {row} is past a table of {rows} rows!!!");

            let from = row * cols;
            let to = slot * cols;
            data[to..to + cols].copy_from_slice(&src[from..from + cols]);
        }

        Tensor::new(data, vec![indices.len(), cols])
    }

    /// Adds rows back into a zero table at the positions they were taken from —
    /// the inverse of [Tensor::gather_rows].
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let grad = Tensor::new(vec![1, 1, 2, 2, 3, 3], vec![3, 2]);
    ///
    /// // row 0 was picked twice, so it collects from both
    /// let table = grad.scatter_add_rows(&[0, 1, 0], 3);
    /// assert_eq!(table.get_shape(), vec![3, 2]);
    /// assert_eq!(table.get_data(), vec![4, 4, 2, 2, 0, 0]);
    /// ```
    ///
    /// # Arguments
    /// * `indices` — the same indices [Tensor::gather_rows] was given.
    /// * `rows` — how many rows the table has.
    ///
    /// # Panics
    /// If `self` does not have one row per index, or an index is out of range.
    ///
    /// # Notes
    /// The accumulation is the whole point: a token that occurs several times in a
    /// batch takes a gradient from every occurrence.
    pub fn scatter_add_rows(&self, indices: &[usize], rows: usize) -> Tensor<T> {
        assert_eq!(self.shape.len(), 2,
            "!!!scatter_add_rows() expects a 2-D tensor, got {:?}!!!", self.shape);
        assert_eq!(self.shape[0], indices.len(),
            "!!!scatter_add_rows(): {} rows for {} indices!!!", self.shape[0], indices.len());

        let cols = self.shape[1];
        let src = self.packed_data();
        let mut data = vec![T::default(); rows * cols];

        for (slot, &row) in indices.iter().enumerate() {
            assert!(row < rows,
                "!!!scatter_add_rows(): index {row} is past a table of {rows} rows!!!");

            let to = row * cols;
            let from = slot * cols;
            for c in 0..cols {
                data[to + c] += src[from + c];
            }
        }

        Tensor::new(data, vec![rows, cols])
    }

    /// Takes the largest value along `axis` and drops that dimension.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 5, 3, 4, 2, 6], vec![2, 3]);
    /// assert_eq!(a.max_axis(1).get_data(), vec![5, 6]);
    /// assert_eq!(a.max_axis(0).get_data(), vec![4, 5, 6]);
    /// ```
    ///
    /// # Arguments
    /// * `axis` — the dimension to reduce.
    ///
    /// # Panics
    /// If `axis` is not smaller than the rank of the tensor, or the axis is empty.
    ///
    /// # Notes
    /// [Tensor::argmax_axis] reports where those values came from, which is what
    /// the backward pass of a max needs — see [MaxPool2d](crate::nn::MaxPool2d).
    /// Surrounds the tensor with zeros.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
    /// let padded = a.pad_zero(&[(1, 1), (1, 1)]);
    ///
    /// assert_eq!(padded.get_shape(), vec![4, 4]);
    /// assert_eq!(padded.get_data(), vec![
    ///     0, 0, 0, 0,
    ///     0, 1, 2, 0,
    ///     0, 3, 4, 0,
    ///     0, 0, 0, 0,
    /// ]);
    /// ```
    ///
    /// # Arguments
    /// * `pad` — how much to add before and after, one `(before, after)` pair per
    ///   dimension.
    ///
    /// # Panics
    /// If `pad` does not hold one pair per dimension.
    ///
    /// # Notes
    /// Zero padding treats everything outside the tensor as empty, which is the
    /// usual choice inside a network. [Tensor::pad_mirror] instead continues the
    /// data, which suits image filtering where a black border would be an artefact.
    pub fn pad_zero(&self, pad: &[(usize, usize)]) -> Tensor<T> {
        self.padded(pad, false)
    }

    /// Surrounds the tensor with a mirror image of its own edges.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 2, 3], vec![3]);
    /// // the edge values are reflected outwards: 1 | 1 2 3 | 3
    /// assert_eq!(a.pad_mirror(&[(1, 1)]).get_data(), vec![1, 1, 2, 3, 3]);
    /// ```
    ///
    /// # Arguments
    /// * `pad` — how much to add before and after, one `(before, after)` pair per
    ///   dimension.
    ///
    /// # Panics
    /// If `pad` does not hold one pair per dimension.
    ///
    /// # Notes
    /// The reflection repeats when the padding is wider than the tensor, so any
    /// amount is valid. Unlike [Tensor::pad_zero] this introduces no edge, which
    /// keeps a filter from seeing a step that is not in the data.
    pub fn pad_mirror(&self, pad: &[(usize, usize)]) -> Tensor<T> {
        self.padded(pad, true)
    }

    /// Shared body of [Tensor::pad_zero] and [Tensor::pad_mirror].
    fn padded(&self, pad: &[(usize, usize)], mirror: bool) -> Tensor<T> {
        let ndim = self.shape.len();
        assert_eq!(pad.len(), ndim,
            "!!!pad expects one (before, after) pair per dimension: got {} for rank {ndim}!!!",
            pad.len());

        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .zip(pad.iter())
            .map(|(&dim, &(before, after))| dim + before + after)
            .collect();

        let src = self.packed_data();

        // strides of the packed source, which is contiguous by construction
        let mut src_strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            src_strides[i] = src_strides[i + 1] * self.shape[i + 1];
        }

        let mut data = vec![T::default(); product(&out_shape[..])];

        if data.len() < PARALLEL_THRESHOLD {

            data.iter_mut().enumerate().for_each(|(out_idx, slot)| {
            let mut rem = out_idx;
            let mut src_at = 0usize;
            let mut inside = true;

            for i in (0..ndim).rev() {
                let coord = (rem % out_shape[i]) as isize - pad[i].0 as isize;
                rem /= out_shape[i];

                let source = if mirror {
                    reflect(coord, self.shape[i])
                } else if coord < 0 || coord >= self.shape[i] as isize {
                    inside = false;
                    0
                } else {
                    coord as usize
                };

                src_at += source * src_strides[i];
            }

            if inside {
                *slot = src[src_at].clone();
            }
        });

        } else {

            data.par_iter_mut().enumerate().for_each(|(out_idx, slot)| {
            let mut rem = out_idx;
            let mut src_at = 0usize;
            let mut inside = true;

            for i in (0..ndim).rev() {
                let coord = (rem % out_shape[i]) as isize - pad[i].0 as isize;
                rem /= out_shape[i];

                let source = if mirror {
                    reflect(coord, self.shape[i])
                } else if coord < 0 || coord >= self.shape[i] as isize {
                    inside = false;
                    0
                } else {
                    coord as usize
                };

                src_at += source * src_strides[i];
            }

            if inside {
                *slot = src[src_at].clone();
            }
        });

        }

        Tensor::new(data, out_shape)
    }

    pub fn max_axis(&self, axis: usize) -> Tensor<T> {
        let (out_shape, reduce_dim) = self.reduce_dims(axis);
        let mut out_data = vec![T::default(); product(&out_shape[..])];

        if out_data.len() < PARALLEL_THRESHOLD {

            out_data.iter_mut().enumerate().for_each(|(out_idx, x)| {
            let base = self.reduce_base_offset(out_idx, &out_shape, axis);
            let stride = self.strides[axis] as usize;

            let mut best = self.storage.data[base].clone();
            for k in 1..reduce_dim {
                let value = self.storage.data[base + k * stride].clone();
                if value > best {
                    best = value;
                }
            }

            *x = best;
        });

        } else {

            out_data.par_iter_mut().enumerate().for_each(|(out_idx, x)| {
            let base = self.reduce_base_offset(out_idx, &out_shape, axis);
            let stride = self.strides[axis] as usize;

            let mut best = self.storage.data[base].clone();
            for k in 1..reduce_dim {
                let value = self.storage.data[base + k * stride].clone();
                if value > best {
                    best = value;
                }
            }

            *x = best;
        });

        }

        Tensor::new(out_data, out_shape)
    }

    /// Reports where along `axis` the largest value sits.
    ///
    /// # Example
    /// ```
    /// use tensorrs::linalg::Tensor;
    ///
    /// let a = Tensor::new(vec![1, 5, 3, 4, 2, 6], vec![2, 3]);
    /// assert_eq!(a.argmax_axis(1), vec![1, 2]);
    /// ```
    ///
    /// # Arguments
    /// * `axis` — the dimension to reduce.
    ///
    /// # Returns
    /// One index per element of [Tensor::max_axis], in the same order. Ties go to
    /// the lowest index.
    ///
    /// # Panics
    /// If `axis` is not smaller than the rank of the tensor, or the axis is empty.
    pub fn argmax_axis(&self, axis: usize) -> Vec<usize> {
        let (out_shape, reduce_dim) = self.reduce_dims(axis);
        let mut out_data = vec![0usize; product(&out_shape[..])];

        if out_data.len() < PARALLEL_THRESHOLD {

            out_data.iter_mut().enumerate().for_each(|(out_idx, x)| {
            let base = self.reduce_base_offset(out_idx, &out_shape, axis);
            let stride = self.strides[axis] as usize;

            let mut best = self.storage.data[base].clone();
            let mut best_at = 0usize;

            for k in 1..reduce_dim {
                let value = self.storage.data[base + k * stride].clone();
                if value > best {
                    best = value;
                    best_at = k;
                }
            }

            *x = best_at;
        });

        } else {

            out_data.par_iter_mut().enumerate().for_each(|(out_idx, x)| {
            let base = self.reduce_base_offset(out_idx, &out_shape, axis);
            let stride = self.strides[axis] as usize;

            let mut best = self.storage.data[base].clone();
            let mut best_at = 0usize;

            for k in 1..reduce_dim {
                let value = self.storage.data[base + k * stride].clone();
                if value > best {
                    best = value;
                    best_at = k;
                }
            }

            *x = best_at;
        });

        }

        out_data
    }

    /// Validates a reduction axis and returns the resulting shape with its length.
    fn reduce_dims(&self, axis: usize) -> (Vec<usize>, usize) {
        assert!(axis < self.shape.len(), "!!!Axis index out of range!!!");

        let mut out_shape = self.shape.clone();
        let reduce_dim = out_shape.remove(axis);
        assert!(reduce_dim > 0, "!!!Cannot reduce an empty axis!!!");

        (out_shape, reduce_dim)
    }

    /// Maps a flat index of the reduced tensor back to where its run starts.
    fn reduce_base_offset(&self, out_idx: usize, out_shape: &[usize], axis: usize) -> usize {
        let mut tmp = out_idx;
        let mut base = self.offset as usize;

        for (i, &dim) in out_shape.iter().enumerate().rev() {
            let idx = tmp % dim;
            tmp /= dim;

            let in_axis = if i >= axis { i + 1 } else { i };
            base += idx * (self.strides[in_axis] as usize);
        }

        base
    }

    pub fn sum_axis_keepdim(&self, axis: usize) -> Tensor<T> {
        assert!(axis < self.shape.len(), "!!!Axis index out of range!!!");

        let mut out_shape = self.shape.clone();
        let reduce_dim = out_shape[axis];
        out_shape[axis] = 1;

        let out_len = product(&out_shape[..]);
        let mut out_data = vec![T::default(); out_len];

        if out_data.len() < PARALLEL_THRESHOLD {

            out_data.iter_mut().enumerate().for_each(|(out_idx, x)| {
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

        } else {

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

        }

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
        crate::linalg::tensor::ops::each_mut(&mut new_data, |x| *x = f(*x));

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

        let kernel = kernel.to_vec();   // a copy
        let stride = stride.to_vec();   // a copy

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
        let data = self.packed_data(); // a logically packed Vec<T>

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

        // Flatten, moving the elements out
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

        // Check that the sizes are consistent
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

impl<T: Num> From<Vec<Vec<Vec<Vec<T>>>>> for Tensor<T> {
    fn from(groups: Vec<Vec<Vec<Vec<T>>>>) -> Self {
        let d0 = groups.len();
        let d1 = if d0 == 0 { 0 } else { groups[0].len() };
        let d2 = if d1 == 0 { 0 } else { groups[0][0].len() };
        let d3 = if d2 == 0 { 0 } else { groups[0][0][0].len() };

        // Check that the sizes are consistent
        for (i, group) in groups.iter().enumerate() {
            if group.len() != d1 {
                panic!(
                    "Tensor::from(Vec<Vec<Vec<Vec<T>>>>): inconsistent group sizes: group 0 has {}, group {} has {}",
                    d1, i, group.len()
                );
            }
            for (j, block) in group.iter().enumerate() {
                if block.len() != d2 {
                    panic!(
                        "Tensor::from(Vec<Vec<Vec<Vec<T>>>>): inconsistent block sizes: group 0 block 0 has {}, group {} block {} has {}",
                        d2, i, j, block.len()
                    );
                }
                for (k, row) in block.iter().enumerate() {
                    if row.len() != d3 {
                        panic!(
                            "Tensor::from(Vec<Vec<Vec<Vec<T>>>>): inconsistent row sizes: group 0 block 0 row 0 has {}, group {} block {} row {} has {}",
                            d3, i, j, k, row.len()
                        );
                    }
                }
            }
        }

        // Flatten in row-major, the outermost axis varying slowest
        let mut flat = Vec::with_capacity(d0 * d1 * d2 * d3);
        for group in groups {
            for block in group {
                for row in block {
                    flat.extend(row);
                }
            }
        }

        Tensor::new(flat, vec![d0, d1, d2, d3])
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
                        // the separator between large blocks
                        if shape.len() >= 3 {
                            f.write_str("\n\n")?;
                        } else {
                            f.write_str("\n")?;
                        }
                        // the indent of a new block
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
    fn tensor_macro_builds_4d() {
        // [2, 2, 2, 2], read row-major
        let a = tensor![
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]
        ];
        assert_eq!(a.get_shape(), vec![2, 2, 2, 2]);
        assert_eq!(a.get_data(), (1..=16).map(|v| v as f64).collect::<Vec<_>>());

        // an asymmetric shape catches axes that got swapped
        let b = tensor![[[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]];
        assert_eq!(b.get_shape(), vec![1, 3, 2, 1]);
        assert_eq!(b.get_data(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // the layout Conv2d takes: 2 images, 3 channels, 1x2 each
        let c = tensor![
            [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]],
            [[[7.0, 8.0]], [[9.0, 10.0]], [[11.0, 12.0]]]
        ];
        assert_eq!(c.get_shape(), vec![2, 3, 1, 2]);

        // a trailing comma on every level
        let d = tensor![[[[1.0, 2.0,],],], [[[3.0, 4.0,],],],];
        assert_eq!(d.get_shape(), vec![2, 1, 1, 2]);
        assert_eq!(d.get_data(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "inconsistent")]
    fn tensor_macro_rejects_a_ragged_4d_literal() {
        let _ = tensor![[[[1.0, 2.0]], [[3.0]]]];
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
