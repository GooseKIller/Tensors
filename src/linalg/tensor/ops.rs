use crate::linalg::{Tensor, product};
use crate::{Float, Num};
use rayon::iter::IntoParallelRefMutIterator;
use rayon::prelude::*;
use std::ops::{BitXor, Div, Neg};
use std::{ops::{
    Add, AddAssign, BitAnd, BitAndAssign, Mul, MulAssign, Sub, SubAssign
}, sync::Arc};

pub(crate) fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let ndim = a.len().max(b.len());
    let mut out = Vec::with_capacity(ndim);

    for i in 0..ndim {
        let da = a.get(a.len().wrapping_sub(1 + i)).copied().unwrap_or(1);
        let db = b.get(b.len().wrapping_sub(1 + i)).copied().unwrap_or(1);

        if da == db || da == 1 || db == 1 {
            out.push(da.max(db));
        } else {
            return None;
        }

    }

    out.reverse();
    Some(out)
}

//
// Scalar Ops
// 

// ====== ADD ======
impl<T: Num> Add<T> for &Tensor<T> {
    type Output = Tensor<T>;
    fn add(self, rhs: T) -> Self::Output {
        let mut out = self.packed_data();
        out.par_iter_mut()
            .for_each(|x| *x += rhs );
        Tensor::new(out, self.shape.clone())
    }
}

impl<T: Num> Add<T> for Tensor<T> {
    type Output = Tensor<T>;
    fn add(mut self, rhs: T) -> Self::Output {
        if self.can_inplace() {
            let storage = Arc::make_mut(&mut self.storage);
            storage.data.par_iter_mut().for_each(|x| *x += rhs);
            self
        } else {
            (&self).add(rhs)
        }
    }
}

impl<T: Num> AddAssign<T> for Tensor<T> {
    fn add_assign(&mut self, rhs: T) {
        assert!(
            self.can_inplace(),
            "AddAssign: in-place operation disallowed on non-contiguous / aliased / broadcasted tensor"
        );
        let storage = Arc::make_mut(&mut self.storage);
        storage.data.par_iter_mut().for_each(|x| *x += rhs);

    }
}

// ====== SUB ======
impl<T: Num> Sub<T> for &Tensor<T> {
    type Output = Tensor<T>;
    fn sub(self, rhs: T) -> Self::Output {
        let mut out = self.packed_data();
        out.par_iter_mut()
            .for_each(|x| *x -= rhs );
        Tensor::new(out, self.shape.clone())
    }
}

impl<T: Num> Sub<T> for Tensor<T> {
    type Output = Tensor<T>;
    fn sub(mut self, rhs: T) -> Self::Output {
        if self.can_inplace() {
            let storage = Arc::make_mut(&mut self.storage);
            storage.data.par_iter_mut().for_each(|x| *x -= rhs);
            self
        } else {
            (&self).sub(rhs)
        }
    }
}


impl<T: Num> SubAssign<T> for Tensor<T> {
    fn sub_assign(&mut self, rhs: T) {
        assert!(
            self.can_inplace(),
            "SubAssign: in-place operation disallowed on non-contiguous / aliased / broadcasted tensor"
        );
        let storage = Arc::make_mut(&mut self.storage);
        storage.data.par_iter_mut().for_each(|x| *x -= rhs);
    }
}

macro_rules! impl_sub_tensor_for_types {
    ($($type:ty),*) => {
        $(
            impl Sub<&Tensor<$type>> for $type {
                type Output = Tensor<$type>;

                fn sub(self, rhs: &Tensor<$type>) -> Tensor<$type> {
                    let mut out = rhs.packed_data();
                    out.par_iter_mut()
                        .for_each(|x| *x = self - *x );
                    Tensor::new(out, rhs.shape.clone())
                }
            }

            impl Sub<Tensor<$type>> for $type {
                type Output = Tensor<$type>;

                fn sub(self, mut rhs: Tensor<$type>) -> Tensor<$type> {
                    if rhs.can_inplace() {
                        if let Some(storage) = Arc::get_mut(&mut rhs.storage) {
                            storage.data.par_iter_mut().for_each(|x| *x = self - *x);
                            return rhs;
                        }
                    }
                    self.sub(&rhs)
                }
            }
        )*
    };
}
impl_sub_tensor_for_types!(i16, i32, i64, i128, f32, f64);

impl<T:Num> Neg for Tensor<T> {
    type Output = Tensor<T>;
    fn neg(mut self) -> Self::Output {
        if self.can_inplace() {
            let storage = Arc::make_mut(&mut self.storage);
            storage.data.par_iter_mut().for_each(|x| *x = -*x);
            self
        } else {
            let mut data = self.packed_data();
            data.par_iter_mut().for_each(|x| *x = -*x);
            Tensor::new(data, self.shape.clone())

        }
    }
}

impl<T:Num> Neg for &Tensor<T> {
    type Output = Tensor<T>;
    fn neg(self) -> Self::Output {
        let mut data = self.packed_data();
        data.par_iter_mut().for_each(|x| *x = -*x);
        Tensor::new(data, self.shape.clone())
    }
}


// ====== Mul ======
impl<T: Num> Mul<T> for &Tensor<T> {
    type Output = Tensor<T>;
    fn mul(self, rhs: T) -> Self::Output {
        let mut out = self.packed_data();
        out.par_iter_mut()
            .for_each(|x| *x *= rhs );
        Tensor::new(out, self.shape.clone())
    }
}

impl<T: Num> Mul<T> for Tensor<T> {
    type Output = Tensor<T>;
    fn mul(mut self, rhs: T) -> Self::Output {
        if self.can_inplace() {
            let storage = Arc::make_mut(&mut self.storage);
            storage.data.par_iter_mut().for_each(|x| *x *= rhs);
            self
        } else {
            (&self).mul(rhs)
        }
    }
}

impl<T: Num> MulAssign<T> for Tensor<T> {
    fn mul_assign(&mut self, rhs: T) {
        assert!(
            self.can_inplace(),
            "MulAssign: in-place operation disallowed on non-contiguous / aliased / broadcasted tensor"
        );
        let storage = Arc::make_mut(&mut self.storage);
        storage.data.par_iter_mut().for_each(|x| *x *= rhs);
    }
}


// ======== DIV =======
impl<T: Num> Div<T> for &Tensor<T> {
    type Output = Tensor<T>;
    fn div(self, rhs: T) -> Self::Output {
        let mut out = self.packed_data();
        out.par_iter_mut()
            .for_each(|x| *x = *x / rhs);
        Tensor::new(out, self.shape.clone())
    }
}
impl<T: Num> Div<T> for Tensor<T> {
    type Output = Tensor<T>;
    fn div(mut self, rhs: T) -> Self::Output {
        if self.can_inplace() {
            let storage = Arc::make_mut(&mut self.storage);
            storage.data.par_iter_mut().for_each(|x| *x = *x / rhs);
            self
        } else {
            (&self).div(rhs)
        }
    }
}

// ====================
// Tensor to Tensor ops
// ====================


fn binary_op<F, T:Num>(a: &Tensor<T>, b: &Tensor<T>, f: F, error_msg: &str) -> Tensor<T>
    where F: Fn(T, T) -> T + Sync + Send {
    let shape = broadcast_shape(&a.shape[..], &b.shape[..])
        .expect(error_msg);

    let a_view = a.broadcast_to(&shape).expect("broadcast_to failed (bug)");
    let b_view = b.broadcast_to(&shape).expect("broadcast_to failed (bug)");
    

    let a_data = &a_view.storage.data;
    let b_data = &b_view.storage.data;
    let a_idx = a_view.storage_indices();
    let b_idx = b_view.storage_indices();

    let n = product(&shape[..]);

    let mut out = vec![T::default(); n];

    out.par_iter_mut()
        .enumerate()
        .for_each(|(i,slot)| {
            let av = a_data[a_idx[i]];
            let bv = b_data[b_idx[i]];
            *slot = f(av, bv);
        });
    
    Tensor::new(out, shape)
}

// ====== ADD ======
pub fn add<T: Num>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    binary_op(a, b, |x, y| x + y,
         "!!!Add: incompatible shapes!!!")
}

impl<T: Num> Add<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;
    fn add(self, rhs: &Tensor<T>) -> Self::Output {
        add(self, rhs)
    }
}

impl<T: Num> Add<&Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;
    fn add(mut self, rhs: &Tensor<T>) -> Self::Output {
        if self.can_inplace() &&
            self.shape == rhs.shape &&
            Arc::ptr_eq(&self.storage, &rhs.storage) {
                let rhs_packed = rhs.packed_data();
                let storage = Arc::make_mut(&mut self.storage);
                storage.data
                    .par_iter_mut()
                    .zip(rhs_packed.par_iter())
                    .for_each(|(x, &y)| *x += y);
                self
        } else {
            add(&self, &rhs)
        }
    }
}

impl<T: Num> AddAssign<&Tensor<T>> for Tensor<T> {
    fn add_assign(&mut self, rhs: &Tensor<T>) {
        if self.shape == rhs.shape 
            && self.can_inplace()
            && Arc::ptr_eq(&self.storage, &rhs.storage) {
                let storage = Arc::make_mut(&mut self.storage);

                let rhs_data = rhs.packed_data();
                storage
                    .data
                    .par_iter_mut()
                    .zip(rhs_data.par_iter())
                    .for_each(|(a, &b)| *a += b);
                return;
        }

        let result = add(self, rhs);

        self.storage = result.storage;
        self.shape = result.shape;
        self.strides = result.strides;
        self.offset = result.offset;
    }
}
/*
impl<T: Num + Send + Sync> Tensor<T> {
    pub fn add(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
        let shape = broadcast_shape(&a.shape, &b.shape)
            .expect("add: incompatible shapes");

        let a_view = a.broadcast_to(&shape)
            .expect("broadcast_to failed (BUG)");
        let b_view = b.broadcast_to(&shape)
            .expect("broadcast_to failed (BUG)");

        // ---------- fast-path inplace ----------
        if a_view.can_inplace() && a_view.shape == shape {
            let mut out = a_view.clone(); // shallow clone, Arc strong_count == 1
            let a_idx = out.storage_indices();
            let b_idx = b_view.storage_indices();

            let storage = Arc::make_mut(&mut out.storage);
            let data = &mut storage.data;

            a_idx
                .par_iter()
                .zip(b_idx.par_iter())
                .for_each(|(&ia, &ib)| {
                    data[ia] += b_view.storage.data[ib];
                });

            return out;
        }

        // ---------- fallback: materialize ----------
        let a_idx = a_view.storage_indices();
        let b_idx = b_view.storage_indices();

        let mut out = Vec::with_capacity(product(&shape));
        let a_data = &a_view.storage.data;
        let b_data = &b_view.storage.data;

        for (&ia, &ib) in a_idx.iter().zip(b_idx.iter()) {
            out.push(a_data[ia] + b_data[ib]);
        }

        Tensor::new(out, shape)
    }
}

*/

// 
// Sub
//
pub fn sub<T:Num>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    binary_op(a, b, |x, y| x - y,
         "!!!Sub: incompatible shapes!!!")
}

impl<T: Num> Sub<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;
    fn sub(self, rhs: &Tensor<T>) -> Self::Output {
        sub(&self, rhs)
    }
}

impl<T:Num> Sub<&Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;
    fn sub(mut self, rhs: &Tensor<T>) -> Self::Output {
        if self.can_inplace() &&
            self.shape == rhs.shape &&
            Arc::ptr_eq(&self.storage, &rhs.storage) {
                let rhs_packed = rhs.packed_data();
                let storage = Arc::make_mut(&mut self.storage);
                storage.data
                    .par_iter_mut()
                    .zip(rhs_packed.par_iter())
                    .for_each(|(x, &y)| *x -= y);
                self
        } else {
            sub(&self, &rhs)
        }
    }
}

impl<T: Num> SubAssign<&Tensor<T>> for Tensor<T> {
    fn sub_assign(&mut self, rhs: &Tensor<T>) {
        if self.shape == rhs.shape 
            && self.can_inplace()
            && Arc::ptr_eq(&self.storage, &rhs.storage) {
                let storage = Arc::make_mut(&mut self.storage);

                let rhs_data = rhs.packed_data();
                storage
                    .data
                    .par_iter_mut()
                    .zip(rhs_data.par_iter())
                    .for_each(|(a, &b)| *a -= b);
                return;
        }

        let result = sub(self, rhs);

        self.storage = result.storage;
        self.shape = result.shape;
        self.strides = result.strides;
        self.offset = result.offset;
    }
}

//
// Element-wise Mul
//


pub fn mul_elem<T:Num>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    binary_op(a, b, |x, y| x * y,
        "!!!MulElems: incompatible shapes!!!")
}

impl<T:Num> BitAnd<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;
    fn bitand(self, rhs: &Tensor<T>) -> Self::Output {
        mul_elem(self, rhs)
    }
}

impl<T:Num> BitAnd<&Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;
    fn bitand(mut self, rhs: &Tensor<T>) -> Self::Output {
        if self.can_inplace() &&
            self.shape == rhs.shape &&
            Arc::ptr_eq(&self.storage, &rhs.storage) {
                let rhs_packed = rhs.packed_data();
                let storage = Arc::make_mut(&mut self.storage);
                storage.data
                    .par_iter_mut()
                    .zip(rhs_packed.par_iter())
                    .for_each(|(x, &y)| *x *= y);
                self
        } else {
            mul_elem(&self, &rhs)
        }
    }
}

impl<T:Num>  BitAndAssign<&Tensor<T>> for Tensor<T> {
    fn bitand_assign(&mut self, rhs: &Tensor<T>) {
        if self.shape == rhs.shape 
            && self.can_inplace()
            && Arc::ptr_eq(&self.storage, &rhs.storage) {
                let storage = Arc::make_mut(&mut self.storage);

                let rhs_data = rhs.packed_data();
                storage
                    .data
                    .par_iter_mut()
                    .zip(rhs_data.par_iter())
                    .for_each(|(a, &b)| *a *= b);
                return;
        }

        let result = mul_elem(self, rhs);

        self.storage = result.storage;
        self.shape = result.shape;
        self.strides = result.strides;
        self.offset = result.offset;
    }
}


impl<T:Num> Mul<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;
    fn mul(self, rhs: &Tensor<T>) -> Self::Output {
        self.matmul(rhs)
    }
}

pub fn powf<T:Float>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    binary_op(a, b, |a, b| a.powf(b), "!!!Powf: incompatible shapes!!!")
}

impl<T: Float> BitXor<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;
    fn bitxor(self, rhs: &Tensor<T>) -> Self::Output {
        powf(self, rhs)
    }
}

pub fn div<T:Num>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    binary_op(a, b, |a, b| a/b, "!!!DIV: incompatible shapes!!!")
}

impl<T:Float> Div<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;
    fn div(self, rhs: &Tensor<T>) -> Self::Output {
        div(self, rhs)
    }
}

#[cfg(test)]
mod test {
    use crate::tensor;

    #[test]
    fn add_test() {
        let a = tensor![1,2,3];
        let b= tensor![3,4,5];

        let c = &a + &b;

        assert_eq!(c.get_shape(), &[3]);
        assert_eq!(c.get_data(), vec![4, 6, 8]);
    }

    #[test]
    fn scalar_broadcast_add() {
        let a = tensor![1, 2, 3, 4];
        let b = tensor![10]; // scalar-like

        let c = &a + &b;

        assert_eq!(c.get_shape(), &[4]);
        assert_eq!(c.get_data(), vec![11, 12, 13, 14]);
    }

    #[test]
    fn vector_matrix_broadcast_add() {
        let a = tensor![[1, 2, 3],
            [4, 5, 6]];

        let b = tensor![[10, 20, 30]];
        
        let c = &a + &b;

        assert_eq!(c.get_shape(), &[2, 3]);
        assert_eq!(
            c.get_data(),
            vec![
                11, 22, 33,
                14, 25, 36
            ]
        );
    }

    #[test]
    fn broadcast_disables_inplace() {
        let a = tensor![
            [1, 2, 3],
            [4, 5, 6]
        ];

        let b = tensor![10, 20, 30];

        // здесь broadcast → inplace быть не должно
        let c = &a + &b;

        // убеждаемся, что a не изменился
        assert_eq!(
            a.get_data(),
            vec![
                1, 2, 3,
                4, 5, 6
            ]
        );

        assert_eq!(
            c.get_data(),
            vec![
                11, 22, 33,
                14, 25, 36
            ]
        );
    }

    #[test]
    fn add_assign_no_broadcast() {
        let mut a = tensor![1, 2, 3];
        let b = tensor![10, 20, 30];

        a += &b;

        assert_eq!(a.get_shape(), &[3]);
        assert_eq!(a.get_data(), vec![11, 22, 33]);
    }

    #[test]
    fn view_then_materialize() {
        let a = tensor![
            [1, 2, 3],
            [4, 5, 6]
        ];

        let b = a.slice(&[0, 1], &[2, 2]).unwrap(); // shape [2,2], non-contiguous

        assert!(!b.is_contiguous());

        let c = b.clone();
        //println!("{a}, {b}, {c}");

        assert!(c.is_contiguous());
        assert_eq!(c.get_shape(), &[2, 2]);
        assert_eq!(c.get_data(), vec![2, 3, 5, 6]);
    }
}
