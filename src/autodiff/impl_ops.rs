use std::ops::{Add, BitAnd, BitXor, Div, Mul, Neg, Sub};

use crate::{
    autodiff::{
        core::{Var, VarRef},
        ops::{add_op, div_op, matmul_op, mul_op, powf_op, sub_op},
    },
    linalg::Tensor,
    Float,
};

#[inline]
fn scalar_leaf<T: Float>(x: T) -> VarRef<T> {
    Var::leaf(Tensor::scalar(x), false)
}

/* =========================
   VarRef + VarRef
   ========================= */

impl<T: Float> Add<&VarRef<T>> for &VarRef<T> {
    type Output = VarRef<T>;
    fn add(self, rhs: &VarRef<T>) -> Self::Output {
        add_op(self, rhs)
    }
}

impl<T: Float> Sub<&VarRef<T>> for &VarRef<T> {
    type Output = VarRef<T>;
    fn sub(self, rhs: &VarRef<T>) -> Self::Output {
        sub_op(self, rhs)
    }
}

impl<T: Float> BitAnd<&VarRef<T>> for &VarRef<T> {
    type Output = VarRef<T>;
    fn bitand(self, rhs: &VarRef<T>) -> Self::Output {
        mul_op(self, rhs)
    }
}

impl<T: Float> BitXor<&VarRef<T>> for &VarRef<T> {
    type Output = VarRef<T>;
    fn bitxor(self, rhs: &VarRef<T>) -> Self::Output {
        powf_op(self, rhs)
    }
}

impl<T: Float> Div<&VarRef<T>> for &VarRef<T> {
    type Output = VarRef<T>;
    fn div(self, rhs: &VarRef<T>) -> Self::Output {
        div_op(self, rhs)
    }
}

impl<T: Float> Mul<&VarRef<T>> for &VarRef<T> {
    type Output = VarRef<T>;
    fn mul(self, rhs: &VarRef<T>) -> Self::Output {
        matmul_op(self, rhs)
    }
}

impl<T: Float> Neg for &VarRef<T> {
    type Output = VarRef<T>;
    fn neg(self) -> Self::Output {
        sub_op(&scalar_leaf(T::default()), self)
    }
}

/* =========================
   VarRef + scalar
   ========================= */

impl<T: Float> Add<T> for &VarRef<T> {
    type Output = VarRef<T>;
    fn add(self, rhs: T) -> Self::Output {
        add_op(self, &scalar_leaf(rhs))
    }
}

impl<T: Float> Sub<T> for &VarRef<T> {
    type Output = VarRef<T>;
    fn sub(self, rhs: T) -> Self::Output {
        sub_op(self, &scalar_leaf(rhs))
    }
}

impl<T: Float> Mul<T> for &VarRef<T> {
    type Output = VarRef<T>;
    fn mul(self, rhs: T) -> Self::Output {
        mul_op(self, &scalar_leaf(rhs))
    }
}

impl<T: Float> BitXor<T> for &VarRef<T> {
    type Output = VarRef<T>;
    fn bitxor(self, rhs: T) -> Self::Output {
        powf_op(self, &scalar_leaf(rhs))
    }
}

impl<T: Float> Div<T> for &VarRef<T> {
    type Output = VarRef<T>;
    fn div(self, rhs: T) -> Self::Output {
        div_op(self, &scalar_leaf(rhs))
    }
}

/* =========================
   VarRef + Tensor
   ========================= */

impl<T: Float> BitAnd<&Tensor<T>> for &VarRef<T> {
    type Output = VarRef<T>;
    fn bitand(self, rhs: &Tensor<T>) -> Self::Output {
        mul_op(self, &Var::leaf(rhs.shallow_copy(), false))
    }
}

impl<T: Float> Add<&Tensor<T>> for &VarRef<T> {
    type Output = VarRef<T>;
    fn add(self, rhs: &Tensor<T>) -> Self::Output {
        add_op(self, &Var::leaf(rhs.shallow_copy(), false))
    }
}

impl<T: Float> Sub<&Tensor<T>> for &VarRef<T> {
    type Output = VarRef<T>;
    fn sub(self, rhs: &Tensor<T>) -> Self::Output {
        sub_op(self, &Var::leaf(rhs.shallow_copy(), false))
    }
}

impl<T: Float> Mul<&Tensor<T>> for &VarRef<T> {
    type Output = VarRef<T>;
    fn mul(self, rhs: &Tensor<T>) -> Self::Output {
        matmul_op(self, &Var::leaf(rhs.shallow_copy(), false))
    }
}

impl<T: Float> Div<&Tensor<T>> for &VarRef<T> {
    type Output = VarRef<T>;
    fn div(self, rhs: &Tensor<T>) -> Self::Output {
        div_op(self, &Var::leaf(rhs.shallow_copy(), false))
    }
}

impl<T: Float> BitXor<&Tensor<T>> for &VarRef<T> {
    type Output = VarRef<T>;
    fn bitxor(self, rhs: &Tensor<T>) -> Self::Output {
        powf_op(self, &Var::leaf(rhs.shallow_copy(), false))
    }
}

/* =========================
   Tensor + VarRef
   ========================= */

impl<T: Float> Sub<&VarRef<T>> for &Tensor<T> {
    type Output = VarRef<T>;
    fn sub(self, rhs: &VarRef<T>) -> Self::Output {
        sub_op(&Var::leaf(self.shallow_copy(), false), rhs)
    }
}

impl<T: Float> Mul<&VarRef<T>> for &Tensor<T> {
    type Output = VarRef<T>;
    fn mul(self, rhs: &VarRef<T>) -> Self::Output {
        matmul_op(&Var::leaf(self.shallow_copy(), false), rhs)
    }
}