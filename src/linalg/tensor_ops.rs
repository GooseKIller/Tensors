use crate::linalg::Tensor;
use crate::Num;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

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