use crate::Float;
use crate::linalg::Matrix;
use crate::loss::Loss;

pub struct SSE<T: Float>(T);

impl<T: Float> SSE<T> {
    pub fn new(datatype_number: T) -> Self {
        Self(datatype_number)
    }
}

impl<T: Float> Loss<T> for SSE<T> {
    fn call(&self, output: &Matrix<T>, target: &Matrix<T>) -> T {
        if output.size() != target.size() {
            panic!("!!!Size of output matrix and target must be equal!!!\nOutput size:{:?} Target size: {:?}", output.size(), target.size())
        }
        let diff = target - output;
        diff.map(|x| x.powf(T::from(2))).sum()

    }

    fn gradient(&self, output: &Matrix<T>, target: &Matrix<T>) -> Matrix<T> {
        if output.size() != target.size() {
            panic!("!!!Size of output matrix and target must be equal!!!")
        }
        target - output
    }
}