use crate::{Float, linalg::{PaddingMode, Tensor}};

impl<T: Float> Tensor<T> {
    pub fn conv(&self, kernel: &Tensor<T>, padding: PaddingMode) -> Tensor<T> {
        match padding {
            PaddingMode::Valid => {
                let stride = vec![1; self.shape.len()];

                let out_shape: Vec<usize> = self.shape.iter()
                    .zip(kernel.shape.iter())
                    .map(|(&dim, &k)| dim - k + 1)
                    .collect();

                let data: Vec<T> = self.window(&kernel.shape, &stride)
                    .map(|w| w.mul_sum(kernel))
                    .collect();

                Tensor::new(data, out_shape)
            }
            PaddingMode::Mirror(_, _) => todo!(),
            PaddingMode::Zero(_, _) => todo!(),
        }
    }
}

#[cfg(test)]
mod tests{
    use crate::tensor;

    #[test]
    fn simple_conv() {
        let img = tensor![
            [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
            [0.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0],
            [1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0],
            [1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0],
            [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],
            [1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0],
            [1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0],
            [0.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0],
            [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
        ];

        let kernel = tensor![[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];

        println!("{}", img.conv(&kernel, crate::linalg::PaddingMode::Valid));

        let w = img.window(&kernel.shape, &[1,1]).next().unwrap();
        println!("{:?}", w.mul_sum(&kernel));
    }
}