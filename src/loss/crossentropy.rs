use crate::linalg::{Matrix, Vector};
use crate::loss::Loss;
use crate::Float;

pub struct CrossEntropy<T: Float>(T);

impl<T: Float> CrossEntropy<T> {
    pub fn new(_data_type: T) -> Self {
        Self(_data_type)
    }

    fn vec_fun(&self, vector: Vector<T>) -> Vector<T> {
        let max = vector.max_val().unwrap();
        let shifted = vector.map_vec(|x| x - max);
        let sum = shifted.map_vec(|x| x.exp()).sum_all();
        shifted.map_vec(|x| x.exp() / sum)
    }

    fn softmax(&self, matrix: &Matrix<T>) -> Matrix<T> {
        let mut data: Vec<Vector<T>> = Vec::with_capacity(matrix.rows);
        for i in 0..matrix.rows {
            let vector = self.vec_fun(matrix.get_row(i));
            data.push(vector)
        }
        Matrix::from(data)
    }
}

impl<T: Float> Loss<T> for CrossEntropy<T> {
    fn call(&self, output: &Matrix<T>, target: &Matrix<T>) -> T {
        //let mut loss = T::default();
        let num_samples = output.rows;
        let epsilon = T::from_f64(1e-10);

        let softmax_output = output.max(epsilon);//self.softmax(output).max(epsilon);

        let mut loss = T::default();


        for i in 0..num_samples {
            for j in 0..softmax_output.cols {
                let predicted = softmax_output[[i, j]];/* < epsilon {
                    epsilon
                } else {
                    output[[i, j]]
                };*/
                loss -= target[[i, j]] * predicted.ln();
            }
        }

        loss / T::from_usize(num_samples)
    }
    fn gradient(&self, output: &Matrix<T>, target: &Matrix<T>) -> Matrix<T> {
        let num_samples = output.rows;
        let softmax_output = self.softmax(output);
        //(output - &target) * (T::one() / T::from_usize(num_samples))
        (softmax_output - target) * (T::one() / T::from_usize(num_samples))
    }
}

#[cfg(test)]
mod test {
    use crate::linalg::Matrix;
    use crate::loss::{CrossEntropy, Loss};
    use crate::DataType;

    #[test]
    fn test_cross_entropy_loss() {
        let loss_fn = CrossEntropy::new(DataType::f64());

        // Пример 1: Идеальное предсказание
        let output = Matrix::from(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
        let target = Matrix::from(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
        let loss = loss_fn.call(&output, &target);
        assert_eq!(loss, 0.0); // Идеальное предсказание должно давать 0 потерь

        // Пример 2: Неправильное предсказание
        let output = Matrix::from(vec![vec![0.9, 0.1, 0.0], vec![0.2, 0.7, 0.1]]);
        let target = Matrix::from(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
        let loss = loss_fn.call(&output, &target);
        assert!(loss > 0.0); // Потери должны быть больше 0

        // Пример 3: Смешанные предсказания
        let output = Matrix::from(vec![vec![0.5, 0.5, 0.0], vec![0.0, 1.0, 0.0]]);
        let target = Matrix::from(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
        let loss = loss_fn.call(&output, &target);
        assert!(loss > 0.0 && loss < 1.0); // Потери должны быть в разумных пределах

        // Пример 4: Проверка градиента
        let grad = loss_fn.gradient(&output, &target);
        assert_eq!(grad.rows, 2); // Должно быть столько же строк, сколько образцов
        assert_eq!(grad.cols, 3); // Должно быть столько же столбцов, сколько классов
    }
}
