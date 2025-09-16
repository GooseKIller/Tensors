use crate::activation::Function;
use crate::linalg::{Matrix, Vector};
use crate::Float;
use rand::distributions::{Distribution, Standard};
use rand::random;
use onnx_pb::{AttributeProto, NodeProto, TensorProto};

/// Representation of a linear layer in a neural network.
///
/// # Example
/// ```
/// use tensorrs::activation::Function;
/// use tensorrs::nn::Linear;
/// use tensorrs::linalg::Matrix;
/// use tensorrs::matrix;
///
/// let lay = Linear::new(2, 4, true);
///
/// let b = matrix![[1.0, 2.0]];
/// let shape = lay.call(b).shape();
/// assert_eq!([1, 4], shape);
/// ```
///
/// # Formula
///
/// ```math
/// y = x^{+} \cdot W
/// ```
///
/// where:
/// ```math
/// - y \text{is the output.}\\\
/// - W \text{is the weight matrix and bias(including bias if applicable),}\\\
/// - x^+ \text{is the input matrix with added ones column,}\\\
/// ```
/// The bias is added as an additional row in the weight matrix. This allows the bias to be applied to all outputs in a single matrix multiplication.
///
pub struct Linear<T: Float> {
    pub matrix: Matrix<T>,
    pub(crate) bias: bool,
}

impl<T: Float> Linear<T>
where
    Standard: Distribution<T>,
{
    /// Creates a new matrix with added bias(optional)
    ///
    /// bias realized as double row
    pub fn new(row: usize, col: usize, bias: bool) -> Self {
        // Xavier method
        let mut data = Vec::with_capacity(row * col);
        let limit = (T::from_usize(6) / T::from_usize(row + col)).sqrt(); //sqrt(6) / sqrt(n_i + n_i+1)

        for _ in 0..(col * row) {
            let value = random::<T>() * T::from(2) * limit - limit; // [-limit, limit)
            data.push(value);
        }

        if bias {
            let mut bias_data = Vec::with_capacity(col);
            for _ in 0..col {
                bias_data.push(T::default()); // Инициализация смещений нулями
            }
            data.extend(bias_data);
        }

        let matrix = Matrix::new(data, row + if bias { 1 } else { 0 }, col);
        Self { matrix, bias }
    }

    /// Creates a matrix without random numbers
    ///
    /// the same as ::new method
    pub fn zeros(row: usize, col: usize, bias: bool) -> Self {
        if bias {
            let data = vec![T::default(); col * (row + 1)];
            let matrix = Matrix::new(data, row + 1, col);
            return Self { matrix, bias };
        }

        let data = vec![T::default(); col * row];
        let matrix = Matrix::new(data, row, col);
        Self { matrix, bias }
    }

    /// From Matrix with added bias
    ///
    ///# Example
    ///
    /// ```
    ///use tensorrs::linalg::Matrix;
    ///use tensorrs::matrix;
    ///use tensorrs::nn::Linear;
    ///
    ///let linear:Linear<f64> = Linear::new(1, 2, true);//with one bias row it will be 2x2
    ///assert_eq!([2, 2], linear.shape());
    ///```
    pub fn shape(&self) -> [usize; 2] {
        [self.matrix.rows, self.matrix.cols]
    }

    /// Return weights matrix
    ///
    /// # Example
    ///
    /// ```
    /// use tensorrs::matrix;
    /// use tensorrs::linalg::Matrix;
    /// use tensorrs::nn::Linear;
    /// let act1:Linear<f64> = Linear::from(matrix![[1.0],
    ///                                 [1.0]]);
    /// let sum_num = 2.0;
    ///assert_eq!(sum_num, act1.get_weights().sum());
    /// ```
    pub fn get_weights(&self) -> Matrix<T> {
        self.matrix.clone()
    }


    /// HI
    pub(crate) fn save_onnx(&self, name: &str) -> (NodeProto, Vec<TensorProto>) {
        let weights = self.get_weights();
        let bias = self.get_bias();

        let _input_dim = weights.rows as i64;
        let _output_dim = weights.cols as i64;

        let w_data: Vec<_> = weights.data.clone();
        let w_shape = vec![weights.rows as i64, weights.cols as i64];
        let w_tensor = TensorProto{
            name: format!("{name}_W"),

            data_type: T::if_f32_f64(1, 11), // 1 - Float, 11 -Double
            dims: w_shape,

            float_data: T::if_f32_f64(w_data.clone().iter().map(|x| T::to_f32(*x)).collect(), vec![]),
            double_data: T::if_f32_f64(vec![], w_data.clone().iter().map(|x| T::to_f64(*x)).collect()),

            int32_data: vec![],
            string_data: vec![],
            segment: None,
            int64_data: vec![],
            doc_string: String::new(),
            raw_data: Vec::new(),
            external_data: vec![],
            data_location: 0,
            uint64_data: vec![],
        };
        let mut initializers = vec![w_tensor];

        let mut input_names = vec!["input".to_string(), format!("{name}_W")];
        let mut output_names = vec![format!("{name}_output")];

        if let Some(b) = bias {
            let b_data = b.data.clone();
            let b_tensor = TensorProto {
                name: format!("{name}_B"),
                data_type: T::if_f32_f64(1, 11),
                dims: vec![b.cols() as i64],
                float_data: T::if_f32_f64(b_data.clone().iter().map(|x| T::to_f32(*x)).collect(), vec![]),
                double_data: T::if_f32_f64(vec![], b_data.clone().iter().map(|x| T::to_f64(*x)).collect()),

                int32_data: vec![],
                string_data: vec![],
                segment: None,
                int64_data: vec![],
                doc_string: String::new(),
                raw_data: Vec::new(),
                external_data: vec![],
                data_location: 0,
                uint64_data: vec![],
            };

            initializers.push(b_tensor);
            input_names.push(format!("{name}_B"));
        };

        let node = NodeProto {
            name: name.to_string(),
            op_type: "Gemm".to_string(),
            input: input_names,
            output: output_names.clone(),
            attribute: vec![
                AttributeProto {
                    name: "transA".to_string(),
                    r#type: 2, // INT
                    i: 0,

                    ref_attr_name: String::new(),
                    doc_string: String::new(),
                    f: 0.0,
                    s: Vec::new(),
                    t: None,
                    g: None,
                    sparse_tensor: None,
                    floats: vec![],
                    ints: vec![],
                    strings: vec![],
                    tensors: vec![],
                    graphs: vec![],
                    sparse_tensors: vec![],
                },
                AttributeProto {
                    name: "TransB".to_string(),
                    r#type: 2, // INT
                    i: 1,

                    ref_attr_name: String::new(),
                    doc_string: "".to_string(),
                    f: 0.0,
                    s: Vec::new(),
                    t: None,
                    g: None,
                    sparse_tensor: None,
                    floats: vec![],
                    ints: vec![],
                    strings: vec![],
                    tensors: vec![],
                    graphs: vec![],
                    sparse_tensors: vec![],
                },
                AttributeProto {
                    name: "alpha".to_string(),
                    r#type: 1, // Float
                    f: 1.0,

                    ref_attr_name: "".to_string(),
                    doc_string: "".to_string(),
                    i: 0,
                    s: Vec::new(),
                    t: None,
                    g: None,
                    sparse_tensor: None,
                    floats: vec![],
                    ints: vec![],
                    strings: vec![],
                    tensors: vec![],
                    graphs: vec![],
                    sparse_tensors: vec![],
                },
                AttributeProto {
                    name: "beta".to_string(),
                    r#type: 1, // Float
                    f: 1.0,

                    ref_attr_name: "".to_string(),
                    doc_string: "".to_string(),
                    i: 0,
                    s: Vec::new(),
                    t: None,
                    g: None,
                    sparse_tensor: None,
                    floats: vec![],
                    ints: vec![],
                    strings: vec![],
                    tensors: vec![],
                    graphs: vec![],
                    sparse_tensors: vec![],
                },
            ],
            doc_string: "".to_string(),
            domain: "".to_string(),
        };

        (node, initializers)
    }
}

impl<T: Float> From<Matrix<T>> for Linear<T> {
    /// From Matrix with added bias
    ///
    ///# Example
    ///
    /// ```
    ///use tensorrs::linalg::Matrix;
    ///use tensorrs::matrix;
    ///use tensorrs::nn::Linear;
    ///
    ///let mx:Matrix<f64> = matrix![[1.0],
    ///                 [1.0]]; // this is bias
    ///let linear = Linear::from(mx);
    ///println!("{:?}", linear.shape());
    ///```
    fn from(value: Matrix<T>) -> Self {
        Self {
            matrix: value,
            bias: true,
        }
    }
}

impl<T: Float> Function<T> for Linear<T> {
    fn name(&self) -> String {
        let shape = self.matrix.shape();
        format!("Linear_{}:{} {}", self.bias as u8, shape[0], shape[1])
    }
    fn call(&self, mut matrix: Matrix<T>) -> Matrix<T> {
        if self.bias {
            let rows = matrix.rows();
            let num_bias: Vector<T> = Vector::from_num(1.into(), rows);
            matrix.add_column(num_bias.into())
        }
        matrix * &self.matrix
    }

    /// not real derivative just gradient calculating
    fn derivative(&self, matrix: Matrix<T>) -> Matrix<T> {
        if self.bias {
            let ans = &matrix * &self.matrix.transpose();
            return ans.rem_col(ans.cols - 1);
        }
        &matrix * &self.matrix.transpose()
    }

    fn is_linear(&self) -> bool {
        true
    }

    fn get_data(&self) -> Option<Matrix<T>> {
        Some(self.matrix.clone())
    }

    fn set_data(&mut self, _data: Matrix<T>) {
        self.matrix = _data;
    }

    fn get_weights(&self) -> Option<Matrix<T>> {
        let weights = &self.matrix.data[0..(self.matrix.rows - 1) * self.matrix.cols];
        Some(Matrix::new(
            weights.to_owned(),
            self.matrix.rows - 1,
            self.matrix.cols,
        ))
    }

    fn get_bias(&self) -> Option<Matrix<T>> {
        if !self.bias {
            return None;
        }
        Some(Matrix::from(self.matrix.get_row(self.matrix.rows() - 1)))
    }

    fn is_bias(&self) -> bool {
        self.bias
    }
/*
    fn get_onnx(&self, _name: &str) -> Option<(NodeProto, Vec<TensorProto>)> {
        Some(self.save_onnx(_name))
    }

 */
}

#[cfg(test)]
mod tests {
    use crate::activation::Function;
    use crate::linalg::Matrix;
    use crate::matrix;
    use crate::nn::linear::Linear;

    #[test]
    fn new_linear() {
        let a: Linear<f64> = Linear::new(1, 1, true);
        println!("{}", a.matrix);
    }

    #[test]
    fn call_linear() {
        let a: Linear<f64> = Linear::new(1, 1, true);
        let m = Matrix::from_num(1.0, 1, 1);
        let call = a.call(m);
        assert_eq!(Matrix::from_num(a.matrix.sum(), 1, 1), call);
    }

    #[test]
    fn from_matrix() {
        let matrix = matrix![[1.0], [2.0]];
        let linear = Linear::from(matrix);
        let m = matrix![[1.0]];
        let call = linear.call(m);
        assert_eq!(Matrix::from_num(3.0, 1, 1), call)
    }
}
