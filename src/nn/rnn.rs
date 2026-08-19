use rand::{distributions::Standard, prelude::Distribution};

use crate::{
    Float,
    activation::{Module, Tanh},
    autodiff::{AutoGrad, Var, VarRef, select_op, stack_op},
    linalg::Tensor,
};

/// What an [RNN] hands back once the last time step is done.
pub enum RNNOutput {
    /// Only the final hidden state, of shape `[batch, hidden]`.
    ///
    /// This is what a classification or regression head expects, so it is the
    /// default.
    Last,
    /// The hidden state of every step, of shape `[batch, seq, hidden]`.
    ///
    /// Needed to stack one recurrent layer on top of another, or to predict a
    /// value per step.
    Sequence,
}

/// One step of a recurrent network.
///
/// # Formula
///```math
///  h_t = \tanh(x_t W_{ih} + h_{t-1} W_{hh} + b)
///```
/// Where $`x_t`$ is the input at step $`t`$ and $`h_{t-1}`$ the hidden state
/// carried over from the previous step
///
/// # Example
/// ```
/// use tensorrs::{tensor, nn::RNNCell, autodiff::{AutoGrad, Var}};
///
/// let cell = RNNCell::<f32>::new(3, 4, true);
///
/// let mut h = cell.zero_state(2); // batch of 2
/// for _ in 0..5 {
///     let x_t = Var::leaf(tensor![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], false);
///     h = cell.forward(&x_t, &h);
/// }
/// assert_eq!(h.value().get_shape(), vec![2, 4]);
/// ```
///
/// # Notes
/// Driving the cell by hand is what you want when the sequence arrives one step
/// at a time, or when the state has to survive between calls. To run over a whole
/// sequence at once use [RNN], which owns a cell and loops over it.
pub struct RNNCell<T: Float> {
    /// Input-to-hidden weights, of shape `[input_size, hidden_size]`.
    pub w_ih: VarRef<T>,
    /// Hidden-to-hidden weights, of shape `[hidden_size, hidden_size]`.
    pub w_hh: VarRef<T>,
    /// Bias, of shape `[1, hidden_size]`.
    pub bias: Option<VarRef<T>>,
    hidden_size: usize,
    act: Box<dyn Module<T>>,
}

impl<T: Float> RNNCell<T>
where
    Standard: Distribution<T>,
{
    /// Creates a cell with a [Tanh] activation.
    ///
    /// # Example
    /// ```
    /// use tensorrs::nn::RNNCell;
    ///
    /// let cell = RNNCell::<f32>::new(10, 20, true);
    /// assert_eq!(cell.parameters().len(), 3); // w_ih, w_hh, bias
    /// ```
    ///
    /// # Arguments
    /// * `input_size` — the number of features of one time step.
    /// * `hidden_size` — the size of the hidden state.
    /// * `bias` — whether to add a bias term.
    ///
    /// # Notes
    /// Both weight matrices start from $`U(-k, k)`$ with
    /// $`k = 1/\sqrt{\text{hidden\_size}}`$. Keeping the recurrent weights small
    /// is what stops the state from growing without bound as steps pile up.
    pub fn new(input_size: usize, hidden_size: usize, bias: bool) -> Self {
        Self::with_activation(input_size, hidden_size, bias, Box::new(Tanh::new()))
    }

    /// Creates a cell with an activation of your choice.
    ///
    /// # Example
    /// ```
    /// use tensorrs::{activation::ReLU, nn::RNNCell};
    ///
    /// let cell = RNNCell::<f32>::with_activation(10, 20, true, Box::new(ReLU::new()));
    /// ```
    ///
    /// # Arguments
    /// * `input_size` — the number of features of one time step.
    /// * `hidden_size` — the size of the hidden state.
    /// * `bias` — whether to add a bias term.
    /// * `act` — the activation applied to the pre-activation of every step.
    ///
    /// # Notes
    /// An activation with parameters of its own, such as
    /// [PReLU](crate::activation::PReLU), is trained along with the cell: its
    /// parameters are reported by [RNNCell::parameters]. The same instance runs
    /// on every step, so those parameters are shared across time.
    pub fn with_activation(
        input_size: usize,
        hidden_size: usize,
        bias: bool,
        act: Box<dyn Module<T>>,
    ) -> Self {
        let limit = T::one() / T::from_usize(hidden_size).sqrt();

        Self {
            w_ih: Var::leaf(Self::uniform(vec![input_size, hidden_size], limit), true),
            w_hh: Var::leaf(Self::uniform(vec![hidden_size, hidden_size], limit), true),
            bias: if bias {
                Some(Var::leaf(
                    Tensor::from_num(T::default(), vec![1, hidden_size]),
                    true,
                ))
            } else {
                None
            },
            hidden_size,
            act,
        }
    }

    /// Draws from `U(-limit, limit)`.
    fn uniform(shape: Vec<usize>, limit: T) -> Tensor<T> {
        (Tensor::rand(shape) * T::from_usize(2) - T::one()) * limit
    }
}

impl<T: Float> RNNCell<T> {
    /// Returns the size of the hidden state.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Builds the all-zero hidden state a sequence starts from.
    ///
    /// # Example
    /// ```
    /// use tensorrs::{nn::RNNCell, autodiff::AutoGrad};
    ///
    /// let cell = RNNCell::<f32>::new(3, 4, true);
    /// assert_eq!(cell.zero_state(8).value().get_shape(), vec![8, 4]);
    /// ```
    ///
    /// # Arguments
    /// * `batch` — how many sequences are processed side by side.
    ///
    /// # Notes
    /// The state is a leaf that does not require a gradient, so nothing flows
    /// past the beginning of the sequence.
    pub fn zero_state(&self, batch: usize) -> VarRef<T> {
        Var::leaf(
            Tensor::from_num(T::default(), vec![batch, self.hidden_size]),
            false,
        )
    }

    /// Advances the state by one time step.
    ///
    /// # Example
    /// ```
    /// use tensorrs::{tensor, nn::RNNCell, autodiff::{AutoGrad, Var}};
    ///
    /// let cell = RNNCell::<f32>::new(2, 3, true);
    /// let x = Var::leaf(tensor![[1.0f32, 2.0]], false);
    ///
    /// let h1 = cell.forward(&x, &cell.zero_state(1));
    /// let h2 = cell.forward(&x, &h1);
    /// assert_eq!(h2.value().get_shape(), vec![1, 3]);
    /// ```
    ///
    /// # Arguments
    /// * `x` — the input at this step, of shape `[batch, input_size]`.
    /// * `h` — the hidden state carried over, of shape `[batch, hidden_size]`.
    ///
    /// # Returns
    /// The new hidden state, of shape `[batch, hidden_size]`.
    pub fn forward(&self, x: &VarRef<T>, h: &VarRef<T>) -> VarRef<T> {
        let pre = &(x * &self.w_ih) + &(h * &self.w_hh);

        let pre = match &self.bias {
            Some(b) => &pre + b,
            None => pre,
        };

        self.act.forward(&pre)
    }

    /// Returns the trainable parameters of the cell.
    ///
    /// # Returns
    /// The two weight matrices, the bias if there is one, and any parameters the
    /// activation carries.
    pub fn parameters(&self) -> Vec<VarRef<T>> {
        let mut params = vec![self.w_ih.clone(), self.w_hh.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params.extend(self.act.parameters());
        params
    }
}

/// A recurrent layer: runs an [RNNCell] over a whole sequence.
///
/// # Formula
///```math
///  h_t = \tanh(x_t W_{ih} + h_{t-1} W_{hh} + b), \qquad h_0 = 0
///```
///
/// # Example
/// ```
/// use tensorrs::{tensor, activation::Module, nn::{Linear, RNN, Sequential},
///                autodiff::{AutoGrad, Var}};
///
/// // one recurrent layer followed by a regression head
/// let model: Sequential<f32> = Sequential::new(vec![
///     Box::new(RNN::new(2, 8, true)),
///     Box::new(Linear::new(8, 1, true)),
/// ]);
///
/// // a batch of 2 sequences, 3 steps each, 2 features per step
/// let x = Var::leaf(tensor![
///     [[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]],
///     [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// ], false);
///
/// let y = model.forward(&x);
/// assert_eq!(y.value().get_shape(), vec![2, 1]);
/// ```
///
/// # Notes
/// The input is always `[batch, seq, features]`. By default only the final hidden
/// state comes back, which is what a head expects; ask for
/// [RNNOutput::Sequence] with [RNN::with_output] to get every step instead.
///
/// The graph is unrolled over the sequence, so `backward()` performs
/// backpropagation through time on its own. Memory grows with the sequence
/// length, since every step keeps its own nodes alive.
///
/// # See Also
/// [Wikipedia: Recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network)
pub struct RNN<T: Float> {
    cell: RNNCell<T>,
    output: RNNOutput,
}

impl<T: Float> RNN<T>
where
    Standard: Distribution<T>,
{
    /// Creates a recurrent layer returning the final hidden state.
    ///
    /// # Example
    /// ```
    /// use tensorrs::nn::RNN;
    ///
    /// let rnn = RNN::<f32>::new(10, 20, true);
    /// ```
    ///
    /// # Arguments
    /// * `input_size` — the number of features of one time step.
    /// * `hidden_size` — the size of the hidden state.
    /// * `bias` — whether to add a bias term.
    pub fn new(input_size: usize, hidden_size: usize, bias: bool) -> Self {
        Self {
            cell: RNNCell::new(input_size, hidden_size, bias),
            output: RNNOutput::Last,
        }
    }

    /// Creates a recurrent layer that returns what `output` asks for.
    ///
    /// # Example
    /// ```
    /// use tensorrs::{tensor, activation::Module, nn::{RNN, RNNOutput},
    ///                autodiff::{AutoGrad, Var}};
    ///
    /// let rnn = RNN::<f32>::with_output(2, 4, true, RNNOutput::Sequence);
    ///
    /// let x = Var::leaf(tensor![[[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]]], false);
    /// assert_eq!(rnn.forward(&x).value().get_shape(), vec![1, 3, 4]);
    /// ```
    ///
    /// # Arguments
    /// * `input_size` — the number of features of one time step.
    /// * `hidden_size` — the size of the hidden state.
    /// * `bias` — whether to add a bias term.
    /// * `output` — see [RNNOutput].
    ///
    /// # Notes
    /// Two layers stacked on each other need the lower one to return
    /// [RNNOutput::Sequence], so that the upper one still sees a sequence.
    pub fn with_output(
        input_size: usize,
        hidden_size: usize,
        bias: bool,
        output: RNNOutput,
    ) -> Self {
        Self {
            cell: RNNCell::new(input_size, hidden_size, bias),
            output,
        }
    }

    /// Creates a recurrent layer around a cell you built yourself.
    ///
    /// # Arguments
    /// * `cell` — the cell to run, see [RNNCell::with_activation].
    /// * `output` — see [RNNOutput].
    pub fn from_cell(cell: RNNCell<T>, output: RNNOutput) -> Self {
        Self { cell, output }
    }
}

impl<T: Float> RNN<T> {
    /// Returns the cell this layer runs.
    pub fn cell(&self) -> &RNNCell<T> {
        &self.cell
    }
}

impl<T: Float> Module<T> for RNN<T> {
    /// Runs the sequence through the cell, step by step.
    ///
    /// # Arguments
    /// * `x` — the input, of shape `[batch, seq, features]`.
    ///
    /// # Returns
    /// `[batch, hidden]` for [RNNOutput::Last], `[batch, seq, hidden]` for
    /// [RNNOutput::Sequence].
    ///
    /// # Panics
    /// If the input is not 3-D, or the sequence is empty.
    fn forward(&self, x: &VarRef<T>) -> VarRef<T> {
        let shape = x.value().get_shape();
        assert_eq!(shape.len(), 3,
            "!!!RNN expects an input of shape [batch, seq, features], got {shape:?}!!!");

        let (batch, seq) = (shape[0], shape[1]);
        assert!(seq > 0, "!!!RNN got an empty sequence!!!");

        let mut h = self.cell.zero_state(batch);
        let mut steps = Vec::new();

        for t in 0..seq {
            let x_t = select_op(x, 1, t);
            h = self.cell.forward(&x_t, &h);

            if matches!(self.output, RNNOutput::Sequence) {
                steps.push(h.clone());
            }
        }

        match self.output {
            RNNOutput::Sequence => stack_op(&steps, 1),
            RNNOutput::Last => h,
        }
    }

    fn parameters(&self) -> Vec<VarRef<T>> {
        self.cell.parameters()
    }

    /// Switches the activation of the cell into training mode.
    fn train(&mut self) {
        self.cell.act.train();
    }

    /// Switches the activation of the cell into inference mode.
    fn eval(&mut self) {
        self.cell.act.eval();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        loss::mse,
        optim::{Adam, Optimizer},
        tensor,
    };

    #[test]
    fn shapes_match_the_requested_output() {
        let x = Var::leaf(
            tensor![
                [[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]],
                [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]]
            ],
            false,
        );

        let last = RNN::<f32>::new(2, 5, true);
        assert_eq!(last.forward(&x).value().get_shape(), vec![2, 5]);

        let seq = RNN::<f32>::with_output(2, 5, true, RNNOutput::Sequence);
        assert_eq!(seq.forward(&x).value().get_shape(), vec![2, 3, 5]);
    }

    #[test]
    fn the_last_step_of_a_sequence_is_the_last_hidden_state() {
        // the same cell driven both ways has to agree
        let cell = RNNCell::<f64>::new(2, 3, true);
        let x = Var::leaf(
            tensor![[[1.0f64, 2.0], [3.0, 4.0], [5.0, 6.0]]],
            false,
        );

        let mut h = cell.zero_state(1);
        for t in 0..3 {
            let x_t = select_op(&x, 1, t);
            h = cell.forward(&x_t, &h);
        }

        let rnn = RNN::from_cell(cell, RNNOutput::Sequence);
        let out = rnn.forward(&x);
        let last_step = out.value().select(1, 2);

        assert_eq!(last_step.get_data(), h.value().get_data());
    }

    #[test]
    fn gradients_reach_every_parameter() {
        let rnn = RNN::<f64>::new(2, 4, true);
        let x = Var::leaf(tensor![[[1.0f64, 2.0], [3.0, 4.0]]], false);
        let y = Var::leaf(tensor![[0.5f64, 0.5, 0.5, 0.5]], false);

        let loss = mse(&rnn.forward(&x), &y);
        loss.backward();

        for (i, p) in rnn.parameters().iter().enumerate() {
            let g = p.grad().get_data();
            assert_eq!(g.len(), p.value().get_data().len(),
                "parameter {i} got a gradient of the wrong size");
            assert!(g.iter().any(|&v| v != 0.0),
                "parameter {i} never received a gradient");
        }
    }

    #[test]
    fn learns_to_sum_a_sequence() {
        // the target is the sum of every step, so the state has to carry
        // information across time for this to be solvable at all
        let rnn = RNN::<f64>::new(1, 12, true);
        let head = crate::nn::Linear::<f64>::new(12, 1, true);

        let x = Var::leaf(
            tensor![
                [[0.1f64], [0.2], [0.3]],
                [[0.5f64], [0.1], [0.1]],
                [[0.2f64], [0.2], [0.2]],
                [[0.4f64], [0.3], [0.2]]
            ],
            false,
        );
        let y = Var::leaf(tensor![[0.6f64], [0.7], [0.6], [0.9]], false);

        let mut params = rnn.parameters();
        params.extend(head.parameters());
        let mut optim = Adam::new(params, 0.05);

        let first = mse(&head.forward(&rnn.forward(&x)), &y).value().item();

        for _ in 0..400 {
            optim.zero_grad();
            let loss = mse(&head.forward(&rnn.forward(&x)), &y);
            loss.backward();
            optim.step();
        }

        let last = mse(&head.forward(&rnn.forward(&x)), &y).value().item();
        assert!(last < first * 0.01,
            "the loss barely moved: {first} -> {last}");
    }
}
