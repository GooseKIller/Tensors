//! # Building Blocks for Neural Networks
//!
//! 1.[Linear] - Fully connected layer
//!
//! 2.[Sequential] - Container that stacks layers into a model
//!
//! 3.[RNN] - Recurrent layer, and [RNNCell] for driving one step at a time
//!
//! 4.[LayerNorm] - Layer normalization
//!
//! 5.[Dropout] - Randomly zeroes activations while training
//!
//! 6.[Conv2d] - 2-D convolution, with [Flatten] to join it to a dense head
//!
//! 7.[MaxPool2d] and [AvgPool2d] - Downsampling by pooling windows
//!
//! 8.[MultiHeadAttention] - Attention over a sequence, with [causal_mask] for decoders
//!
//! 9.[Embedding] and [PositionalEncoding] - Token ids into vectors, and their order
//!
//! Every one of them implements [Module](crate::activation::Module), so they can
//! be mixed freely inside a [Sequential] model.

mod attention;
mod conv2d;
mod embedding;
mod flatten;
mod linear;
mod pool;
mod rnn;
mod sequential;
mod dropout;
mod layer_norm;

pub use attention::*;
pub use conv2d::*;
pub use embedding::*;
pub use flatten::*;
pub use linear::*;
pub use pool::*;
pub use rnn::*;
pub use sequential::*;
pub use dropout::*;
pub use layer_norm::*;
