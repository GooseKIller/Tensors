use std::{cell::RefCell, rc::Rc};

use crate::{Float, autodiff::{OpKind, core::{Var, VarRef}}, linalg::Tensor};

impl<T: Float> VarRef<T> {
    pub fn sum(&self) -> Self{
        sum_op(self)
    }
}

/// Addition: builds a new VarRef and keeps strong references to the parents.
pub fn add_op<T: Float>(a: &VarRef<T>, b: &VarRef<T>) -> VarRef<T> {
    let va = a.borrow().value.shallow_copy();
    let vb = b.borrow().value.shallow_copy();
    let out_value = &va + &vb;
    let shape = out_value.get_shape();
    let out_zero = Tensor::from_num(T::default(), shape.clone());
    let requires_grad = a.borrow().requires_grad || b.borrow().requires_grad;

    let out = VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::Add(a.clone(), b.clone()),
        requires_grad,
        cached_topo: RefCell::new(None),
    })));
    out
}

pub fn sum_op<T:Float>(x: &VarRef<T>) -> VarRef<T> {
    let x_val = x.borrow().value.shallow_copy();
    let out_value = x_val.sum();

    let out = VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(Tensor::scalar(T::default())),
        op: OpKind::Sum(x.clone()),
        requires_grad: x.borrow().requires_grad,            
        cached_topo: RefCell::new(None),
    })));
    out
}

pub fn sum_axis_op<T: Float>(a: &VarRef<T>, axis: usize, keepdim: bool) -> VarRef<T> {
    let va = a.borrow().value.shallow_copy();
    
    let out_value = if keepdim {
        va.sum_axis_keepdim(axis)
    } else {
        va.sum_axis(axis)
    };
    
    let shape = out_value.get_shape();
    let out_zero = Tensor::from_num(T::default(), shape.clone());
    
    let requires_grad = a.borrow().requires_grad;

    let out = VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::SumAxis(a.clone(), axis, keepdim),
        requires_grad,
        cached_topo: RefCell::new(None),
    })));
    
    out
}

pub fn sub_op<T: Float>(a: &VarRef<T>, b: &VarRef<T>) -> VarRef<T> {
    let va = a.borrow().value.shallow_copy();
    let vb = b.borrow().value.shallow_copy();
    let out_value = &va - &vb;
    let shape = out_value.get_shape();
    let out_zero = Tensor::from_num(T::default(), shape.clone());
    let requires_grad = a.borrow().requires_grad || b.borrow().requires_grad;

    let out = VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::Sub(a.clone(), b.clone()),
        requires_grad,
        cached_topo: RefCell::new(None),
    })));
    out
}

/// Element-wise multiplication: builds a new VarRef and keeps strong references to the parents.
pub fn mul_op<T: Float>(a: &VarRef<T>, b: &VarRef<T>) -> VarRef<T> {
    let va = a.borrow().value.shallow_copy();
    let vb = b.borrow().value.shallow_copy();
    let out_value = &va & (&vb);
    let shape = out_value.get_shape();
    let out_zero = Tensor::from_num(T::default(), shape.clone());
    let requires_grad = a.borrow().requires_grad || b.borrow().requires_grad;

    let out = VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::Mul(a.clone(), b.clone()),
        requires_grad,
        cached_topo: RefCell::new(None),
    })));
    out
}

pub fn powf_op<T:Float>(a: &VarRef<T>, b: &VarRef<T>) -> VarRef<T> {
    let va = a.borrow().value.shallow_copy();
    let vb = b.borrow().value.shallow_copy();
    let out_value = &va^(&vb);
    let shape = out_value.get_shape();
    let out_zero = Tensor::from_num(T::default(), shape.clone());
    let requires_grad = a.borrow().requires_grad || b.borrow().requires_grad;

    let out = VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::Pow(a.clone(), b.clone()),
        requires_grad,
        cached_topo: RefCell::new(None),
    })));
    out
}

pub fn div_op<T: Float>(a: &VarRef<T>, b: &VarRef<T>) -> VarRef<T> {
    let va = a.borrow().value.shallow_copy();
    let vb = b.borrow().value.shallow_copy();

    let out_value = &va / &vb;
    let shape = out_value.get_shape();
    let out_zero = Tensor::from_num(T::default(), shape.clone());
    let requires_grad = a.borrow().requires_grad || b.borrow().requires_grad;

    let out = VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::Div(a.clone(), b.clone()),
        requires_grad,
        cached_topo: RefCell::new(None),
    })));
    out
}

pub fn matmul_op<T: Float>(a: &VarRef<T>, b: &VarRef<T>) -> VarRef<T> {
    let va = a.borrow().value.shallow_copy();
    let vb = b.borrow().value.shallow_copy();

    let out_value = &va * &vb;
    let shape = out_value.get_shape();
    let out_zero = Tensor::from_num(T::default(), shape.clone());
    let requires_grad = a.borrow().requires_grad || b.borrow().requires_grad;

    let out = VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::MatMul(a.clone(), b.clone()),
        requires_grad,
        cached_topo: RefCell::new(None),
    })));
    out
}

pub fn log_op<T:Float>(a: &VarRef<T>) -> VarRef<T> {
    let va = a.borrow().value.shallow_copy();
    let out_value = va.map(|x| x.ln()); // by way of the ln method on Float
    
    let shape = out_value.get_shape();
    let out_zero = Tensor::from_num(T::default(), shape);
    
    VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::Log(a.clone()),
        requires_grad: a.borrow().requires_grad,
        cached_topo: RefCell::new(None),
    })))
}

/// Element-wise absolute value.
///
/// # Formula
///```math
///  f(x) = |x| \qquad \frac{\partial f}{\partial x} = \operatorname{sign}(x)
///```
///
/// # Example
/// ```
/// use tensorrs::{tensor, autodiff::{AutoGrad, Var, abs_op}};
///
/// let x = Var::leaf(tensor![-2.0f32, 0.0, 3.0], true);
/// let y = abs_op(&x);
/// assert_eq!(y.value().get_data(), vec![2.0, 0.0, 3.0]);
///
/// y.sum().backward();
/// assert_eq!(x.grad().get_data(), vec![-1.0, 0.0, 1.0]);
/// ```
///
/// # Notes
/// The derivative is taken as `0` at `x = 0`, where `|x|` has no real one.
pub fn abs_op<T: Float>(a: &VarRef<T>) -> VarRef<T> {
    let va = a.borrow().value.shallow_copy();
    let out_value = va.map(|x| x.abs());

    let shape = out_value.get_shape();
    let out_zero = Tensor::from_num(T::default(), shape);

    VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::Abs(a.clone()),
        requires_grad: a.borrow().requires_grad,
        cached_topo: RefCell::new(None),
    })))
}

/// Element-wise clamp into `[min, max]`.
///
/// # Formula
///```math
///  f(x) = \min(\max(x, a), b) \qquad
///  \frac{\partial f}{\partial x} = \begin{cases} 1, & a \le x \le b \\ 0, & \text{otherwise} \end{cases}
///```
///
/// # Example
/// ```
/// use tensorrs::{tensor, autodiff::{AutoGrad, Var, clamp_op}};
///
/// let x = Var::leaf(tensor![-1.0f32, 0.5, 4.0], true);
/// let y = clamp_op(&x, 0.0, 1.0);
/// assert_eq!(y.value().get_data(), vec![0.0, 0.5, 1.0]);
///
/// // only the value that was left alone keeps its gradient
/// y.sum().backward();
/// assert_eq!(x.grad().get_data(), vec![0.0, 1.0, 0.0]);
/// ```
///
/// # Arguments
/// * `a` — the node to clamp.
/// * `min` — the lower bound.
/// * `max` — the upper bound.
///
/// # Notes
/// Values pushed to a bound stop receiving a gradient, since the output no longer
/// follows the input there. That is what keeps a saturated probability from
/// blowing up the logarithm in [cross_entropy](crate::loss::cross_entropy).
pub fn clamp_op<T: Float>(a: &VarRef<T>, min: T, max: T) -> VarRef<T> {
    let va = a.borrow().value.shallow_copy();
    let out_value = va.map(|x| if x < min { min } else if x > max { max } else { x });

    let shape = out_value.get_shape();
    let out_zero = Tensor::from_num(T::default(), shape);

    VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::Clamp(a.clone(), min, max),
        requires_grad: a.borrow().requires_grad,
        cached_topo: RefCell::new(None),
    })))
}

/// Takes the slice at `index` along `axis`, dropping that dimension.
///
/// # Example
/// ```
/// use tensorrs::{tensor, autodiff::{AutoGrad, Var, select_op}};
///
/// let x = Var::leaf(tensor![[1.0f32, 2.0], [3.0, 4.0]], true);
/// let row = select_op(&x, 0, 1);
/// assert_eq!(row.value().get_data(), vec![3.0, 4.0]);
///
/// // only the selected row is on the path to the output
/// row.sum().backward();
/// assert_eq!(x.grad().get_data(), vec![0.0, 0.0, 1.0, 1.0]);
/// ```
///
/// # Arguments
/// * `a` — the node to index into.
/// * `axis` — the dimension to index.
/// * `index` — the position along that dimension.
///
/// # Notes
/// The graph counterpart of [Tensor::select]. Pulling one time step out of a
/// `[batch, seq, features]` input is what drives the loop in
/// [RNN](crate::nn::RNN); [stack_op] puts the results back together.
pub fn select_op<T: Float>(a: &VarRef<T>, axis: usize, index: usize) -> VarRef<T> {
    let out_value = a.borrow().value.select(axis, index);

    let shape = out_value.get_shape();
    let out_zero = Tensor::from_num(T::default(), shape);

    VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::Select(a.clone(), axis, index),
        requires_grad: a.borrow().requires_grad,
        cached_topo: RefCell::new(None),
    })))
}

/// Joins nodes of an equal shape along a new axis.
///
/// # Example
/// ```
/// use tensorrs::{tensor, autodiff::{AutoGrad, Var, stack_op}};
///
/// let a = Var::leaf(tensor![1.0f32, 2.0], true);
/// let b = Var::leaf(tensor![3.0f32, 4.0], true);
///
/// let s = stack_op(&[a.clone(), b.clone()], 0);
/// assert_eq!(s.value().get_shape(), vec![2, 2]);
/// assert_eq!(s.value().get_data(), vec![1.0, 2.0, 3.0, 4.0]);
///
/// s.sum().backward();
/// assert_eq!(a.grad().get_data(), vec![1.0, 1.0]);
/// ```
///
/// # Arguments
/// * `parts` — the nodes to join; all of them must share one shape.
/// * `axis` — where to insert the new dimension.
///
/// # Panics
/// If `parts` is empty, or the shapes disagree.
///
/// # Notes
/// The graph counterpart of [Tensor::stack] and the inverse of [select_op]. A
/// node listed twice accumulates both gradients, as anywhere else in the graph.
pub fn stack_op<T: Float>(parts: &[VarRef<T>], axis: usize) -> VarRef<T> {
    assert!(!parts.is_empty(), "!!!stack_op(): needs at least one node!!!");

    let values: Vec<Tensor<T>> = parts
        .iter()
        .map(|p| p.borrow().value.shallow_copy())
        .collect();
    let out_value = Tensor::stack(&values, axis);

    let requires_grad = parts.iter().any(|p| p.borrow().requires_grad);
    let out_zero = Tensor::from_num(T::default(), out_value.get_shape());

    VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::Stack(parts.to_vec(), axis),
        requires_grad,
        cached_topo: RefCell::new(None),
    })))
}

/// Rearranges a node into a new shape, keeping the elements in the same order.
///
/// # Example
/// ```
/// use tensorrs::{tensor, autodiff::{AutoGrad, Var, reshape_op}};
///
/// let x = Var::leaf(tensor![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], true);
/// let y = reshape_op(&x, vec![3, 2]);
///
/// assert_eq!(y.value().get_shape(), vec![3, 2]);
/// assert_eq!(y.value().get_data(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
///
/// # Arguments
/// * `a` — the node to reshape.
/// * `shape` — the new shape; it has to hold the same number of elements.
///
/// # Panics
/// If the element count changes.
///
/// # Notes
/// The backward pass only reshapes the gradient back, so this costs nothing but
/// a copy in either direction.
pub fn reshape_op<T: Float>(a: &VarRef<T>, shape: Vec<usize>) -> VarRef<T> {
    let packed = a.borrow().value.packed_data();
    let out_value = Tensor::new(packed, shape);

    let out_zero = Tensor::from_num(T::default(), out_value.get_shape());

    VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::Reshape(a.clone()),
        requires_grad: a.borrow().requires_grad,
        cached_topo: RefCell::new(None),
    })))
}

/// Reorders the dimensions of a node.
///
/// # Example
/// ```
/// use tensorrs::{tensor, autodiff::{AutoGrad, Var, permute_op}};
///
/// let x = Var::leaf(tensor![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], true);
/// let y = permute_op(&x, &[1, 0]);
///
/// assert_eq!(y.value().get_shape(), vec![3, 2]);
/// assert_eq!(y.value().get_data(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
/// ```
///
/// # Arguments
/// * `a` — the node to permute.
/// * `axes` — a permutation of `0..ndim`, where `axes[i]` is the old position of
///   the new `i`-th dimension.
///
/// # Panics
/// If `axes` is not a permutation of the dimensions of `a`.
///
/// # Notes
/// The graph counterpart of [Tensor::permute]. Unlike that method the result is
/// repacked rather than left as a strided view, so it can be reshaped afterwards.
/// The backward pass applies the inverse permutation.
pub fn permute_op<T: Float>(a: &VarRef<T>, axes: &[usize]) -> VarRef<T> {
    let view = a
        .borrow()
        .value
        .permute(axes)
        .expect("!!!permute_op(): axes must be a permutation of the dimensions!!!");
    let out_value = Tensor::new(view.packed_data(), view.get_shape());

    let out_zero = Tensor::from_num(T::default(), out_value.get_shape());

    VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::Permute(a.clone(), axes.to_vec()),
        requires_grad: a.borrow().requires_grad,
        cached_topo: RefCell::new(None),
    })))
}

/// Lays every sliding window of a node out as a row.
///
/// # Example
/// ```
/// use tensorrs::{linalg::Tensor, autodiff::{AutoGrad, Var, unfold_op}};
///
/// let x = Var::leaf(Tensor::new((1..=9).map(|v| v as f32).collect(), vec![1, 1, 3, 3]), true);
/// let cols = unfold_op(&x, (2, 2), (1, 1), (0, 0));
///
/// assert_eq!(cols.value().get_shape(), vec![4, 4]);
///
/// // the centre pixel is read by all four windows
/// cols.sum().backward();
/// assert_eq!(x.grad().get_data(), vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0]);
/// ```
///
/// # Arguments
/// * `a` — the input, of shape `[batch, channels, height, width]`.
/// * `kernel`, `stride`, `padding` — see [Tensor::unfold_2d].
///
/// # Notes
/// The graph counterpart of [Tensor::unfold_2d]; the backward pass is
/// [Tensor::fold_2d]. Followed by a matrix multiplication this is a convolution,
/// which is how [Conv2d](crate::nn::Conv2d) is built — and it means the gradient
/// of the convolution comes from the existing matmul rule rather than a rule of
/// its own.
pub fn unfold_op<T: Float>(
    a: &VarRef<T>,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> VarRef<T> {
    let out_value = a.borrow().value.unfold_2d(kernel, stride, padding);
    let out_zero = Tensor::from_num(T::default(), out_value.get_shape());

    VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::Unfold(a.clone(), kernel, stride, padding),
        requires_grad: a.borrow().requires_grad,
        cached_topo: RefCell::new(None),
    })))
}

/// Takes the largest value along `axis`, dropping that dimension.
///
/// # Example
/// ```
/// use tensorrs::{tensor, autodiff::{AutoGrad, Var, max_axis_op}};
///
/// let x = Var::leaf(tensor![[1.0f32, 5.0, 3.0], [4.0, 2.0, 6.0]], true);
/// let y = max_axis_op(&x, 1);
/// assert_eq!(y.value().get_data(), vec![5.0, 6.0]);
///
/// // only the winners are on the path to the output
/// y.sum().backward();
/// assert_eq!(x.grad().get_data(), vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
/// ```
///
/// # Arguments
/// * `a` — the node to reduce.
/// * `axis` — the dimension to take the maximum over.
///
/// # Panics
/// If `axis` is out of range, or the axis is empty.
///
/// # Notes
/// The graph counterpart of [Tensor::max_axis]. A maximum is not differentiable
/// where two values tie; the gradient then goes entirely to the lower index,
/// which is the same choice [Tensor::argmax_axis] makes.
///
/// Combined with [unfold_op] this is max pooling — see
/// [MaxPool2d](crate::nn::MaxPool2d).
pub fn max_axis_op<T: Float>(a: &VarRef<T>, axis: usize) -> VarRef<T> {
    let out_value = a.borrow().value.max_axis(axis);
    let out_zero = Tensor::from_num(T::default(), out_value.get_shape());

    VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::MaxAxis(a.clone(), axis),
        requires_grad: a.borrow().requires_grad,
        cached_topo: RefCell::new(None),
    })))
}

/// Hyperbolic tangent, element-wise.
///
/// # Formula
///```math
///  \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} \qquad
///  \frac{\partial}{\partial x} \tanh(x) = 1 - \tanh^2(x)
///```
///
/// # Example
/// ```
/// use tensorrs::{tensor, autodiff::{AutoGrad, Var, tanh_op}};
///
/// let x = Var::leaf(tensor![[0.0f32, 90.0, -90.0]], true);
/// let y = tanh_op(&x);
///
/// // saturates cleanly instead of overflowing
/// assert_eq!(y.value().get_data(), vec![0.0, 1.0, -1.0]);
///
/// // and a saturated input gets no gradient, as it should
/// y.sum().backward();
/// assert_eq!(x.grad().get_data(), vec![1.0, 0.0, 0.0]);
/// ```
///
/// # Notes
/// Built as one node rather than assembled from exponentials. Writing it out as
/// `(e^x - e^-x) / (e^x + e^-x)` overflows at `|x| > 88` in `f32` and turns into
/// `inf / inf`, and even below that the chain rule through those five nodes loses
/// the derivative — at `x = 50` it reports `1` where the true value is `0`.
///
/// The derivative is read straight off the output, so the backward pass costs one
/// multiply per element and no exponentials at all.
pub fn tanh_op<T: Float>(a: &VarRef<T>) -> VarRef<T> {
    let out_value = a.borrow().value.map(|x| x.tanh());
    let out_zero = Tensor::from_num(T::default(), out_value.get_shape());

    VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::Tanh(a.clone()),
        requires_grad: a.borrow().requires_grad,
        cached_topo: RefCell::new(None),
    })))
}

/// Logistic sigmoid, element-wise.
///
/// # Formula
///```math
///  \sigma(x) = \frac{1}{1 + e^{-x}} \qquad
///  \frac{\partial}{\partial x} \sigma(x) = \sigma(x)\,(1 - \sigma(x))
///```
///
/// # Example
/// ```
/// use tensorrs::{tensor, autodiff::{AutoGrad, Var, sigmoid_op}};
///
/// let x = Var::leaf(tensor![[0.0f32, 400.0, -400.0]], true);
/// let y = sigmoid_op(&x);
///
/// assert_eq!(y.value().get_data(), vec![0.5, 1.0, 0.0]);
///
/// y.sum().backward();
/// assert_eq!(x.grad().get_data(), vec![0.25, 0.0, 0.0]);
/// ```
///
/// # Notes
/// Evaluated as `1 / (1 + e^{-x})` for `x >= 0` and `e^x / (1 + e^x)` below it, so
/// the exponent is never positive and can never overflow. The naive single form
/// produces a `NaN` gradient for large negative inputs.
pub fn sigmoid_op<T: Float>(a: &VarRef<T>) -> VarRef<T> {
    let out_value = a.borrow().value.map(|x| {
        if x >= T::default() {
            T::one() / (T::one() + (-x).exp())
        } else {
            // the exponent stays negative on this side, so it cannot overflow
            let e = x.exp();
            e / (T::one() + e)
        }
    });
    let out_zero = Tensor::from_num(T::default(), out_value.get_shape());

    VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::Sigmoid(a.clone()),
        requires_grad: a.borrow().requires_grad,
        cached_topo: RefCell::new(None),
    })))
}

/// Picks rows out of a table by index.
///
/// # Example
/// ```
/// use tensorrs::{linalg::Tensor, autodiff::{AutoGrad, Var, gather_op}};
///
/// let table = Var::leaf(Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]), true);
///
/// // the first row is taken twice
/// let picked = gather_op(&table, &[0, 0, 1]);
/// assert_eq!(picked.value().get_data(), vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0]);
///
/// picked.sum().backward();
/// assert_eq!(table.grad().get_data(), vec![2.0, 2.0, 1.0, 1.0]);
/// ```
///
/// # Arguments
/// * `a` — the table, of shape `[rows, columns]`.
/// * `indices` — which rows to take, in order; repeats are allowed.
///
/// # Panics
/// If `a` is not 2-D, or an index is past the end of the table.
///
/// # Notes
/// The graph counterpart of [Tensor::gather_rows]; the backward pass is
/// [Tensor::scatter_add_rows]. A row taken several times collects a gradient from
/// every occurrence, which is what makes a repeated token train from all of its
/// appearances at once. This is how [Embedding](crate::nn::Embedding) works.
pub fn gather_op<T: Float>(a: &VarRef<T>, indices: &[usize]) -> VarRef<T> {
    let out_value = a.borrow().value.gather_rows(indices);
    let out_zero = Tensor::from_num(T::default(), out_value.get_shape());

    VarRef(Rc::new(RefCell::new(Var {
        value: out_value,
        grad: RefCell::new(out_zero),
        op: OpKind::Gather(a.clone(), Rc::new(indices.to_vec())),
        requires_grad: a.borrow().requires_grad,
        cached_topo: RefCell::new(None),
    })))
}
