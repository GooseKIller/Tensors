use std::{cell::RefCell, collections::HashSet, fmt::Display, rc::{Rc, Weak}};
use crate::{Float, linalg::Tensor};

pub struct Var<T: Float> {
    pub value: Tensor<T>,
    pub grad: RefCell<Tensor<T>>,
    pub(crate) op: OpKind<T>,
    pub requires_grad: bool,
    /// The topological order, remembered between `zero_grad` and `backward`.
    ///
    /// Weak, not strong: the order includes the node itself, so owning it would
    /// make the node keep itself alive - a cycle `Rc` can never collect, leaking
    /// the whole graph on every backward pass. The graph is already held by the
    /// chain of parents for as long as the root lives, so the cache has nothing
    /// to own.
    pub(crate) cached_topo: RefCell<Option<Vec<Weak<RefCell<Var<T>>>>>>,
}

#[derive(Clone)]
pub struct VarRef<T: Float>(pub Rc<RefCell<Var<T>>>);

impl<T: Float> Var<T> {
    pub fn leaf(value: Tensor<T>, requires_grad: bool) -> VarRef<T> {
        let shape = value.get_shape();
        let zero = Tensor::from_num(T::default(), shape);
        VarRef(Rc::new(RefCell::new(Var {
            value,
            grad: RefCell::new(zero),
            op: OpKind::Leaf,
            requires_grad,
            cached_topo: RefCell::new(None),
        })))
    }

}

impl<T: Float> VarRef<T> {
    /// Convenient accessors
    pub fn rc(&self) -> &Rc<RefCell<Var<T>>> {
        &self.0
    }

    pub fn borrow(&self) -> std::cell::Ref<'_, Var<T>> {
        self.0.borrow()
    }

    pub fn borrow_mut(&self) -> std::cell::RefMut<'_, Var<T>> {
        self.0.borrow_mut()
    }
}

impl<T: Float> Display for VarRef<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let var = self.0.borrow();
        
        // 1. A short name for the operation
        let op_name = match &var.op {
            OpKind::Leaf => if var.requires_grad { "Param" } else { "Input" },
            OpKind::Add(_, _) => "Add",
            OpKind::Mul(_, _) => "Mul",
            OpKind::Log(_) => "Log",
            OpKind::MatMul(_, _) => "MatMul",
            OpKind::Pow(_, _) => "Pow",
            OpKind::Sub(_, _) => "Sub",
            OpKind::Sum(_) => "Sum",
            OpKind::SumAxis(_, _, _) => "SumAxis",
            _ => "Op", 
        };

        // 2. Print the shape and the kind of operation
        write!(f, "Var({} | shape: {:?} | req_grad: {})", 
            op_name, 
            var.value.shape, 
            var.requires_grad
        )?;

        // 3. If the tensor is small (a scalar, or 1-D up to 5 elements), the
        // value itself can be printed through the Tensor Display
        if var.value.get_data().len() <= 25 {
            write!(f, " => {}", var.value)?;
        }

        Ok(())
    }
}


/// Builds the topological order of the nodes, starting from the root.
/// Returns them in traversal order: the parents first, then the node itself.
#[allow(dead_code)]
pub(crate) fn build_topo<T: Float>(root: &VarRef<T>) -> Vec<VarRef<T>> {
    let mut order_rcs: Vec<VarRef<T>> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();

    fn dfs_rec<T: Float>(node: VarRef<T>, visited: &mut HashSet<usize>, order: &mut Vec<VarRef<T>>) {
        let key = Rc::as_ptr(&node.0) as usize;
        if visited.contains(&key) { return; }
        visited.insert(key);

        let parents = {
            let n = node.borrow();
            n.op.parents()
        };
        for p in parents {
            dfs_rec(p, visited, order);
        }

        order.push(node);
    }

    dfs_rec(root.clone(), &mut visited, &mut order_rcs);

    order_rcs.into_iter().collect()
}

#[derive(Clone)]
pub(crate) enum OpKind<T: Float> {
    Leaf,
    Add(VarRef<T>, VarRef<T>),
    Sum(VarRef<T>),
    Sub(VarRef<T>, VarRef<T>),
    Mul(VarRef<T>, VarRef<T>),
    Pow(VarRef<T>, VarRef<T>),
    Div(VarRef<T>, VarRef<T>),
    MatMul(VarRef<T>, VarRef<T>),
    SumAxis(VarRef<T>, usize, bool),
    Log(VarRef<T>),
    Abs(VarRef<T>),
    Clamp(VarRef<T>, T, T),
    Select(VarRef<T>, usize, usize),
    Stack(Vec<VarRef<T>>, usize),
    Reshape(VarRef<T>),
    Permute(VarRef<T>, Vec<usize>),
    Unfold(VarRef<T>, (usize, usize), (usize, usize), (usize, usize)),
    MaxAxis(VarRef<T>, usize),
    Tanh(VarRef<T>),
    Sigmoid(VarRef<T>),
    Gather(VarRef<T>, std::rc::Rc<Vec<usize>>),
}

impl<T:Float> OpKind<T> {
    fn parents(&self) -> Vec<VarRef<T>> {
        match self {
            OpKind::Leaf => vec![],
            OpKind::Add(a, b) => vec![a.clone(), b.clone()],
            OpKind::Sum(a) => vec![a.clone()],
            OpKind::Sub(a, b) => vec![a.clone(), b.clone()],
            OpKind::Mul(a, b) => vec![a.clone(), b.clone()],
            OpKind::Pow(a, b) => vec![a.clone(), b.clone()],
            OpKind::Div(a, b) => vec![a.clone(), b.clone()],
            OpKind::MatMul(a, b) => vec![a.clone(), b.clone()],
            OpKind::SumAxis(a, _, _) => vec![a.clone()],
            OpKind::Log(a) => vec![a.clone()],
            OpKind::Abs(a) => vec![a.clone()],
            OpKind::Clamp(a, _, _) => vec![a.clone()],
            OpKind::Select(a, _, _) => vec![a.clone()],
            OpKind::Stack(parts, _) => parts.clone(),
            OpKind::Reshape(a) => vec![a.clone()],
            OpKind::Permute(a, _) => vec![a.clone()],
            OpKind::Unfold(a, _, _, _) => vec![a.clone()],
            OpKind::MaxAxis(a, _) => vec![a.clone()],
            OpKind::Tanh(a) => vec![a.clone()],
            OpKind::Sigmoid(a) => vec![a.clone()],
            OpKind::Gather(a, _) => vec![a.clone()],
        }
    }

    #[allow(dead_code)]
    pub(crate) fn backward(&self, out_grad: &Tensor<T>, out_value: &Tensor<T>) 
        -> Vec<(VarRef<T>, Tensor<T>)> {
        match self {
            OpKind::Leaf => vec![],
            OpKind::Add(a, b) => {
                // For Add with broadcasting, gradient goes to both but must be reduced to original shapes
                let ga_full = out_grad.shallow_copy();
                let gb_full = out_grad.shallow_copy();
                
                let a_shape = a.borrow().value.get_shape();
                let b_shape = b.borrow().value.get_shape();
                
                let ga = if ga_full.get_shape().as_slice() != a_shape.as_slice() {
                    ga_full.reduce_to_shape(&a_shape).unwrap_or(ga_full)
                } else {
                    ga_full
                };
                let gb = if gb_full.get_shape().as_slice() != b_shape.as_slice() {
                    gb_full.reduce_to_shape(&b_shape).unwrap_or(gb_full)
                } else {
                    gb_full
                };
                
                vec![(a.clone(), ga), (b.clone(), gb)]
            }
            OpKind::Sum(x) => {
                debug_assert!(out_grad.is_scalar());
                let x_shape = x.borrow().value.get_shape();
                let g = Tensor::from_num(out_grad.item(), x_shape);
                vec![(x.clone(), g)]
            }
            OpKind::Sub(a, b) => {
                vec![
                    (a.clone(), out_grad.shallow_copy()),
                    (b.clone(), -&out_grad.shallow_copy()),
                ]
            }
            OpKind::Mul(a, b) => {
                // grad_a = out_grad * b
                // grad_b = out_grad * a
                let a_val = a.borrow().value.shallow_copy();
                let b_val = b.borrow().value.shallow_copy();
                let ga_full = out_grad & (&b_val);
                let gb_full = out_grad & (&a_val);
                
                // Reduce gradients to match original shapes
                let ga = if ga_full.get_shape().as_slice() != a_val.get_shape().as_slice() {
                    ga_full.reduce_to_shape(&a_val.get_shape()).unwrap_or(ga_full)
                } else {
                    ga_full
                };
                let gb = if gb_full.get_shape().as_slice() != b_val.get_shape().as_slice() {
                    gb_full.reduce_to_shape(&b_val.get_shape()).unwrap_or(gb_full)
                } else {
                    gb_full
                };
                
                vec![(a.clone(), ga), (b.clone(), gb)]
            }
            OpKind::Pow(a, b) => {
                let z = out_value.shallow_copy();
                let a_val = a.borrow().value.shallow_copy();
                let b_val = b.borrow().value.shallow_copy();

                // grad wrt a: out_grad * b * z / a
                let ga = out_grad & (&b_val) & &(&z / &a_val);

                // grad wrt b: out_grad * z * ln(a)
                let ln_a = a_val.map(|x| x.ln());
                let gb = out_grad & (&z) & (&ln_a);

                vec![(a.clone(), ga), (b.clone(), gb)]
            }
            OpKind::Div(a, b) => {
                let a_val = a.borrow().value.shallow_copy();
                let b_val = b.borrow().value.shallow_copy();
                // grad_a = out_grad / b
                let ga = out_grad / &b_val;
                
                // grad_b = - out_grad * a / b^2
                let b2 = &b_val & &b_val;
                let gb_full = -(&(out_grad & &a_val) / &b2);
                
                // Reduce gb to b's shape by summing over dimensions where b was broadcast
                let gb = if gb_full.get_shape().as_slice() != b_val.get_shape().as_slice() {
                    gb_full.reduce_to_shape(&b_val.get_shape()).unwrap_or(gb_full)
                } else {
                    gb_full
                };
                
                vec![(a.clone(), ga), (b.clone(), gb)]
            }
            OpKind::MatMul(a, b) => {
                let a_val = a.borrow().value.shallow_copy();
                let b_val = b.borrow().value.shallow_copy();

                let ndim_a = a_val.get_shape().len();
                let ndim_b = b_val.get_shape().len();

                assert!(ndim_a >= 2 && ndim_b >= 2, "!!!Tensor must be at least 2D!!!");
                
                // transpose (a permute of the last two axes) as a view
                let mut axes_a_t: Vec<usize> = (0..ndim_a).collect();
                axes_a_t.swap(ndim_a - 1, ndim_a - 2);
                
                let mut axes_b_t: Vec<usize> = (0..ndim_b).collect();
                axes_b_t.swap(ndim_b - 1, ndim_b - 2);

                let a_t = a_val.permute(&axes_a_t).unwrap(); // A^T view
                let b_t = b_val.permute(&axes_b_t).unwrap(); // B^T view

                // Correct:
                // dA = dC @ B^T
                let ga = out_grad.matmul(&b_t);

                // dB = A^T @ dC
                let gb = a_t.matmul(out_grad);

                vec![
                    (a.clone(), ga),
                    (b.clone(), gb)
                ]
            }


            OpKind::SumAxis(a, _axis, _keepdim) => {
                let a_shape = a.borrow().value.shape.clone();
                let grad_tensor = out_grad.clone();

                // Handle broadcasting in backward: gradient might need to be reduced
                // to match the original input shape
                let ga = if grad_tensor.get_shape().as_slice() != a_shape.as_slice() {
                    // Try reducing to match a's shape
                    if let Some(reduced) = grad_tensor.reduce_to_shape(&a_shape) {
                        reduced
                    } else {
                        // Try expanding: if grad is [101, 1] and we need [101, 16],
                        // broadcast it first then reduce
                        if let Some(broadcast) = grad_tensor.broadcast_to(&a_shape) {
                            broadcast
                        } else {
                            // Last resort: just create tensor with right shape filled with sum
                            // (this is a fallback for broken gradients)
                            eprintln!("WARN: SumAxis backward failed to reduce {:?} to {:?}", grad_tensor.shape, a_shape);
                            let data = grad_tensor.packed_data();
                            let total = data.iter().fold(T::default(), |acc, &x| acc + x);
                            Tensor::from_num(total, a_shape)
                        }
                    }
                } else {
                    grad_tensor
                };
                vec![(a.clone(), ga)]
            }
            OpKind::Log(a) => {
                // d(ln(x))/dx = 1/x
                let ga = out_grad / &a.borrow().value;
                vec![(a.clone(), ga)]
            }
            OpKind::Abs(a) => {
                // d|x|/dx = sign(x), and 0 at x = 0
                let sign = a.borrow().value.map(|x| x.sign());
                let ga = out_grad & &sign;
                vec![(a.clone(), ga)]
            }
            OpKind::Gather(a, indices) => {
                // every picked row hands its gradient back to its own place in
                // the table, and repeats are summed
                let rows = a.borrow().value.get_shape()[0];
                let ga = out_grad.scatter_add_rows(indices, rows);
                vec![(a.clone(), ga)]
            }
            OpKind::Tanh(a) => {
                // d/dx tanh(x) = 1 - tanh(x)^2, and tanh(x) is the value we
                // already computed, so nothing has to be recomputed
                let ga = out_grad & &out_value.map(|y| T::one() - y * y);
                vec![(a.clone(), ga)]
            }
            OpKind::Sigmoid(a) => {
                // d/dx s(x) = s(x) (1 - s(x)), likewise straight from the value
                let ga = out_grad & &out_value.map(|y| y * (T::one() - y));
                vec![(a.clone(), ga)]
            }
            OpKind::MaxAxis(a, axis) => {
                // only the winning element of each run was passed on, so only it
                // receives a gradient; the rest of the run contributed nothing
                let value = a.borrow().value.shallow_copy();
                let winners = value.argmax_axis(*axis);

                let shape = value.get_shape();
                let reduce_dim = shape[*axis];
                let inner: usize = shape[*axis + 1..].iter().product();

                let grad = out_grad.packed_data();
                let mut data = vec![T::default(); crate::linalg::product(&shape)];

                for (out_idx, &k) in winners.iter().enumerate() {
                    // out_idx walks the reduced tensor; splitting it at the axis
                    // gives the position of the run in the full one
                    let outer = out_idx / inner;
                    let within = out_idx % inner;

                    data[(outer * reduce_dim + k) * inner + within] = grad[out_idx];
                }

                vec![(a.clone(), Tensor::new(data, shape))]
            }
            OpKind::Reshape(a) => {
                // the elements are the same ones in the same order, only the
                // shape changed, so the gradient just changes shape back
                let a_shape = a.borrow().value.get_shape();
                let ga = Tensor::new(out_grad.packed_data(), a_shape);
                vec![(a.clone(), ga)]
            }
            OpKind::Permute(a, axes) => {
                // undo the permutation: inverse[axes[i]] = i
                let mut inverse = vec![0usize; axes.len()];
                for (i, &axis) in axes.iter().enumerate() {
                    inverse[axis] = i;
                }

                let permuted = out_grad
                    .permute(&inverse)
                    .expect("permute backward got a bad axis list (bug)");
                let ga = Tensor::new(permuted.packed_data(), permuted.get_shape());

                vec![(a.clone(), ga)]
            }
            OpKind::Unfold(a, kernel, stride, padding) => {
                // a pixel read by several windows collects a gradient from each,
                // which is exactly what fold_2d sums up
                let shape = a.borrow().value.get_shape();
                let ga = out_grad.fold_2d(
                    [shape[0], shape[1], shape[2], shape[3]],
                    *kernel,
                    *stride,
                    *padding,
                );
                vec![(a.clone(), ga)]
            }
            OpKind::Select(a, axis, index) => {
                // the gradient belongs to the slice that was taken; every other
                // position of the input contributed nothing
                let a_shape = a.borrow().value.get_shape();
                let ga = out_grad.scatter_select(&a_shape, *axis, *index);
                vec![(a.clone(), ga)]
            }
            OpKind::Stack(parts, axis) => {
                // each part gets back exactly the slice of the gradient it filled
                parts
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (p.clone(), out_grad.select(*axis, i)))
                    .collect()
            }
            OpKind::Clamp(a, min, max) => {
                // the gradient passes through untouched inside the range and is
                // cut off outside of it, where the output no longer follows x
                let (min, max) = (*min, *max);
                let mask = a.borrow().value.map(|x| {
                    if x < min || x > max { T::default() } else { T::one() }
                });
                let ga = out_grad & &mask;
                vec![(a.clone(), ga)]
            }

        }
    }   
}
