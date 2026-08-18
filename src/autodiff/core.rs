use std::{cell::RefCell, collections::HashSet, fmt::Display, rc::Rc};
use crate::{Float, linalg::Tensor};

pub struct Var<T: Float> {
    pub value: Tensor<T>,
    pub grad: RefCell<Tensor<T>>,
    pub(crate) op: OpKind<T>,
    pub requires_grad: bool,
    pub(crate) cached_topo: RefCell<Option<Vec<VarRef<T>>>>,
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
    /// Удобные аксессоры
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
        
        // 1. Короткое имя операции
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

        // 2. Выводим форму и тип операции
        write!(f, "Var({} | shape: {:?} | req_grad: {})", 
            op_name, 
            var.value.shape, 
            var.requires_grad
        )?;

        // 3. Если тензор маленький (например, скаляр или 1D до 5 элементов), 
        // можно сразу напечатать значение через твой Tensor Display
        if var.value.get_data().len() <= 25 {
            write!(f, " => {}", var.value)?;
        }

        Ok(())
    }
}


/// Построить топологический порядок (list of nodes) начиная с корня.
/// Возвращаем Vec<VarRef<T>> в порядке обхода (parents сначала, потом node).
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
                
                // transpose (permute последних двух осей) как view
                let mut axes_a_t: Vec<usize> = (0..ndim_a).collect();
                axes_a_t.swap(ndim_a - 1, ndim_a - 2);
                
                let mut axes_b_t: Vec<usize> = (0..ndim_b).collect();
                axes_b_t.swap(ndim_b - 1, ndim_b - 2);

                let a_t = a_val.permute(&axes_a_t).unwrap(); // A^T view
                let b_t = b_val.permute(&axes_b_t).unwrap(); // B^T view

                // Правильно:
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
