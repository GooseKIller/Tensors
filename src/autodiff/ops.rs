use std::{cell::RefCell, rc::Rc};

use crate::{Float, autodiff::{OpKind, core::{Var, VarRef}}, linalg::Tensor};

impl<T: Float> VarRef<T> {
    pub fn sum(&self) -> Self{
        sum_op(self)
    }
}

/// Операция сложения: создаёт новый VarRef и сохраняет strong ссылки на родителей.
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

/// Операция поэлементного умножения: создаёт новый VarRef и сохраняет strong ссылки на родителей.
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
    let out_value = va.map(|x| x.ln()); // Предполагаем метод get_log в Float
    
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
