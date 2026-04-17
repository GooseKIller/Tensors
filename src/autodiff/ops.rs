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
