use crate::{Float, linalg::Tensor};
use std::{cell::RefCell, collections::HashSet, ops::{Add, BitAnd, BitXor, Div, Mul, Neg, Sub}, rc::Rc};

pub struct Var<T: Float> {
    pub value: Tensor<T>,
    pub grad: RefCell<Tensor<T>>,

    /// Теперь храним strong Rc-ссылки на родителей, чтобы временные узлы не удалялись до вызова backward.
    op: OpKind<T>,
    pub requires_grad: bool,
}

/// NEWTYPE: VarRef — локальный тип-обёртка вокруг Rc<RefCell<Var<T>>>.
/// Наличие локального типа снимает проблему orphan rules при реализации Add/Mul.
#[derive(Clone)]
pub struct VarRef<T: Float>(pub Rc<RefCell<Var<T>>>);

impl<T: Float> VarRef<T> {
    pub fn sum(&self) -> Self{
        sum_op(self)
    }
}

impl<T: Float> Var<T> {
    pub fn leaf(value: Tensor<T>, requires_grad: bool) -> VarRef<T> {
        let shape = value.get_shape();
        let zero = Tensor::from_num(T::default(), shape);
        VarRef(Rc::new(RefCell::new(Var {
            value,
            grad: RefCell::new(zero),
            op: OpKind::Leaf,
            requires_grad,
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
enum OpKind<T: Float> {
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
        }
    }

    #[allow(dead_code)]
    fn backward(&self, out_grad: &Tensor<T>, out_value: &Tensor<T>) 
        -> Vec<(VarRef<T>, Tensor<T>)> {
        match self {
            OpKind::Leaf => vec![],
            OpKind::Add(a, b) => {
                vec![
                    (a.clone(), out_grad.shallow_copy()),
                    (b.clone(), out_grad.shallow_copy()),
                ]
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
                let ga = out_grad & (&b_val);
                let gb = out_grad & (&a_val);
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
                let gb = -(&(out_grad & &a_val) / &b2);
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
            OpKind::SumAxis(a, axis, keepdim) => {
                let a_shape = a.borrow().value.shape.clone();
                let mut grad_tensor = out_grad.clone();

                // Если суммировали без keepdim, нужно сначала вернуть единичную размерность
                // чтобы broadcast_to понимал, какую ось расширять.
                if !*keepdim {
                    let mut expanded_shape = out_grad.shape.clone();
                    expanded_shape.insert(*axis, 1);
                    grad_tensor = grad_tensor.reshape(expanded_shape);
                }

                // Растягиваем градиент до исходной формы тензора 'a'
                let ga = grad_tensor.broadcast_to(&a_shape).unwrap();
                vec![(a.clone(), ga)]
            }
            OpKind::Log(a) => {
                // d(ln(x))/dx = 1/x
                let ga = out_grad / &a.borrow().value;
                vec![(a.clone(), ga)]
            }

        }
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
    })))
}


/// Интерфейс автограда
#[allow(dead_code)]
pub trait AutoGrad {
    type Elem: Float;
    fn value(&self) -> Tensor<Self::Elem>;
    fn grad(&self) -> Tensor<Self::Elem>;
    fn zero_grad(&self);
    fn backward(&self);
}

/// impl для VarRef (struct)
impl<T: Float> AutoGrad for VarRef<T> {
    type Elem = T;

    fn value(&self) -> Tensor<T> {
        self.borrow().value.clone()
    }

    fn grad(&self) -> Tensor<T> {
        self.borrow().grad.borrow().clone()
    }

    fn zero_grad(&self) {
        let topo = build_topo(self);
        for node in topo {
            if node.borrow().requires_grad {
                node.0.borrow().grad.borrow_mut().full_(T::default());
            }
        }
    }

    fn backward(&self) {
        let topo = build_topo(self);
        
        {
            let root = self.0.borrow_mut();
            let ones = if root.value.get_shape().is_empty(){
                Tensor::scalar(T::one())
            } else {
                Tensor::from_num(T::one(), root.value.get_shape())
            };

            *root.grad.borrow_mut() = ones;
        }
        
        for node in topo.iter().rev() {
            // Одновременно получаем все необходимые данные
            let (out_grad, out_value, op) = {
                let node_ref = node.0.borrow();
                let grad = node_ref.grad.borrow().clone();
                let val = node_ref.value.shallow_copy();
                let op = node_ref.op.clone();
                (grad, val, op)
            };

            let parents_grad = op.backward(&out_grad, &out_value);
            for (p_varref, parent_grad) in parents_grad {
                if !p_varref.borrow().requires_grad {
                    continue;
                }

                let expected_shape = p_varref.borrow().grad.borrow().get_shape();
                
                /*let parent_grad_reduced = parent_grad
                    .reduce_to_shape(&excepted_shape)
                    .expect("Grad Shape Missmatch");*/
                let grad_to_add = if parent_grad.get_shape() == expected_shape {
                    parent_grad
                } else {
                    parent_grad
                        .reduce_to_shape(&expected_shape)
                        .expect("Grad Shape Missmatch")
                };

                let current_grad = {
                    let p = p_varref.borrow();
                    let grad = p.grad.borrow();
                    grad.clone()
                };

                //let new_grad = current_grad.add(&parent_grad_reduced);
                let new_grad = current_grad.add(&grad_to_add);
                
                let p = p_varref.borrow_mut();

                *p.grad.borrow_mut() = new_grad;
            }
        }
    }
}

impl<T: Float> Add for &VarRef<T> {
    type Output = VarRef<T>;
    fn add(self, rhs: Self) -> Self::Output {
        add_op(self, rhs)
    }
}

impl<T:Float> Sub for &VarRef<T> {
    type Output = VarRef<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        sub_op(self, rhs)
    }
    
}

impl<T: Float> BitAnd for &VarRef<T> {
    type Output = VarRef<T>;
    fn bitand(self, rhs: Self) -> Self::Output {
        mul_op(self, rhs)
    }
}

impl<T: Float> BitXor for &VarRef<T> {
    type Output = VarRef<T>;
    fn bitxor(self, rhs: Self) -> Self::Output {
        powf_op(self, rhs)
    }
}

impl<T: Float> Div for &VarRef<T> {
    type Output = VarRef<T>;
    fn div(self, rhs: Self) -> Self::Output {
        div_op(self, rhs)
    }
}

impl<T: Float> Mul for &VarRef<T> {
    type Output = VarRef<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        matmul_op(self, rhs)
    }
}

impl<T: Float> Neg for &VarRef<T> {
    type Output = VarRef<T>;
    fn neg(self) -> Self::Output {
        sub_op(
            &Var::leaf(Tensor::scalar(T::default()), false),
            self,
        )
    }
}

/*
impl<T: Float> Drop for Var<T> {
    fn drop(&mut self) {
        self.parents.clear();
    }
}*/

// Scalars of Tensors (to simplicity)
// Scalar
impl<T: Float> Add<T> for &VarRef<T> {
    type Output = VarRef<T>;
    fn add(self, rhs: T) -> Self::Output {
        add_op(
            self,
            &Var::leaf(Tensor::scalar(rhs), false)
        )
    }
}

impl<T:Float> Sub<T> for &VarRef<T> {
    type Output = VarRef<T>;
    fn sub(self, rhs: T) -> Self::Output {
        sub_op(
            self,
            &Var::leaf(Tensor::scalar(rhs), false)
        )
    }
}

impl<T:Float> Mul<T> for &VarRef<T> {
    type Output = VarRef<T>;
    fn mul(self, rhs: T) -> Self::Output {
        mul_op(
            self,
            &Var::leaf(Tensor::scalar(rhs), false)
        )
    }
}

impl<T:Float> BitXor<T> for &VarRef<T> {
    type Output = VarRef<T>;
    fn bitxor(self, rhs: T) -> Self::Output {
        powf_op(
            self,
            &Var::leaf(Tensor::scalar(rhs), false)
        )
    }
}

impl<T:Float> Div<T> for  &VarRef<T> {
    type Output = VarRef<T>;
    fn div(self, rhs: T) -> Self::Output {
        div_op(
            self,
            &Var::leaf(Tensor::scalar(rhs), false)
        )
    }
}

// Tensor
impl<T:Float> BitAnd<&Tensor<T>> for &VarRef<T> {
    type Output = VarRef<T>;
    fn bitand(self, rhs: &Tensor<T>) -> Self::Output {
        mul_op(
            self,
            &Var::leaf(rhs.shallow_copy(), false)
        )
    }
}

impl<T:Float> Sub<&VarRef<T>> for &Tensor<T> {
    type Output = VarRef<T>;
    fn sub(self, rhs: &VarRef<T>) -> Self::Output {
        sub_op(
            &Var::leaf(self.shallow_copy(), false),
            rhs
        )
    }
}

impl<T: Float> Mul<&VarRef<T>> for &Tensor<T>{
    type Output = VarRef<T>;
    fn mul(self, rhs: &VarRef<T>) -> Self::Output {
        matmul_op(
            &Var::leaf(self.shallow_copy(), false),
            rhs
        )
    }
}

impl<T: Float> Mul<&Tensor<T>> for &VarRef<T> {
    type Output = VarRef<T>;
    fn mul(self, rhs: &Tensor<T>) -> Self::Output {
        matmul_op(
            self,
            &Var::leaf(rhs.shallow_copy(), false)
        )
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::E;

    use crate::linalg::{Tensor, Vector};
    use crate::tensor;
    use crate::utils::autodiff::{AutoGrad, Var, VarRef, div_op, powf_op, sum_op};
    
    #[test]
    fn autograd() {
        let a_val = tensor![1.0, 2.0, 3.0];
        let b_val = tensor![4.0, 5.0, 6.0];

        let a= Var::leaf(a_val, true);
        let b = Var::leaf(b_val, true);

        // оба варианта теперь работают
        let ab = &a & &b;
        let c = &ab + &a;

        c.backward();

        println!("c.value = {}", c.value());
        println!("dc/da = {}", a.grad()); // ожидаем: b + 1
        println!("dc/db = {}", b.grad());

        c.zero_grad();
        // временный-в-выражении тоже должен работать
        let a2: VarRef<f32> = Var::leaf(tensor![1.0, 2.0, 3.0], true);
        let b2: VarRef<f32> = Var::leaf(tensor![4.0, 5.0, 6.0], true);
        let c: VarRef<f32> = &(&a2 & &b2) + &a2;
        c.backward();

        println!("c2.value = {}", c.value());
        println!("dc2/da2 = {}", a2.grad()); // ожидаем: b2 + 1
        println!("dc2/db2 = {}", b2.grad());
    }


    #[test]
    fn another_grad() {
        let a_t = tensor![1.0, 2.0, 3.0];
        let b_t = Tensor::from_num(6.0, vec![3]);

        let a = Var::leaf(a_t, true);
        let b = Var::leaf(b_t, true);

        let c = &(&a & &a) + &b;
        
        c.backward();

        println!("c.value = {}", c.value());
        println!("dc/da = {}", a.grad());
        println!("dc/db = {}", b.grad());
    }

    #[test]
    fn all_autograd() {
        let a_t = tensor![1.0, 2.0, 3.0];
        let b_t = Tensor::from_num(6.0, vec![3]);

        let a = Var::leaf(a_t, true);
        let b = Var::leaf(b_t, true);

        let c = &(&(&a & &a) - &b) + &b;
        
        c.backward();

        println!("c.value = {}", c.value());
        println!("dc/da = {}", a.grad());
        println!("dc/db = {}", b.grad());
    }

    #[test]
    fn strange_autograd() {
        let a_t = tensor![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
        let b_t = tensor![1.0, 1.0, 1.0];

        let a = Var::leaf(a_t, true);
        let b = Var::leaf(b_t, true);

        let c = sum_op(&(&(&a & &b) - &a));
        
        c.backward();

        println!("c.value = {}", c.value());
        println!("dc/da = {}", a.grad());
        println!("dc/db = {}", b.grad());
    }

    #[test]
    fn is_this_a_softmax() {
        let x: VarRef<f32> = Var::leaf(tensor![1.0, 2.0, 3.0], true);
        let e = Var::leaf(tensor![E], false);
        let exp = &e ^ &x;
        let ans = &exp / &sum_op(&exp);

        let _a = &e - 1.0;

        let what = ans.sum();
        what.backward();

        println!("{}", what.value());

        println!("{}", ans.value());
        println!("{}", x.grad());
    }

    #[test]
    fn gradient_descent_visualizer() {
        // Функция: f(x,y) = sin(x)*cos(y) + 0.1*(x²+y²)
        let x = Var::leaf(tensor![2.0], true);
        let y: VarRef<f32> = Var::leaf(tensor![2.0], true);
        
        let mut positions = vec![];
        
        for step in 0..50 {
            x.zero_grad();
            y.zero_grad();
            // Вычисляем функцию
            let term2 = &(&(&x & &x) + &(&y & &y)) & &Var::leaf(Tensor::scalar(0.1), false);
            
            // Обратное распространение для градиента
            term2.backward();
            
            let grad_x = x.grad().item();
            let grad_y = y.grad().item();
            
            // Обновляем позицию (градиентный спуск)
            {
                let mut x_val = x.borrow_mut();
                let grad = x_val.grad.borrow().clone();
                x_val.value = &x_val.value - &(grad * 0.1);
            }
            {
                let mut y_val = y.borrow_mut();
                let grad = y_val.grad.borrow().clone();
                y_val.value = &y_val.value - &(grad * 0.1);
            }
            
            positions.push((x.value().item(), y.value().item(), term2.value().item()));
            
            println!("Шаг {}: x={:.3}, y={:.3}, f={:.3}, grad=({:.3}, {:.3})", 
                    step, x.value().item(), y.value().item(), term2.value().item(), grad_x, grad_y);
        }
        
        // Можно визуализировать в ASCII
        println!("\nТраектория градиентного спуска:");
        for (x, y, f) in positions {
            let plot_x = ((x + 3.0) * 10.0) as usize;
            let plot_y = ((y + 3.0) * 5.0) as usize;
            // ASCII визуализация...
            println!("{plot_x} {plot_y} {f}");
        }
    }

    #[test]
    fn linear_reg() {
        let x_val = Tensor::from(Vector::linspace(0.0, 1.0, 15, true))
            .reshape(vec![15, 1]);
        let y_val = &x_val * 2.0 + &Tensor::rand(vec![15, 1]) + 6.0;

        let x = Var::leaf(x_val, false);
        let y = Var::leaf(y_val, false);

        let w = Var::leaf(tensor![-0.123], true);
        let b = Var::leaf(tensor![0.5], true);
        for _ in 0..5 {
            let y_pred = &(&w & &x) + &b;
            let mut loss = &(&y - &y_pred) ^ &Var::leaf(Tensor::scalar(2.0), false);
            loss = &loss.sum() / &Var::leaf(tensor![15.0], false);
            println!("{}", loss.borrow().value);
            loss.backward();

            {
                let mut w = w.borrow_mut();
                let grad = w.grad.borrow().clone();
                w.value = &w.value - &(&grad * 0.0001);
                println!("W: {}", w.value);
            }

            {
                let mut b = b.borrow_mut();
                let grad = b.grad.borrow().clone();
                b.value = &b.value - &(&grad * 0.0001);
                println!("B: {}", b.value);
            }
            loss.zero_grad();
        }
        let y_pred = &(&w & &x) + &b;
        println!("{:?}", y_pred.borrow().value.shape);
        println!("Pred: {}", y_pred.borrow().value.transpose());
        println!("{:?}", y.borrow().value.shape);
        println!("Target: {}", y.borrow().value.transpose());
    }

    #[test]
    fn linear_reg2() {
    // Создаем x как вектор-столбец [15, 1]
        let x_val = Tensor::from(Vector::linspace(0.0, 1.0, 15, true)).reshape(vec![15, 1]);
        
        // Создаем y как вектор-столбец [15, 1]
        // y = 2x + 6 + шум
        let noise = Tensor::rand(vec![15, 1]) * 0.5; // Шум той же формы
        let y_val = &x_val * 2.0 + 6.0 + &noise;
        
        let x = x_val;//Var::leaf(x_val, false);
        let y = y_val;//Var::leaf(y_val, false);

        // Инициализируем параметры как скаляры
        let w = Var::leaf(tensor![[-0.123]], true); // Сделаем тензором [[-0.123]]
        let b = Var::leaf(tensor![[0.5]], true);    // Сделаем тензором [[0.5]]
        
        let learning_rate = 0.01;
        let n_samples = 15.0;
        
        for epoch in 0..120 { // Увеличим количество эпох
            // Прямой проход: y_pred = w * x + b
            let y_pred = &(&w & &x) + &b;
            
            // Вычисляем MSE loss = mean((y - y_pred)^2)
            let diff = &y - &y_pred;
            let loss = &(&diff ^ 2.0).sum() / (n_samples as f32);
            
            if epoch % 10 == 0 {
                println!("Epoch {}: Loss = {}", epoch, loss.borrow().value);
            }
            
            // Обратный проход
            loss.backward();
            
            // Обновляем параметры с градиентным спуском
            {
                let mut w_mut = w.borrow_mut();
                let grad = w_mut.grad.borrow().clone();
                w_mut.value = &w_mut.value - &(&grad * learning_rate);
            }
            
            {
                let mut b_mut = b.borrow_mut();
                let grad = b_mut.grad.borrow().clone();
                b_mut.value = &b_mut.value - &(&grad * learning_rate);
            }
            
            // Обнуляем градиенты
            loss.zero_grad();
        }
        
        // Финальные предсказания
        let y_pred = &(&w & &x) + &b;
        println!("\n=== Results ===");
        println!("Learned parameters:");
        println!("w = {}", w.borrow().value);
        println!("b = {}", b.borrow().value);
        
        println!("\nPredictions (first 5):");
        for i in 0..5 {
            println!("x={:.2}, y_pred={:.4}, y_true={:.4}", 
                x.get(&[i, 0]).unwrap(),
                y_pred.borrow().value.get(&[i, 0]).unwrap(),
                y.get(&[i, 0]).unwrap());
        }
    }

    #[test]
    fn mpl() {
        let x = tensor![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let y = tensor![[0.0], [1.0], [1.0], [0.0]];
        let ly1_val: Tensor<f32> = Tensor::randn(vec![2, 2]) * 0.5;
        let b1_val: Tensor<f32> = Tensor::randn(vec![1, 2]) * 0.5;

        let ly2_val: Tensor<f32> = Tensor::randn(vec![2, 1]);
        let b2_val: Tensor<f32> = Tensor::randn(vec![1, 1]);

        let inter = &x * &ly1_val + &b1_val;
        let out = &inter * &ly2_val + &b2_val;
        println!("{out}");

        let sigmoid = |x: &VarRef<f32>| {
            let e_minus = powf_op(&Var::leaf(Tensor::scalar(E), false),
             &-x);
            let denom = &e_minus + 1.0;
            div_op(&Var::leaf(Tensor::scalar(1.0), false),
             &denom)
        };

        let ly1 = Var::leaf(ly1_val, true);
        let b1 = Var::leaf(b1_val, true);

        let ly2 = Var::leaf(ly2_val, true);
        let b2 = Var::leaf(b2_val, true);

        let lr = 1.0;
        for i in 0..1000 {
            let inter = &(&x * &ly1) + &b1;
            let fnc = sigmoid(&inter);
            
            // 3. Add sigmoid to the output layer
            let out_linear = &(&fnc * &ly2) + &b2;
            let out = sigmoid(&out_linear); 

            let diff = &y - &out;
            let loss = &(&diff ^ 2.0).sum() / 4.0;

            loss.backward();

            {
                let mut w_mut = ly1.borrow_mut();
                let grad = w_mut.grad.borrow().shallow_copy();

                w_mut.value = &w_mut.value - &(&grad * lr);
            }
            {
                let mut b_mut = b1.borrow_mut();
                let grad = b_mut.grad.borrow().shallow_copy();

                b_mut.value = &b_mut.value - &(&grad * lr);
            }

            {
                let mut w_mut = ly2.borrow_mut();
                let grad = w_mut.grad.borrow().shallow_copy();

                w_mut.value = &w_mut.value - &(&grad * lr);
            }
            {
                let mut b_mut = b2.borrow_mut();
                let grad = b_mut.grad.borrow().shallow_copy();

                b_mut.value = &b_mut.value - &(&grad * lr);
            }
            loss.zero_grad();

            if i % 100 == 0{
                println!("{i}: {}", out.value());
                println!("{i}: {}", loss.value());
            }

        }

    }

}
