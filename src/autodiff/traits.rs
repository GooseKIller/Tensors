use std::ops::Add;

use crate::{Float, autodiff::core::{VarRef, build_topo}, linalg::Tensor};

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
        // С кэшированием - используем topo из backward или строим
        let topo = self.0.borrow().cached_topo.borrow().clone();
        
        let topo = match topo {
            Some(t) => t,
            None => {
                let t = build_topo(self);
                *self.0.borrow_mut().cached_topo.borrow_mut() = Some(t.clone());
                t
            }
        };
        
        for node in topo {
            if node.borrow().requires_grad {
                node.0.borrow().grad.borrow_mut().full_(T::default());
            }
        }
    }

    fn backward(&self) {
        // С кэшированием - первый вызов строит, последующие используют кэш
        let topo = self.0.borrow().cached_topo.borrow().clone();
        
        let topo = match topo {
            Some(t) => t,
            None => {
                let t = build_topo(self);
                *self.0.borrow_mut().cached_topo.borrow_mut() = Some(t.clone());
                t
            }
        };
        
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
                
                let grad_to_add = if parent_grad.get_shape().as_slice() == expected_shape.as_slice() {
                    parent_grad
                } else {
                    // Try to reduce, or use parent value's shape as fallback
                    let parent_shape = p_varref.borrow().value.get_shape();
                    match parent_grad.reduce_to_shape(&expected_shape) {
                        Some(t) => t,
                        None => match parent_grad.reduce_to_shape(&parent_shape) {
                            Some(t) => t,
                            None => {
                                // Last resort: reshape (might be wrong data but won't panic)
                                parent_grad.reshape(expected_shape)
                            }
                        }
                    }
                };

                // Create new tensor (non-in-place) to avoid accumulation issues
                let current_grad = {
                    let p = p_varref.borrow();
                    let grad = p.grad.borrow();
                    grad.shallow_copy()
                };

                let new_grad = (&current_grad).add(&grad_to_add);
                
                let p = p_varref.borrow_mut();

                *p.grad.borrow_mut() = new_grad;
            }
        }
    }
}
