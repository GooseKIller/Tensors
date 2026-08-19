use std::ops::Add;

use crate::{Float, autodiff::core::{VarRef, build_topo}, linalg::Tensor};

/// The autograd interface
#[allow(dead_code)]
pub trait AutoGrad {
    type Elem: Float;
    fn value(&self) -> Tensor<Self::Elem>;
    fn grad(&self) -> Tensor<Self::Elem>;
    fn zero_grad(&self);
    fn backward(&self);
}

/// The implementation for VarRef
impl<T: Float> AutoGrad for VarRef<T> {
    type Elem = T;

    fn value(&self) -> Tensor<T> {
        // shallow_copy rather than clone: clone on a tensor copies all of the
        // data, and this method is called in every forward pass just for the
        // shape. The copy shares the buffer, but the value of a node can still
        // only be changed by assignment, and full_ clones the buffer on write -
        // so the observable behaviour is the same.
        self.borrow().value.shallow_copy()
    }

    fn grad(&self) -> Tensor<T> {
        self.borrow().grad.borrow().clone()
    }

    fn zero_grad(&self) {
        // Cached: reuse the topology from backward, or build it
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
        // Cached: the first call builds it, later ones reuse it
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
            // Everything needed is taken in one go
            let (out_grad, out_value, op) = {
                let node_ref = node.0.borrow();

                // If a node requires no gradient then neither does any of its
                // ancestors - requires_grad propagates forward from the leaves.
                // So its whole branch of the backward pass can be skipped.
                if !node_ref.requires_grad {
                    continue;
                }

                // shallow_copy rather than clone: the gradient is only read
                // here, while clone on a tensor is a full deep copy - that is,
                // an allocation for every node of the graph.
                let grad = node_ref.grad.borrow().shallow_copy();
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

#[cfg(test)]
mod tests {
    use crate::{
        autodiff::{AutoGrad, Var},
        tensor,
    };

    #[test]
    fn a_branch_without_grad_is_skipped_but_the_other_one_still_learns() {
        // backward() steps over any node that does not require a gradient; the
        // branch that does must be unaffected by that
        let tracked = Var::leaf(tensor![[1.0f64, 2.0]], true);
        let frozen = Var::leaf(tensor![[3.0f64, 4.0]], false);

        let left = &(&tracked + &frozen) & &frozen;
        let right = &frozen & &frozen; // entirely outside the gradient path

        let out = &left + &right;
        out.sum().backward();

        // d/d tracked of (tracked + frozen) * frozen is frozen itself
        assert_eq!(tracked.grad().get_data(), vec![3.0, 4.0]);
        // and the frozen leaf collected nothing
        assert_eq!(frozen.grad().get_data(), vec![0.0, 0.0]);
    }

    #[test]
    fn nothing_requiring_grad_means_nothing_happens() {
        let a = Var::leaf(tensor![[1.0f64, 2.0]], false);
        let b = Var::leaf(tensor![[3.0f64, 4.0]], false);

        let out = &a + &b;
        out.sum().backward(); // must not panic

        assert_eq!(a.grad().get_data(), vec![0.0, 0.0]);
    }
}
