use crate::Float;
use crate::linalg::Tensor;
use crate::autodiff::VarRef;

/// Clip gradients by global norm.
/// 
/// Computes the L2 norm of all gradients and scales them down if the norm exceeds `max_norm`.
/// This prevents exploding gradients in deep networks.
pub fn clip_grad<T: Float>(params: &[VarRef<T>], max_norm: T) -> T {
    let mut total_norm_sq = T::default();
    
    for param in params {
        let var = param.borrow();
        if !var.requires_grad {
            continue;
        }
        
        let grad = var.grad.borrow();
        
        let data = grad.packed_data();
        for &val in &data {
            total_norm_sq = total_norm_sq + val * val;
        }
    }
    
    let total_norm = total_norm_sq.sqrt();
    
    if total_norm > max_norm {
        let scale = max_norm / total_norm;
        
        for param in params {
            let var = param.borrow();
            if !var.requires_grad {
                continue;
            }
            
            let mut grad = var.grad.borrow_mut();
            let data = grad.packed_data();
            let scaled: Vec<T> = data.iter().map(|&x| x * scale).collect();
            
            let shape = grad.get_shape();
            *grad = Tensor::new(scaled, shape);
        }
        
        total_norm
    } else {
        total_norm
    }
}

/// Clip gradients by value (each element clipped to [-clip_value, clip_value])
/// 
/// Simpler than norm clipping - just bounds each gradient element.
pub fn clip_grad_value<T: Float>(params: &[VarRef<T>], clip_value: T) {
    for param in params {
        let var = param.borrow();
        if !var.requires_grad {
            continue;
        }
        
        let mut grad = var.grad.borrow_mut();
        let data = grad.packed_data();
        
        let clipped: Vec<T> = data.iter().map(|&x| {
            if x > clip_value {
                clip_value
            } else if x < -clip_value {
                -clip_value
            } else {
                x
            }
        }).collect();
        
        let shape = grad.get_shape();
        *grad = Tensor::new(clipped, shape);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor;
    use crate::autodiff::Var;
    
    #[test]
    fn test_clip_grad_norm() {
        let p1 = Var::leaf(tensor![[1.0, 2.0], [3.0, 4.0]], true);
        let p2 = Var::leaf(tensor![[10.0, 20.0]], true);
        
        *p1.borrow().grad.borrow_mut() = tensor![[100.0, 100.0], [100.0, 100.0]];
        *p2.borrow().grad.borrow_mut() = tensor![[200.0, 200.0]];
        
        let params: &[VarRef<f32>] = &[p1.clone(), p2.clone()];
        let max_norm = 10.0;
        
        let norm = clip_grad(params, max_norm);
        
        assert!(norm > max_norm);
        
        let var1 = p1.borrow();
        let p1_grad = var1.grad.borrow();
        let var2 = p2.borrow();
        let p2_grad = var2.grad.borrow();
        
        let p1_norm_sq: f32 = p1_grad.packed_data().iter().map(|&x| x*x).sum::<f32>().sqrt();
        let p2_norm_sq: f32 = p2_grad.packed_data().iter().map(|&x| x*x).sum::<f32>().sqrt();
        
        assert!(p1_norm_sq <= max_norm + 0.001);
        assert!(p2_norm_sq <= max_norm + 0.001);
        
        println!("Original norm: {}, Max norm: {}, After clip: p1={}, p2={}", 
                norm, max_norm, p1_norm_sq, p2_norm_sq);
    }
    
    #[test]
    fn test_clip_grad_value() {
        let p = Var::leaf(tensor![1.0, 2.0, 3.0, 4.0], true);
        
        *p.borrow().grad.borrow_mut() = tensor![10.0, -20.0, 5.0, -3.0];
        
        let params: &[VarRef<f32>] = &[p.clone()];
        clip_grad_value(params, 7.0);
        
        let var = p.borrow();
        let grad = var.grad.borrow();
        let data = grad.packed_data();
        
        assert_eq!(data[0], 7.0);
        assert_eq!(data[1], -7.0);
        assert_eq!(data[2], 5.0);
        assert_eq!(data[3], -3.0);
    }
}