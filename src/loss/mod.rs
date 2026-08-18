//! # Loss functions
//!
//! Functions for measuring how far a prediction is from the expected value.
//! Each of them takes two nodes of the autodiff graph and returns a scalar node,
//! so `backward()` can be called on the result.
//!
//! # Example
//! ```
//! use tensorrs::{tensor, loss::mse, autodiff::{AutoGrad, Var}};
//!
//! let y_pred = Var::leaf(tensor![[2.0f32], [4.0]], false);
//! let y      = Var::leaf(tensor![[1.0f32], [2.0]], false);
//!
//! let loss = mse(&y_pred, &y);
//! assert_eq!(loss.value().item(), 2.5); // (1 + 4) / 2
//! ```
//!
//! # Functions
//!
//! 1.[mse] - Mean squared error
//!
//! 2.[sse] - Sum of squared errors
//!
//! 3.[mape] - Mean absolute percentage error
//!
//! 4.[cross_entropy] - Cross entropy, for multi-class classification
//!
//! 5.[binary_cross_entropy] - Binary cross entropy, for two-class classification

mod crossentropy;
mod mape;
mod mse;

mod binarycrossentropy;
mod sse;

pub use binarycrossentropy::*;
pub use crossentropy::*;
pub use mape::*;
pub use mse::*;
pub use sse::*;

#[cfg(test)]
mod tests {
    use crate::{tensor, loss::*, autodiff::{AutoGrad, Var}};

    fn pair(pred: Vec<f64>, target: Vec<f64>) -> (crate::autodiff::VarRef<f64>, crate::autodiff::VarRef<f64>) {
        let n = pred.len();
        (
            Var::leaf(crate::linalg::Tensor::new(pred, vec![n, 1]), true),
            Var::leaf(crate::linalg::Tensor::new(target, vec![n, 1]), false),
        )
    }

    #[test]
    fn mse_matches_formula() {
        let (p, y) = pair(vec![2.0, 4.0], vec![1.0, 2.0]);
        assert_eq!(mse(&p, &y).value().item(), 2.5); // (1 + 4) / 2
    }

    #[test]
    fn sse_sums_squares_not_absolutes() {
        let (p, y) = pair(vec![2.0, 4.0], vec![1.0, 2.0]);
        // the sum of absolutes would be 3 here
        assert_eq!(sse(&p, &y).value().item(), 5.0); // 1 + 4
    }

    #[test]
    fn mape_is_relative_and_survives_a_zero_target() {
        let (p, y) = pair(vec![2.0, 4.0], vec![1.0, 2.0]);
        assert_eq!(mape(&p, &y).value().item(), 1.0); // (|1/1| + |2/2|) / 2

        let (p, y) = pair(vec![3.0, 4.0], vec![4.0, 2.0]);
        // (|-1/4| + |2/2|) / 2 = (0.25 + 1) / 2
        assert!((mape(&p, &y).value().item() - 0.625).abs() < 1e-12);

        // a zero expected value must stay finite instead of turning into infinity
        let (p, y) = pair(vec![1.0, 4.0], vec![0.0, 2.0]);
        assert!(mape(&p, &y).value().item().is_finite());
    }

    #[test]
    fn cross_entropy_matches_formula() {
        let pred   = Var::leaf(tensor![[0.7f64, 0.3]], true);
        let target = Var::leaf(tensor![[1.0f64, 0.0]], false);

        let loss = cross_entropy(&pred, &target);
        assert!((loss.value().item() - 0.35667494393873245).abs() < 1e-9); // -ln(0.7)

        loss.backward();
        // dCE/dp = -target / p
        let g = pred.grad().get_data();
        assert!((g[0] + 1.0 / 0.7).abs() < 1e-9);
        assert_eq!(g[1], 0.0);
    }

    #[test]
    fn binary_cross_entropy_matches_formula() {
        let pred   = Var::leaf(tensor![[0.9f64], [0.1]], true);
        let target = Var::leaf(tensor![[1.0f64], [0.0]], false);

        let loss = binary_cross_entropy(&pred, &target);
        assert!((loss.value().item() - 0.10536051565782628).abs() < 1e-9); // -ln(0.9)

        loss.backward();
        // dBCE/dp = (1/n) * (p - y) / (p (1 - p))
        let g = pred.grad().get_data();
        assert!((g[0] + 0.5555555555555556).abs() < 1e-9);
        assert!((g[1] - 0.5555555555555556).abs() < 1e-9);
    }

    #[test]
    fn saturated_prediction_neither_explodes_nor_produces_nan() {
        // the correct class is predicted with probability 0
        let pred   = Var::leaf(tensor![[0.0f32, 1.0]], true);
        let target = Var::leaf(tensor![[1.0f32, 0.0]], false);

        let loss = cross_entropy(&pred, &target);
        loss.backward();

        assert!(loss.value().item().is_finite());
        // clamped values receive no gradient, so nothing blows up
        assert_eq!(pred.grad().get_data(), vec![0.0, 0.0]);

        // an out-of-range input must not produce NaN either
        let logits = Var::leaf(tensor![[2.0f32, -1.0]], true);
        let t = Var::leaf(tensor![[1.0f32, 0.0]], false);
        assert!(cross_entropy(&logits, &t).value().item().is_finite());
    }
}
