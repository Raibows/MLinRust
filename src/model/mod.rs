use crate::{dataset::TaskLabelType, ndarray::NdArray};
pub mod decision_tree;
pub mod logistic_regression;
pub mod utils;
pub mod linear_regression;
pub mod naive_bayes;
pub mod svm;
pub mod nn;
pub mod knn;
pub mod kmeans;

/// An interface for implementing prediction function of each model
pub trait Model<T: TaskLabelType> {
    /// predict one sample
    /// * return: f32 for regression task, usize for classification task
    fn predict(&self, feature: &Vec<f32>) -> T;

    /// predict a batch
    /// * return: vector of T(f32 or usize)
    fn predict_with_batch(&self, features: &NdArray) -> Vec<T> {
        assert!(features.dim() == 2);
        let mut res = vec![];
        for i in 0..features.shape[0] {
            res.push(self.predict(&features[i].to_vec()))
        }
        res
    }
}