use crate::{dataset::TaskLabelType, ndarray::NdArray};
pub mod decision_tree;
pub mod logistic_regression;
pub mod utils;
pub mod linear_regression;
pub mod naive_bayes;
pub mod svm;
pub mod nn;
pub mod knn;

pub trait Model<T: TaskLabelType> {
    // todo implement generics for feature to accept batch or ndarray etc.
    fn predict(&self, feature: &Vec<f32>) -> T;

    fn predict_with_batch(&self, features: &NdArray) -> Vec<T> {
        assert!(features.dim() == 2);
        let mut res = vec![];
        for i in 0..features.shape[0] {
            res.push(self.predict(&features[i].to_vec()))
        }
        res
    }
}